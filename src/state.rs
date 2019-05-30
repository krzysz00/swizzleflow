// Copyright (C) 2019 Krzysztof Drewniak et al.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
use ndarray::{Array,ArrayD,Ix,Axis,IxDyn};
use ndarray::Dimension;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::cmp::{PartialEq,Eq};
use std::hash::{Hash,Hasher};

pub type Symbolic = u16;
pub type ProgValue = Symbolic;

// Ok, the general pruning rule is this:
// For a symbolic value v, let Loc(s, v), be the set of locations where v is placed in state s
// Suppose we want to prune from state c to state t
// There's a general path with value v, called c ->'(v) t, if, in the pruning matrix, for each l in Loc(v, t), there is a l' in Loc(v, c) such that l' -> l
// A value can continue, if for each pair of values v1, v2, we have c ->'(v1) t and c ->'(v2) t
// Program states
#[derive(Clone, Debug)]
pub struct ProgState {
    pub(crate) state: ArrayD<ProgValue>,
    // Note: IxDyn is basically SmallVec, though that's not obvious anywhere
    pub(crate) inv_state: Vec<Vec<IxDyn>>,
    // The value domain_max is the "trash" value
    pub domain_max: Symbolic,
    pub name: String
}

impl PartialEq for ProgState {
    fn eq(&self, other: &ProgState) -> bool {
        self.domain_max == other.domain_max && self.state == other.state
    }
}

impl Eq for ProgState {}

impl Hash for ProgState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.domain_max.hash(state);
        self.state.hash(state);
    }
}

impl ProgState {
    fn making_inverse(domain_max: Symbolic, state: ArrayD<ProgValue>, name: String) -> Self {
        let mut inverse: Vec<Vec<IxDyn>> = (0..domain_max).map(|_| Vec::with_capacity(1)).collect();
        for (idx, elem) in state.indexed_iter().filter(|&(_, x)| *x < domain_max) {
            inverse[*elem as usize].push(idx.clone())
        }
        Self { domain_max: domain_max, state: state, name: name, inv_state: inverse }
    }

    pub fn new(domain_max: Symbolic, state: ArrayD<ProgValue>, name: impl Into<String>) -> Self {
        Self::making_inverse(domain_max, state, name.into())
    }

    pub fn linear(shape: &[Ix]) -> Self {
        let domain_max: Ix = shape.iter().product();
        let domain_max = domain_max as Symbolic;
        let array = (0..domain_max).collect();
        Self::making_inverse(domain_max, Array::from_shape_vec(shape, array).unwrap(), "id".to_owned())
    }

    pub fn gather_by(&self, gather: &Gather) -> Self {
        let axis_num = Axis(gather.data.ndim() - 1);
        // Read off the edge to get the "garbage" value
        let array = gather.data.map_axis(axis_num,
                                         move |v| self.state.get(v.into_slice().unwrap())
                                         .copied().unwrap_or(self.domain_max));
        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        Self::making_inverse(self.domain_max, array, name)
    }
}

impl Display for ProgState {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.state)
    }
}

// Gather operator creation
fn inc_slice(slice: &mut [Ix], bound: &[Ix]) -> bool {
    if slice.len() != bound.len() {
        panic!("Bound and slice must be the same length")
    }
    let mut success = false;
    for (slice, bound) in slice.iter_mut().zip(bound.iter()).rev() {
        *slice += 1;
        if *slice == *bound {
            *slice = 0;
        }
        else {
            success = true;
            break;
        }
    }
    success
}

#[derive(Clone, Debug)]
pub struct Gather {
    pub data: ArrayD<Ix>,
    pub name: String,
}

impl PartialEq for Gather {
    fn eq(&self, other: &Gather) -> bool {
        self.data == other.data
    }
}

impl Eq for Gather {}

impl Hash for Gather {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Gather {
    pub fn new<F>(source_dim: usize, dest_shape: &[Ix],
                  builder: F, name: impl Into<String>) -> Self
    where F: Fn(&[Ix], &mut Vec<Ix>) {
        let dim_prod: usize = dest_shape.iter().product();
        let length = source_dim * dim_prod;
        let mut array = Vec::with_capacity(length);

        let mut index: IxDyn = IxDyn::zeros(dest_shape.len());
        while {
            builder(index.slice(), &mut array);
            inc_slice(index.slice_mut(), dest_shape)
        } {} // Black magic for a do-while loop

        let mut shape = dest_shape.to_owned();
        shape.push(source_dim);
        let array = Array::from_shape_vec(shape, array).unwrap();
        Self { data: array,  name: name.into() }
    }
}

impl fmt::Display for Gather {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.data)
    }
}
