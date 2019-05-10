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

pub type Symbolic = u16;
pub type ProgValue = Symbolic;

// Ok, the general pruning rule is this:
// For a symbolic value v, let Loc(s, v), be the set of locations where v is placed in state s
// Suppose we want to prune from state c to state t
// There's a general path with value v, called c ->'(v) t, if, in the pruning matrix, for each l in Loc(v, t), there is a l' in Loc(v, c) such that l' -> l
// A value can continue, if for each pair of values v1, v2, we have c ->'(v1) t and c ->'(v2) t
// Program states
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgState {
    state: ArrayD<ProgValue>,
    // Note: IxDyn is basically SmallVec, though that's not obvious anywhere
    inv_state: Vec<Vec<IxDyn>>,
    pub domain_max: Symbolic,
    pub name: String
}

impl ProgState {
    fn making_inverse(domain_max: Symbolic, state: ArrayD<ProgValue>, name: String) -> Self {
        let mut inverse: Vec<Vec<IxDyn>> = (0..domain_max).map(|_| Vec::with_capacity(1)).collect();
        for (idx, elem) in state.indexed_iter() {
            inverse[*elem as usize].push(idx.clone())
        }
        Self { domain_max: domain_max, state: state, name: name, inv_state: inverse }
    }

    pub fn new(domain_max: Symbolic, state: ArrayD<ProgValue>, name: impl Into<String>) -> Self {
        Self::making_inverse(domain_max, state, name.into())
    }

    pub fn linear(domain_max: Symbolic, shape: &[Ix]) -> Self {
        let array = (0..domain_max).collect();
        Self::making_inverse(domain_max, Array::from_shape_vec(shape, array).unwrap(), "id".to_owned())
    }

    pub fn gather_by(&self, gather: &Gather) -> Self {
        let axis_num = Axis(gather.data.ndim() - 1);
        let array = gather.data.map_axis(axis_num,
                                         move |v| self.state[v.into_slice().unwrap()]);
        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        Self::making_inverse(self.domain_max, array, name)
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

impl fmt::Display for ProgState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.state)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Gather {
    pub data: ArrayD<Ix>,
    pub name: String,
}

impl Gather {
    pub fn new<F>(source_dim: usize, dest_shape: &[Ix],
                  builder: F, name: impl Into<String>) -> Self
    where F: Fn(&[Ix], &mut Vec<Ix>) {
        let dim_prod: usize = dest_shape.iter().product();
        let length = source_dim * dim_prod;
        let mut array = Vec::with_capacity(length);

        let mut index: IxDyn = IxDyn::zeros(source_dim);
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
