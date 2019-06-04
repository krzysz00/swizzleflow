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

use hashbrown::HashMap;

pub type Symbolic = u16;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    Symbol(Symbolic),
    Garbage,
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Value::*;
        match self {
            Symbol(s) => write!(f, "{}", s),
            Garbage => write!(f, "⊥")
        }
    }
}

pub type DomRef = usize;

// It's an invariant of this structure that references [0..domain_max)
// contain the symbols [0..domain_max), in order, and than reference domain_max
// contains Garbage
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Domain {
    elements: Vec<Value>,
    element_map: HashMap<Value, DomRef>,
    symbols_in: Vec<Vec<DomRef>>,
    pub symbol_max: usize,
}

impl Domain {
    pub fn new(symbol_max: usize) -> Self {
        let mut elements: Vec<Value> = (0..symbol_max).into_iter()
            .map(|x| Value::Symbol(x as Symbolic)).collect();
        elements.push(Value::Garbage);

        let mut element_map: HashMap<Value, DomRef> = (0..symbol_max).into_iter()
            .map(|x| (Value::Symbol(x as Symbolic), x)).collect();
        element_map.insert(Value::Garbage, symbol_max);

        let mut symbols_in: Vec<Vec<DomRef>> = (0..symbol_max).into_iter()
            .map(|x| vec![x]).collect();
        symbols_in.push(vec![]); // Garbage

        Domain { elements, element_map, symbols_in, symbol_max }
    }

    pub fn find_value(&self, value: &Value) -> Option<DomRef> {
        self.element_map.get(value).copied()
    }

    pub fn get_value(&self, dom_ref: DomRef) -> &Value {
        &self.elements[dom_ref]
    }

    pub fn symbols_in(&self, dom_ref: DomRef) -> &[DomRef] {
        &self.symbols_in[dom_ref]
    }
}
// Ok, the general pruning rule is this:
// For a symbolic value v, let Loc(s, v), be the set of locations where v is placed in state s
// Suppose we want to prune from state c to state t
// There's a general path with value v, called c ->'(v) t, if, in the pruning matrix, for each l in Loc(v, t), there is a l' in Loc(v, c) such that l' -> l
// A value can continue, if for each pair of values v1, v2, we have c ->'(v1) t and c ->'(v2) t
// Program states
#[derive(Debug)]
pub struct ProgState<'d> {
    pub domain: &'d Domain,
    pub(crate) state: ArrayD<DomRef>,
    // Note: IxDyn is basically SmallVec, though that's not obvious anywhere
    pub(crate) inv_state: Vec<Vec<IxDyn>>,
    pub name: String
}

impl Clone for ProgState<'_> {
    fn clone(&self) -> Self {
        ProgState { domain: self.domain,
                    state: self.state.clone(),
                    inv_state: self.inv_state.clone(),
                    name: self.name.clone() }
    }
}
impl PartialEq for ProgState<'_> {
    fn eq(&self, other: &ProgState) -> bool {
        self.state == other.state
    }
}

impl Eq for ProgState<'_> {}

impl Hash for ProgState<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}

impl<'d> ProgState<'d> {
    fn making_inverse(domain: &'d Domain, state: ArrayD<DomRef>, name: String) -> Self {
        let mut inverse: Vec<Vec<IxDyn>> = (0..domain.symbol_max).map(|_| Vec::with_capacity(1)).collect();
        for (idx, elem) in state.indexed_iter() {
            for s in domain.symbols_in(*elem) {
                inverse[*s].push(idx.clone())
            }
        }
        Self { domain: domain , state: state, name: name, inv_state: inverse }
    }

    pub fn new(domain: &'d Domain, state: ArrayD<DomRef>, name: impl Into<String>) -> Self {
        Self::making_inverse(domain, state, name.into())
    }

    pub fn new_from_spec(domain: &'d Domain, state: ArrayD<Value>, name: impl Into<String>) -> Option<Self> {
        let ref_vec: Option<Vec<DomRef>> = state.as_slice().unwrap()
            .into_iter().map(|v| domain.find_value(v))
            .collect();
        let ref_vec = ref_vec?;
        let ref_mat = ArrayD::from_shape_vec(state.shape(), ref_vec).unwrap();
        Some(Self::making_inverse(domain, ref_mat, name.into()))
    }

    pub fn linear(domain: &'d Domain, shape: &[Ix]) -> Self {
        let array = (0..domain.symbol_max).collect();
        Self::making_inverse(domain, Array::from_shape_vec(shape, array).unwrap(), "id".to_owned())
    }

    pub fn gather_by(&self, gather: &Gather) -> Self {
        let axis_num = Axis(gather.data.ndim() - 1);
        // Read off the edge to get the "garbage" value
        let dm = self.domain.symbol_max;
        let array = gather.data.map_axis(axis_num,
                                         move |v| self.state.get(v.into_slice().unwrap())
                                         .copied().unwrap_or(dm));
        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        Self::making_inverse(self.domain, array, name)
    }
}

impl Display for ProgState<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.state.mapv(|r| self.domain.get_value(r)))
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
