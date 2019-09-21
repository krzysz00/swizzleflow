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
use ndarray::{Array,ArrayViewD,ArrayD,Ix,Axis,IxDyn};
use ndarray::Dimension;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::cmp::{PartialEq,Eq};
use std::hash::{Hash,Hasher};
use std::collections::{HashMap,BTreeSet};

use smallvec::SmallVec;

use itertools::Itertools;

pub type Symbolic = u16;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Value {
    Garbage,
    Symbol(Symbolic),
    // Note: as an invariant, these are sorted
    Fold(Vec<Value>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Value::*;
        match self {
            Symbol(s) => write!(f, "{}", s),
            Fold(v) => {
                write!(f, "∑(")?;
                for e in v {
                    write!(f, "{}, ", e)?;
                }
                write!(f, ")")
            }
            Garbage => write!(f, "⊥")
        }
    }
}

impl Value {
    pub fn fold(mut terms: Vec<Value>) -> Self {
        terms.sort();
        Value::Fold(terms)
    }

    pub fn collect_subterms(&self, store: &mut Vec<BTreeSet<Value>>) -> usize {
        match self {
            Value::Garbage => {
                if store.len() == 0 {
                    store.push(BTreeSet::new())
                }
                store[0].insert(self.clone());
                0
            }
            Value::Symbol(_) => {
                for _ in store.len()..=1 {
                    store.push(BTreeSet::new())
                }
                store[1].insert(self.clone());
                1
            }
            Value::Fold(v) => {
                if let Some(sub_max) = v.iter().map(|e| e.collect_subterms(store)).max() {
                    let ret = sub_max + 1;
                    // Need all indices up through (max of subterms) + 1 to be valid
                    for _ in store.len()..=ret {
                        store.push(BTreeSet::new())
                    }
                    store[ret].insert(self.clone());
                    ret
                } else { 0 }
            }
        }
    }
}

pub type DomRef = usize;

// Invariants on Domains
// - Domain reference 0 is to Garbage, which is the only term at level 0
// - DomRefs are sorted: id1 < id1 implies that the corresponding v1 < v2
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Domain {
    elements: Vec<Value>,
    pub level_bounds: Vec<usize>,
    pub levels: usize,
    element_map: HashMap<Value, DomRef>,
    subterms_of: Vec<Vec<DomRef>>,
    fold_ref_map: HashMap<Vec<DomRef>, DomRef>,
}

impl Domain {
    pub fn new(spec: ArrayViewD<Value>) -> Self {
        let mut store: Vec<BTreeSet<Value>> = Vec::new();
        Value::Garbage.collect_subterms(&mut store);

        for e in spec {
            e.collect_subterms(&mut store);
        }

        let mut level_bounds = vec![0; 1];
        let mut elements: Vec<Value> = Vec::new();
        for level in store.into_iter() {
            // Returned in sorted order
            for e in level.into_iter() {
                elements.push(e)
            }
            level_bounds.push(elements.len())
        }
        let levels = level_bounds.len() - 1;

        let element_map: HashMap<Value, DomRef> =
            elements.iter().cloned().enumerate().map(|(i, e)| (e, i)).collect();

        let mut subterm_sets: Vec<_> = (0..elements.len()).map(|i| {
            let mut v = BTreeSet::<DomRef>::new();
            v.insert(i);
            v }).collect();
        // Must exist due to base symbols
        let complex_min = level_bounds[2];
        for (i, e) in (&elements[complex_min..]).into_iter().enumerate() {
            let i = i + complex_min;
            match e {
                Value::Garbage | Value::Symbol(_) =>
                    panic!("Unexpected base symbol at depth > 1: {}", e),
                Value::Fold(v) => {
                    for subterm in v {
                        // At this point, any term with lower depth has complete subterms
                        let idx = *element_map.get(subterm).unwrap();
                        subterm_sets[i] = &subterm_sets[i] | &subterm_sets[idx];
                    }
                }
            }
        }

        let subterms_of = subterm_sets.into_iter().map(|v| v.into_iter().collect()).collect();

        let fold_ref_map =
            elements.iter().enumerate()
            .filter_map(|(i, e)| {
                if let Value::Fold(v) = e {
                    Some((v.iter().map(|s| *element_map.get(s).unwrap())
                          .collect::<Vec<_>>(), i))
                } else { None }
            }).collect::<HashMap<Vec<DomRef>, DomRef>>();
        Domain { level_bounds, levels, elements, element_map, subterms_of, fold_ref_map }
    }

    pub fn find_value(&self, value: &Value) -> Option<DomRef> {
        self.element_map.get(value).copied()
    }

    pub fn get_value(&self, dom_ref: DomRef) -> &Value {
        &self.elements[dom_ref]
    }

    pub fn subterms_of(&self, dom_ref: DomRef) -> &[DomRef] {
        &self.subterms_of[dom_ref]
    }

    pub fn size(&self) -> usize {
        self.elements.len()
    }

    pub fn num_symbols(&self) -> usize {
        self.level_bounds[2] - self.level_bounds[1]
    }

    // The set of indices must be sorted
    pub fn find_fold(&self, elems: &[DomRef]) -> Option<DomRef> {
        self.fold_ref_map.get(elems).copied()
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
    pub level: usize,
    pub name: String
}

impl Clone for ProgState<'_> {
    fn clone(&self) -> Self {
        ProgState { domain: self.domain,
                    state: self.state.clone(),
                    inv_state: self.inv_state.clone(),
                    level: self.level,
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
    fn making_inverse(domain: &'d Domain, state: ArrayD<DomRef>, level: usize,
                      name: String) -> Self {
        let mut inverse: Vec<Vec<IxDyn>> = (0..domain.size()).map(|_| Vec::with_capacity(1)).collect();
        for (idx, elem) in state.indexed_iter() {
            for s in domain.subterms_of(*elem) {
                inverse[*s].push(idx.clone())
            }
        }
        Self { domain, state, level, name, inv_state: inverse }
    }

    pub fn new(domain: &'d Domain, state: ArrayD<DomRef>,
               level: usize, name: impl Into<String>) -> Self {
        Self::making_inverse(domain, state, level, name.into())
    }

    pub fn new_from_spec(domain: &'d Domain, state: ArrayD<Value>, name: impl Into<String>) -> Option<Self> {
        let ref_vec: Option<Vec<DomRef>> = state.as_slice().unwrap()
            .iter().map(|v| domain.find_value(v))
            .collect();
        let ref_vec = ref_vec?;
        let ref_mat = ArrayD::from_shape_vec(state.shape(), ref_vec).unwrap();
        let level = domain.levels - 1;
        Some(Self::new(domain, ref_mat, level, name))
    }

    pub fn linear(domain: &'d Domain, shape: &[Ix]) -> Self {
        let array = (domain.level_bounds[1]..domain.level_bounds[2]).collect();
        Self::making_inverse(domain, Array::from_shape_vec(shape, array).unwrap(),
                             1, "id".to_owned())
    }

    pub fn gather_by(&self, gather: &Gather) -> Self {
        let axis_num = Axis(gather.data.ndim() - 1);
        // Read off the edge to get the "garbage" value
        let array = gather.data.map_axis(axis_num,
                                         move |v| self.state.get(v.into_slice().unwrap())
                                         .copied().unwrap_or(0));

        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        Self::making_inverse(self.domain, array, self.level, name)
    }

    pub fn gather_fold_by(&self, gather: &Gather) -> Option<Self> {
        let index_axis = gather.data.ndim() - 1;
        let fold_axis = index_axis - 1;
        let index_len = gather.data.shape()[index_axis];

        let mut view = gather.data.view();
        if !view.merge_axes(Axis(fold_axis), Axis(index_axis)) {
            panic!("Axes that should've statically been mergeable weren't");
        }

        let mut elements: SmallVec<[DomRef; 6]> = SmallVec::with_capacity(index_len);
        let result: Option<Vec<DomRef>> =
            view.genrows().into_iter()
            .map(|data| {
                let data = data.as_slice().unwrap();
                elements.extend(
                    data.chunks(index_len)
                    .map(|idx| self.state.get(idx)
                         .copied().unwrap_or(0))
                    // Drop garbage values
                    .filter(|v| *v != 0));
                elements.sort_unstable();
                let ret = self.domain.find_fold(&elements);
                elements.clear();
                ret
            }).collect();
        let result = result?;
        let array = ArrayD::from_shape_vec(&gather.data.shape()[0..fold_axis],
                                           result).unwrap();

        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        Some(Self::making_inverse(self.domain, array, self.level + 1, name))
    }

    pub fn merge(states: &[&ProgState<'d>]) -> Self {
        let axis_number = Axis(states[0].state.ndim());
        let views: SmallVec<[ArrayViewD<DomRef>; 6]> =
            states.into_iter().map(|s| s.state.view().insert_axis(axis_number))
            .collect();
        let array = ndarray::stack(axis_number, &views).unwrap();
        let level = states.iter().map(|s| s.level).max().unwrap();
        let name = format!("merge[{}]",
                               states.iter().map(|s| s.name.clone()).join(","));
        Self::making_inverse(states[0].domain, array, level, name)
    }

    pub fn merge_folding(states: &[&ProgState<'d>]) -> Option<Self> {
        let slices: SmallVec<[&[DomRef]; 6]> =
            states.into_iter().map(|s| s.state.as_slice().unwrap())
            .collect();
        let len = slices[0].len();
        let domain = states[0].domain;
        let mut elements: SmallVec<[DomRef; 6]> = SmallVec::with_capacity(len);
        let result: Option<Vec<DomRef>> = (0..len).map(
            |i| {
                elements.extend(slices.iter().map(|s| s[i]));
                elements.sort_unstable();
                let ret = domain.find_fold(&elements);
                elements.clear();
                ret
            }).collect();
        let result = result?;
        let array = ArrayD::from_shape_vec(states[0].state.shape(),
                                           result).unwrap();
        let level = states.iter().map(|s| s.level).max().unwrap() + 1;
        let name = format!("merge_fold[{}]",
                           states.iter().map(|s| s.name.clone()).join(","));
        Some(Self::making_inverse(domain, array, level, name))
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
