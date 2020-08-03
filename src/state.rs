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
use crate::misc::in_bounds;

use ndarray::{Array,ArrayViewD,ArrayD,Ix,Axis};
use ndarray::{Dimension,Zip,FoldWhile};

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
        if terms.is_empty() {
            Value::Garbage
        }
        else {
            terms.sort();
            Value::Fold(terms)
        }
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
                } else { // Empty folds
                    if store.len() == 0 {
                        store.push(BTreeSet::new())
                    }
                    store[0].insert(self.clone());
                    0
                }
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
    element_map: HashMap<Value, DomRef>,
    subterms_of: Vec<Vec<DomRef>>,
    imm_superterms: Vec<Vec<DomRef>>,
    imm_subterms: Vec<BTreeSet<DomRef>>,
    fold_ref_map: HashMap<Vec<DomRef>, DomRef>,
}

impl Domain {
    pub fn new(spec: ArrayViewD<Value>) -> Self {
        let mut store: Vec<BTreeSet<Value>> = Vec::new();
        Value::Garbage.collect_subterms(&mut store);

        for e in spec {
            e.collect_subterms(&mut store);
        }

        let mut elements: Vec<Value> = Vec::new();
        for level in store.into_iter() {
            // Returned in sorted order
            for e in level.into_iter() {
                elements.push(e)
            }
        }

        let element_map: HashMap<Value, DomRef> =
            elements.iter().cloned().enumerate().map(|(i, e)| (e, i)).collect();

        let mut subterm_sets: Vec<_> = (0..elements.len()).map(|i| {
            let mut v = BTreeSet::<DomRef>::new();
            v.insert(i);
            v }).collect();
        let mut imm_subterms: Vec<BTreeSet<DomRef>> =
            (0..elements.len()).map(|_| BTreeSet::new()).collect();
        let mut immediate_superterms: Vec<BTreeSet<DomRef>> =
            (0..elements.len()).map(|_| BTreeSet::new()).collect();
        // Must exist due to base symbols
        for (i, e) in elements.iter().enumerate() {
            match e {
                Value::Garbage | Value::Symbol(_) =>
                    (),
                Value::Fold(v) => {
                    for subterm in v {
                        // At this point, any term with lower depth has complete subterms
                        let idx = *element_map.get(subterm).unwrap();
                        subterm_sets[i] = &subterm_sets[i] | &subterm_sets[idx];
                        imm_subterms[i].insert(idx);
                        immediate_superterms[idx].insert(i);
                    }
                }
            }
        }

        let subterms_of = subterm_sets.into_iter().map(|v| v.into_iter().collect()).collect();
        let imm_superterms = immediate_superterms.into_iter().map(|v| v.into_iter().collect()).collect();

        let mut fold_ref_map =
            elements.iter().enumerate()
            .filter_map(|(i, e)| {
                if let Value::Fold(v) = e {
                    Some((v.iter().map(|s| *element_map.get(s).unwrap())
                          .collect::<Vec<_>>(), i))
                } else { None }
            }).collect::<HashMap<Vec<DomRef>, DomRef>>();
        // Semantics of empty fold
        fold_ref_map.insert(vec![], *element_map.get(&Value::Garbage).unwrap());
        Domain { elements, element_map, subterms_of, fold_ref_map,
                 imm_subterms, imm_superterms }
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

    // The set of indices must be sorted
    pub fn find_fold(&self, elems: &[DomRef]) -> Option<DomRef> {
        self.fold_ref_map.get(elems).copied()
    }

    pub fn subterms_all_within(&self, elem: DomRef, expected: &BTreeSet<DomRef>) -> bool {
        self.imm_subterms[elem].is_subset(expected)
    }

    pub fn imm_superterms(&self, elem: DomRef) -> &[DomRef] {
        &self.imm_superterms[elem]
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
    // We store row-major indices into matrix dimensions for effeciency
    pub(crate) inv_state: Vec<Vec<Ix>>,
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
    fn making_inverse(domain: &'d Domain, state: ArrayD<DomRef>,
                      name: String) -> Self {
        let mut inverse: Vec<Vec<Ix>> = (0..domain.size()).map(|_| Vec::with_capacity(16)).collect();
        let slice = state.as_slice().unwrap();
        for (idx, elem) in slice.iter().copied().enumerate() {
            for s in domain.subterms_of(elem) {
                inverse[*s].push(idx)
            }
        }
        Self { domain, state, name, inv_state: inverse }
    }

    pub fn new(domain: &'d Domain, state: ArrayD<DomRef>,
               name: impl Into<String>) -> Self {
        Self::making_inverse(domain, state, name.into())
    }

    pub fn new_from_spec(domain: &'d Domain, state: ArrayD<Value>,
                         name: impl Into<String>) -> Option<Self> {
        let ref_vec: Option<Vec<DomRef>> = state.as_slice().unwrap()
            .iter().map(|v| domain.find_value(v))
            .collect();
        let ref_vec = ref_vec?;
        let ref_mat = ArrayD::from_shape_vec(state.shape(), ref_vec).unwrap();
        Some(Self::new(domain, ref_mat, name))
    }

    pub fn linear(domain: &'d Domain, start: Ix, shape: &[Ix]) -> Self {
        let shape_len: usize = shape.iter().copied().product();
        let array = (start..start+shape_len).collect();
        Self::making_inverse(domain, Array::from_shape_vec(shape, array).unwrap(),
                             "id".to_owned())
    }

    pub fn gather_by(&self, gather: &Gather) -> Self {
        // Read off the edge to get the "garbage" value
        let slice = self.state.as_slice().unwrap();
        let array = gather.data.mapv(move |v| if v < 0 { 0 }
                                     else { slice.get(v as usize).copied().unwrap_or(0) });
        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        Self::making_inverse(self.domain, array, name)
    }

    pub fn gather_fold_by(&self, gather: &Gather) -> Option<Self> {
        let mut elements: SmallVec<[DomRef; 6]> =
            SmallVec::with_capacity(gather.data.shape()[gather.data.ndim()-1]);
        let slice = self.state.as_slice().unwrap();
        let result: Option<Vec<DomRef>> =
            gather.data.genrows().into_iter()
            .map(|data| {
                let data = data.as_slice().unwrap();
                elements.extend(
                    data.iter().copied()
                        .filter_map(|idx|
                                    if idx < 0 { None }
                                    else { slice.get(idx as usize).copied() })
                        .filter(|x| *x != 0));
                elements.sort_unstable();
                let ret = self.domain.find_fold(&elements);
                elements.clear();
                ret
            }).collect();
        let result = result?;
        let array = ArrayD::from_shape_vec(&gather.data.shape()[0..gather.data.ndim()-1],
                                           result).unwrap();

        let mut name = self.name.to_owned();
        name.push_str(";");
        name.push_str(&gather.name);
        name.push_str("[Σ]");
        Some(Self::making_inverse(self.domain, array, name))
    }

    pub fn stack(states: &[&ProgState<'d>]) -> Self {
        let axis_number = Axis(states[0].state.ndim());
        let views: SmallVec<[ArrayViewD<DomRef>; 6]> =
            states.into_iter().map(|s| s.state.view().insert_axis(axis_number))
            .collect();
        let array = ndarray::stack(axis_number, &views).unwrap();
        let name = format!("stack[{}]",
                               states.iter().map(|s| s.name.clone()).join(","));
        Self::making_inverse(states[0].domain, array, name)
    }

    pub fn stack_folding(states: &[&ProgState<'d>]) -> Option<Self> {
        let slices: SmallVec<[&[DomRef]; 6]> =
            states.into_iter().map(|s| s.state.as_slice().unwrap())
            .collect();
        let len = slices[0].len();
        let domain = states[0].domain;
        let mut elements: SmallVec<[DomRef; 6]> = SmallVec::with_capacity(len);
        let result: Option<Vec<DomRef>> = (0..len).map(
            |i| {
                elements.extend(slices.iter().map(|s| s[i]).filter(|v| *v != 0));
                elements.sort_unstable();
                let ret = domain.find_fold(&elements);
                elements.clear();
                ret
            }).collect();
        let result = result?;
        let array = ArrayD::from_shape_vec(states[0].state.shape(),
                                           result).unwrap();
        let name = format!("stack_fold[{}]",
                           states.iter().map(|s| s.name.clone()).join(","));
        Some(Self::making_inverse(domain, array, name))
    }
}

impl Display for ProgState<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.state.mapv(|r| self.domain.get_value(r)))
    }
}

// Gather operator creation
pub fn to_ix(index: &[Ix], shape: &[Ix]) -> Ix {
    return index.iter().zip(shape.iter())
        .fold(0, |acc, (i, b)| acc * b + i)
}

pub type OptIx = isize;
pub fn to_opt_ix(index: &[Ix], shape: &[Ix]) -> OptIx {
    if in_bounds(index, shape) {
        to_ix(index, shape) as isize
    }
    else {
        -1
    }
}

#[derive(Clone, Debug)]
pub struct Gather {
    pub data: ArrayD<OptIx>,
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
    pub fn new<F>(dest_shape: &[Ix],
                  builder: F, name: impl Into<String>) -> Self
    where F: Fn(&[Ix]) -> isize {
        let array = ArrayD::from_shape_fn(dest_shape, |v| builder(v.slice()));
        Self { data: array,  name: name.into() }
    }

    pub fn new_raw(data: ArrayD<OptIx>, name: String) -> Self {
        Self { data, name }
    }

    pub fn new_blank(dest_shape: &[Ix], name: Option<String>) -> Self {
        let array = ArrayD::from_elem(dest_shape, -1);
        Self { data: array, name: name.unwrap_or("forget".into()) }
    }

    pub fn merge_with(&mut self, other: &Gather) -> bool {
        let ret =
            Zip::from(self.data.view_mut()).and(other.data.view())
            .fold_while((), |_, s, o| {
                if *s >= 0 {
                    if *o < 0 || s == o {
                        FoldWhile::Continue(())
                    }
                    else {
                        FoldWhile::Done(())
                    }
                }
                else {
                    *s = *o;
                    FoldWhile::Continue(())
                }
            });
        !ret.is_done() // No early return -> successful merge
    }

    pub fn is_identity(&self) -> bool {
        self.data.as_slice().unwrap()
            .iter().copied().enumerate()
            .all(|(i, v)| i as isize == v)
    }

    pub fn min_max_copies(&self, input_bound: usize, mins_out: &[u32], maxs_out: &[u32],
                          fold_factor: Option<usize>) -> (Vec<u32>, Vec<u32>) {
        let signed_bound = input_bound as isize;
        let mut mins_ret = vec![0; input_bound];
        let mut maxs_ret = vec![0; input_bound];

        for (i, e) in self.data.iter().copied().enumerate().filter(|&(_, e)| e >= 0 && e < signed_bound) {
            let e = e as usize;
            let i = if let Some(fold) = fold_factor { i / fold } else { i };
            mins_ret[e] += mins_out[i];
            maxs_ret[e] += maxs_out[i];
        }
        (mins_ret, maxs_ret)
    }
}

impl fmt::Display for Gather {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.data)
    }
}
