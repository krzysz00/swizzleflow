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
use crate::misc::{in_bounds,ShapeVec};

use ndarray::{Array,ArrayViewD,ArrayD,Ix,Axis};
use ndarray::{Dimension,Zip,FoldWhile};

use std::borrow::Cow;
use std::cmp::{PartialEq,Eq};
use std::collections::{HashMap,BTreeSet};
use std::fmt;
use std::fmt::{Display,Formatter};
use std::hash::{Hash,Hasher};
use std::num::{NonZeroUsize};

use smallvec::SmallVec;

use itertools::Itertools;

pub type Symbolic = u16;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Value {
    Empty,
    NotInDomain,
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
            Empty => write!(f, "∅"),
            NotInDomain => write!(f, "⊥"),
        }
    }
}

impl Value {
    pub fn fold(mut terms: Vec<Value>) -> Self {
        if terms.is_empty() {
            Value::Empty
        }
        else {
            terms.sort();
            Value::Fold(terms)
        }
    }

    pub fn collect_subterms(&self, store: &mut Vec<BTreeSet<Value>>) -> usize {
        match self {
            Value::Empty => {
                if store.len() == 0 {
                    store.push(BTreeSet::new())
                }
                store[0].insert(self.clone());
                0
            }
            Value::NotInDomain => {
                if store.len() == 0 {
                    store.push(BTreeSet::new())
                }
                // Defensive coding
                if !store[0].contains(&Value::Empty) {
                    store[0].insert(Value::Empty);
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
pub const ZERO: DomRef = 0;
pub const NOT_IN_DOMAIN: DomRef = 1;

// Invariants on Domains
// - Domain reference 0 is to Empty, reference 1 is NotInDomain
// - these are the only terms at level 0
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
        Value::Empty.collect_subterms(&mut store);
        Value::NotInDomain.collect_subterms(&mut store);

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
                Value::Empty | Value::NotInDomain | Value::Symbol(_) =>
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
        fold_ref_map.insert(vec![], *element_map.get(&Value::Empty).unwrap());
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
pub struct ProgState<'d, 't> {
    pub domain: &'d Domain,
    pub(crate) state: Vec<Cow<'t, Option<ArrayD<DomRef>>>>,
    // We store row-major indices into matrix dimensions for effeciency
    pub(crate) inv_state: Vec<Vec<(Ix, Ix)>>,
    pub name: String
}

impl Clone for ProgState<'_, '_> {
    fn clone(&self) -> Self {
        ProgState { domain: self.domain,
                    state: self.state.clone(),
                    inv_state: self.inv_state.clone(),
                    name: self.name.clone() }
    }
}
impl PartialEq for ProgState<'_, '_> {
    fn eq(&self, other: &ProgState) -> bool {
        self.state == other.state
    }
}

impl Eq for ProgState<'_, '_> {}

impl Hash for ProgState<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operation {
    pub fns: Vec<Gather>,
    pub fns_summary: Option<Gather>,

    pub fold_len: Option<NonZeroUsize>,

    pub in_shapes: Vec<ShapeVec>,
    pub in_bounds: Vec<usize>,
    // Already accounts for fold
    pub out_shape: ShapeVec,

    pub var: String,
    pub arg_string: String,

    pub in_lanes: Vec<usize>,
    pub out_lane: usize,
    pub drop_lanes: Vec<usize>,
}


impl<'d, 't> ProgState<'d, 't> {
    fn making_inverse(domain: &'d Domain, state: Vec<Cow<'t, Option<ArrayD<DomRef>>>>,
                      name: String) -> Self {
        let mut inverse: Vec<Vec<Ix>> = (0..domain.size()).map(|_| Vec::with_capacity(16)).collect();
        for (lane, data) in state.iter().enumerate().filter_map(|(i, x)| x.as_ref().map(|v| (i, v))) {
            let slice = data.as_slice().unwrap();
            for (idx, elem) in slice.iter().copied().enumerate() {
                for s in domain.subterms_of(elem) {
                    inverse[*s].push((lane, idx))
                }
            }
        }
        Self { domain, state, name, inv_state: inverse }
    }

    pub fn new(domain: &'d Domain, state: Vec<Option<ArrayD<DomRef>>>,
               name: impl Into<String>) -> Self {
        let state = state.into_iter().map(|x| Cow::Owned(x));
        Self::making_inverse(domain, state, name.into())
    }

    pub fn new_from_spec(domain: &'d Domain, vars: Vec<Option<ArrayD<Value>>>,
                         name: impl Into<String>) -> Self {
        let ref_state = vars.into_iter()
            .map(|mv| Cow::Owned(ms.map(|var|
            var.map(|v| domain.find_value(v).unwrap_or(NOT_IN_DOMAIN)))));
        Self::making_inverse(domain, ref_state, name.into())
    }

    pub fn linear(domain: &'d Domain, start: Ix, shape: &[Ix]) -> Self {
        let shape_len: usize = shape.iter().copied().product();
        let array = (start..start+shape_len).collect();
        Self::making_inverse(domain,
                             vec![Cow::Owned(Some(Array::from_shape_vec(shape, array).unwrap()))],
                             "id".to_owned())
    }

    fn emplace<'u: 't>(&self, value: ArrayD<DomRef>, op: &Operation,
                       gather_name: &str) -> ProgState<'d, 'u> {
        let mut state = self.state.clone();
        state[op.out_lane] = Cow::Owned(Some(value));
        for l in op.drop_lanes {
            state[l] = Cow::Owned(None);
        }
        let name = format!(" {} = {}{};", op.name, gather_name, op.arg_string);
        Self::making_inverse(self.domain, state, name)
    }

    pub fn gather_by<'u: 't>(&self, gather: &Gather, op: &Operation)
                             -> ProgState<'d, 'u> {
        // Read off the edge to get \bot
        // Read past the arguments to get 0
        let slices: SmallVec<[&[usize]; 3]> =
            op.in_lanes.iter().copied().map(|l| self.state[l]
                                            .as_slice().unwrap())
            .collect();
        // 0 = 0
        // 1 = \bot
        let array = gather.data.mapv(move |(arg, idx)|
                                     slices.get(arg)
                                     .map(|s|
                                          if idx < 0 { NOT_IN_DOMAIN }
                                          else { s.get(idx as usize).copied().unwrap_or(NOT_IN_DOMAIN) })
                                     .unwrap_or(ZERO));
        self.emplace(array, op, gather.name)
    }

    #[inline]
    fn fold_find(&self, data: &[(Ix, OptIx)],
                 elements: &mut SmallVec<[DomRef; 6]>,
                 slices: &[&[usize]]) -> DomRef {
        for (arg, idx) in data.iter().copied() {
            if let Some(s) = slices.get(arg).as_ref() {
                let val =
                    if idx < 0 { NOT_IN_DOMAIN }
                    else { s.get(idx as usize).copied().unwrap_or(NOT_IN_DOMAIN) };
                elements.push(val);
            }
        }
        elements.sort_unstable();
        let ret = self.domain.find_fold(&elements).unwrap_or(NOT_IN_DOMAIN);
        elements.clear();
        ret
    }

    pub fn gather_fold_by<'u: 't>(&self, gather: &Gather, op: &Operation)
                                  -> ProgState<'d, 'u> {
        let slices: SmallVec<[&[usize]; 3]> =
            op.in_lanes.iter().copied().map(|l| self.state[l]
                                            .as_slice().unwrap())
            .collect();

        let mut elements: SmallVec<[DomRef; 6]> =
            SmallVec::with_capacity(gather.data.shape()[gather.data.ndim()-1]);
        let axis = Axis(gather.data.ndims() - 1);
        let slice = self.state.as_slice().unwrap();
        let array: ArrayD<DomRef> =
            gather.data.map_axis(
                axis,
                |data| {
                    let data = data.as_slice().unwrap();
                    self.fold_find(data, &mut elements, slices.as_slice())
                });
        self.emplace(array, op, gather.name)
    }
}

impl Display for ProgState<'_, '_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name,
               self.state.iter()
               .filter_map(|s| s.as_ref().map(
                   |a| a.mapv(|r| self.domain.get_value(r)))
               .join("")))
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
    pub data: ArrayD<(Ix, OptIx)>,
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
    where F: Fn(&[Ix]) -> (Ix, OptIx) {
        let array = ArrayD::from_shape_fn(dest_shape, |v| builder(v.slice()));
        Self { data: array,  name: name.into() }
    }

    pub fn new_raw(data: ArrayD<(Ix, OptIx)>, name: String) -> Self {
        Self { data, name }
    }

    pub fn new_blank(dest_shape: &[Ix], name: Option<String>) -> Self {
        let array = ArrayD::from_elem(dest_shape, (1, 0));
        Self { data: array, name: name.unwrap_or("forget".into()) }
    }

    pub fn merge_with(&mut self, other: &Gather, n_args: usize) -> bool {
        let ret =
            Zip::from(self.data.view_mut()).and(other.data.view())
            .fold_while((), |_, s, o| {
                if *s.0 >= n_args {
                    *s = *o;
                    FoldWhile::Continue(())
                }
                else {
                    if s == o || *o.0 >= n_args {
                        FoldWhile::Continue(())
                    }
                    else {
                        FoldWhile::Done(())
                    }
                }
            });
        !ret.is_done() // No early return -> successful merge
    }

    pub fn is_identity(&self) -> bool {
        self.data.as_slice().unwrap()
            .iter().copied().enumerate()
            .all(|(i, (a, v))| i as isize == v || a == 0)
    }

    pub fn min_max_copies(&self, input_bounds: &[usize],
                          mins_out: &[u32], maxs_out: &[u32],
                          fold_factor: Option<usize>) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let n_inputs = input_bounds.len();
        let mut mins_ret = input_bounds.iter().copied().map(|b| vec![0; b]);
        let mut maxs_ret = input_bounds.iter().copied().map(|b| vec![0; b]);

        for (i, (a, e)) in self.data.iter().copied().enumerate()
            .filter(|&(_, (a, e))| a < n_inputs && e >= 0 && e < (input_bounds[a] as isize)) {
            let e = e as usize;
            let i = if let Some(fold) = fold_factor { i / fold } else { i };
            mins_ret[a][e] += mins_out[i];
            maxs_ret[a][e] += maxs_out[i];
        }
        (mins_ret, maxs_ret)
    }
}

impl fmt::Display for Gather {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:\n{}", self.name, self.data)
    }
}
