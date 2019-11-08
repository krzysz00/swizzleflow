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
pub mod swizzle;
pub mod select;
pub mod load;

use crate::errors::*;

use crate::state::{Gather,to_opt_ix};
use crate::transition_matrix::{TransitionMatrix};
use crate::misc::ShapeVec;

use smallvec::SmallVec;

use std::borrow::Cow;

use ndarray::Ix;

pub type IdxVec = SmallVec<[usize; 3]>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpSetKind {
    Gathers(Vec<Gather>),
    Merge(IdxVec, usize),
    Split(usize, IdxVec),
}

impl OpSetKind {
    pub fn gathers(&self) -> Option<&[Gather]> {
        use OpSetKind::*;
        match self {
            Gathers(vec) => Some(vec),
            Merge(_, _) | Split(_, _) => None,
        }
    }

    pub fn merge_target(&self) -> Option<usize> {
        use OpSetKind::*;
        match self {
            Merge(_, to) => Some(*to),
            Gathers(_) | Split(_, _) => None,
        }
    }
}

impl From<Vec<Gather>> for OpSetKind {
    fn from(gathers: Vec<Gather>) -> OpSetKind {
        OpSetKind::Gathers(gathers)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpSet {
    pub name: Cow<'static, str>,
    pub ops: OpSetKind,
    pub in_shape: ShapeVec,
    pub out_shape: ShapeVec,
    pub fused_fold: bool,
}

impl OpSet {
    pub fn new<T>(name: T, ops: OpSetKind, in_shape: ShapeVec, out_shape: ShapeVec,
                  fused_fold: bool) -> Self
    where T: Into<Cow<'static, str>> {
        Self { name: name.into(), ops, in_shape, out_shape, fused_fold }
    }

    pub fn to_name(&self) -> String {
        let in_strings: SmallVec<[String; 4]> = self.in_shape.iter().map(|v| v.to_string()).collect();
        let out_strings: SmallVec<[String; 4]> = self.out_shape.iter().map(|v| v.to_string()).collect();
        format!("{}-{}-{}", out_strings.join(","), self.name, in_strings.join(","))
    }
}

pub fn identity_gather(shape: &[Ix]) -> Gather {
    Gather::new(shape, |idxs| to_opt_ix(idxs, shape), "id")
}

pub fn merge_adapter_gather(out_shape: &[Ix], index: Ix) -> Gather {
    let last = out_shape.len() - 1;
    let in_shape = &out_shape[0..last];
    Gather::new(out_shape, |idxs|
                if idxs[last] != index { -1 }
                else { to_opt_ix(&idxs[0..last], in_shape) },
                format!("(merge){}", index))
}

pub fn identity(shape: &[Ix]) -> Result<OpSetKind> {
    Ok(OpSetKind::Gathers(vec![identity_gather(shape)]))
}

#[derive(Debug)]
pub struct SynthesisLevel {
    pub ops: OpSet,
    pub matrix: Option<TransitionMatrix>,
    pub lane: usize,
    pub expected_syms: usize,
    pub prune: bool,
}

impl SynthesisLevel {
    pub fn new(ops: OpSet, lane: usize, expected_syms: usize, prune: bool) -> Self {
        Self {ops , matrix: None, lane, expected_syms, prune }
    }
}
