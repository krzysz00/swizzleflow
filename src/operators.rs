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
pub mod hvx;

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
    // gathers, summary - choose one of the gathers in gathers
    // Summary is optionally a gather that is a superset of the behavior of all the `gathers`
    // if one exists (that is, if the gathers in the vector only differ by whether they don't perform
    // certain reads)
    Gathers(Vec<Gather>, Option<Gather>),
    // [from, ...], to - merge the `from` lanes into the `to` lanes,
    // stacking the tensors on top of each ohter
    Stack(IdxVec, usize),
    // from, [to ...] - copy lane `from` to the `to` lanes
    Split(usize, IdxVec),
}

impl OpSetKind {
    pub fn new_gathers(gathers: Vec<Gather>) -> Self {
        let summary = if gathers.is_empty() {
            None
        }
        else {
            let mut all_merged = gathers[0].clone();
            if (&gathers[1..]).iter().fold(true, |acc, g| acc && all_merged.merge_with(g)) {
                Some(all_merged)
            }
            else {
                None
            }
        };
        Self::Gathers(gathers, summary)
    }

    pub fn gathers(&self) -> Option<&[Gather]> {
        use OpSetKind::*;
        match self {
            Gathers(vec, _) => Some(vec),
            Stack(_, _) | Split(_, _) => None,
        }
    }

    pub fn summary(&self) -> Option<&Gather> {
        use OpSetKind::*;
        match self {
            Gathers(_, summary) => summary.as_ref(),
            Stack(_, _) | Split(_, _) => None,
        }
    }

    pub fn stack_target(&self) -> Option<usize> {
        use OpSetKind::*;
        match self {
            Stack(_, to) => Some(*to),
            Gathers(_, _) | Split(_, _) => None,
        }
    }
}

impl From<Vec<Gather>> for OpSetKind {
    fn from(gathers: Vec<Gather>) -> OpSetKind {
        OpSetKind::new_gathers(gathers)
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

pub fn transpose_gather(in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    Gather::new(out_shape, |idxs| {
        let mut ret = 0;
        for (idx, scale) in idxs.iter().rev().zip(in_shape.iter()) {
            ret = ret * scale + idx;
        }
        ret as isize
    }, "tr")
}

pub fn stack_adapter_gather(out_shape: &[Ix], index: Ix) -> Gather {
    let last = out_shape.len() - 1;
    let in_shape = &out_shape[0..last];
    Gather::new(out_shape, |idxs|
                if idxs[last] != index { -1 }
                else { to_opt_ix(&idxs[0..last], in_shape) },
                format!("(stack){}", index))
}

pub fn identity(shape: &[Ix]) -> Result<OpSetKind> {
    Ok(OpSetKind::new_gathers(vec![identity_gather(shape)]))
}

pub fn transpose(in_shape: &[Ix], out_shape: &[Ix]) -> Result<OpSetKind> {
    Ok(OpSetKind::new_gathers(vec![transpose_gather(in_shape, out_shape)]))
}



pub fn rot_idx_r(in_shape: &[Ix],
                 out_shape: &[Ix], rot: Ix) -> Result<OpSetKind> {
    let gather =
        Gather::new(out_shape, |idxs| {
            let mut idxs = ShapeVec::from_slice(idxs);
            // The shape is rotated right, here we need the inverse
            // operation to get the original indices
            idxs.rotate_left(rot);
            to_opt_ix(&idxs, in_shape)
        }, format!("rot_idx_r({})", rot));
    Ok(OpSetKind::new_gathers(vec![gather]))
}

pub fn rot_idx_l(in_shape: &[Ix],
                 out_shape: &[Ix], rot: Ix) -> Result<OpSetKind> {
    let gather =
        Gather::new(out_shape, |idxs| {
            let mut idxs = ShapeVec::from_slice(idxs);
            idxs.rotate_right(rot);
            to_opt_ix(&idxs, in_shape)
        }, format!("rot_idx_l({})", rot));
    Ok(OpSetKind::new_gathers(vec![gather]))
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

#[cfg(test)]
mod test {
    #[test]
    fn summary_one() {
        let gather = super::identity(&[3, 3]).unwrap();
        assert_eq!(&gather.gathers().unwrap()[0],
                   gather.summary().unwrap());
    }

    #[test]
    fn summary_fails() {
        let gathers = super::OpSetKind::new_gathers(
            vec![super::identity_gather(&[2, 3]),
                 super::transpose_gather(&[3, 2], &[2, 3])]);
        assert_eq!(gathers.summary(), None)
    }

    #[test]
    fn summary_passes() {
        let map = std::collections::BTreeMap::new();
        let cond_keep = super::select::cond_keep(&[4, 3], &[0, 1, -1],
                                                 &map).unwrap();
        let identity = super::identity_gather(&[4, 3]);
        assert_eq!(cond_keep.summary(), Some(&identity));
    }
}
