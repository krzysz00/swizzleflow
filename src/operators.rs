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
use crate::misc::{ShapeVec,time_since};

use smallvec::SmallVec;

use std::borrow::Cow;
use std::cmp::{min, max};
use std::num::NonZeroUsize;
use std::time::Instant;

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
    pub fold_dim: Option<NonZeroUsize>,
}

impl OpSet {
    pub fn new<T>(name: T, ops: OpSetKind, in_shape: ShapeVec, out_shape: ShapeVec,
                  fold_dim: Option<NonZeroUsize>) -> Self
    where T: Into<Cow<'static, str>> {
        Self { name: name.into(), ops, in_shape, out_shape, fold_dim }
    }

    #[inline(always)]
    pub fn has_fold(&self) -> bool {
        self.fold_dim.is_some()
    }

    pub fn prunes_like_identity(&self) -> bool {
        use OpSetKind::*;
        if self.has_fold() { return false; }

        match &self.ops {
            Gathers(_, summary) => summary.as_ref()
                .map(|g| g.is_identity()).unwrap_or(false),
            Stack(from, to) => from.len() == 1 && from[0] == *to,
            Split(from, to) => to.len() == 1 && *from == to[0],
        }
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
    // min and max, if computed and distinct
    pub copy_bounds: Option<(Vec<u32>, Vec<u32>)>,
    pub lane: usize,
    pub expected_syms: usize,
    pub prune: bool,
}

impl SynthesisLevel {
    pub fn new(ops: OpSet, lane: usize, expected_syms: usize, prune: bool) -> Self {
        Self {ops , matrix: None, copy_bounds: None, lane, expected_syms, prune }
    }
}

pub fn add_copy_bounds(levels: &mut [SynthesisLevel], max_lanes: usize) -> Result<()> {
    let mut prev_bounds: Vec<Option<(Vec<u32>, Vec<u32>)>> = vec![None; max_lanes];
    let output_length = levels[levels.len() - 1].ops.out_shape.iter().product();
    let output_counts = vec![1u32; output_length];
    prev_bounds[levels[levels.len() - 1].lane] = Some((output_counts.clone(), output_counts));

    let mut first_prunes = vec![levels.len(); max_lanes];
    for (idx, l) in levels.iter().enumerate().rev() {
        if l.prune {
            first_prunes[l.lane] = idx;
        }
    }

    let start = Instant::now();
    for (_idx, level) in levels.iter_mut().enumerate().rev()
        .filter(|(i, l)| *i >= first_prunes[l.lane])
    {
        let lane = level.lane;
        if level.prune {
            // Store data for after
            level.copy_bounds = prev_bounds[lane].clone();
            println!("prune_bounds[{}][0] = {:?}", _idx, level.copy_bounds.as_ref()
                     .map(|(small, big)| (small[0], big[0])));
        }

        let old_bounds = prev_bounds[lane].take().unwrap();
        let in_length = level.ops.in_shape.iter().product();
        let fold_dim: Option<usize> = level.ops.fold_dim.map(|v| v.into());
        match level.ops.ops {
            OpSetKind::Gathers(ref swizzles, _) => {
                let mut mins = vec![u32::MAX; in_length];
                let mut maxs = vec![0; in_length];
                for gather in swizzles {
                    let (this_min, this_max) = gather.min_max_copies(
                        in_length, &old_bounds.0, &old_bounds.1, fold_dim);
                    mins.iter_mut().zip(this_min.into_iter())
                        .for_each(|(v, e)| *v = min(*v, e));
                    maxs.iter_mut().zip(this_max.into_iter())
                        .for_each(|(v, e)| *v = max(*v, e));

                }
                let new_bounds = (mins, maxs);
                if new_bounds == old_bounds {
                    level.copy_bounds = None;
                }
                prev_bounds[lane] = Some(new_bounds);
            },
            OpSetKind::Stack(ref from, _to) => {
                level.copy_bounds = None;
                if fold_dim.is_some() {
                    for lane in from.iter().copied() {
                            prev_bounds[lane] = Some(old_bounds.clone());
                    }
                }
                else {
                    let (old_mins, old_maxs) = old_bounds;
                    let chunk_length = old_mins.len() / from.len();
                    assert_eq!(old_maxs.len(), from.len() * chunk_length);
                    for (lane, (min_chunk, max_chunk)) in from.iter().copied()
                        .zip(old_mins.chunks(chunk_length).zip(old_maxs.chunks(chunk_length)))
                    {
                        prev_bounds[lane] = Some((min_chunk.to_owned(), max_chunk.to_owned()));
                    }
                }
            },
            OpSetKind::Split(from, ref to) => {
                level.copy_bounds = None;
                let mut new_mins = vec![0; in_length];
                let mut new_maxs = vec![0; in_length];
                for lane in to.iter().copied() {
                    {
                        let bounds = if lane == from {
                            &old_bounds
                        } else {
                            prev_bounds[lane].as_ref().unwrap()
                        };
                        new_mins.iter_mut().zip(bounds.0.iter()).for_each(|(v, e)| *v += e);
                        new_maxs.iter_mut().zip(bounds.1.iter()).for_each(|(v, e)| *v += e);
                    }
                    prev_bounds[lane] = None;
                }
                prev_bounds[from] = Some((new_mins, new_maxs));
            }
        }
    }
    let dur = time_since(start);
    println!("copy_counts:this time={};", dur);
    Ok(())
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
