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

use crate::state::{Gather,to_opt_ix,Operation};
use crate::transition_matrix::{TransitionMatrix};
use crate::misc::{ShapeVec,time_since};

use smallvec::SmallVec;

use std::borrow::Cow;
use std::cmp::{min, max};
use std::num::NonZeroUsize;
use std::time::Instant;

use ndarray::Ix;


    // pub fn to_name(&self) -> String {
    //     let in_strings: SmallVec<[String; 4]> = self.in_shape.iter().map(|v| v.to_string()).collect();
    //     let out_strings: SmallVec<[String; 4]> = self.out_shape.iter().map(|v| v.to_string()).collect();
    //     format!("{}-{}-{}", out_strings.join(","), self.name, in_strings.join(","))
    // }

pub fn identity_gather(shape: &[Ix]) -> Gather {
    Gather::new(shape, |idxs| (0, to_opt_ix(idxs, shape)), "id")
}

pub fn transpose_gather(in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    Gather::new(out_shape, |idxs| {
        let mut ret = 0;
        for (idx, scale) in idxs.iter().rev().zip(in_shape.iter()) {
            ret = ret * scale + idx;
        }
        (0, ret as isize)
    }, "tr")
}

pub fn stack_gather(in_shapes: &[ShapeVec], out_shape: &[Ix]) -> Gather {
    let last = out_shape.len() - 1;
    let in_shape = &out_shape[0..last];
    Gather::new(out_shape, |idxs|
                (idxs[last], to_opt_ix(&idxs[0..last], in_shapes[last].as_slice())),
                "stack")
}

pub fn identity(in_shapes: &[ShapeVec], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }

    let in_shape: &[usize] = in_shapes[0].as_slice();
    let out_prod: usize = out_shape.iter().product();
    let in_prod: usize = in_shape.iter().product();
    if out_prod != in_prod {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }

    Ok(vec![identity_gather(out_shape)])
}

pub fn transpose(in_shape: &[Ix], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }
    let in_shape = in_shapes[0].as_slice();
    if out_shape.iter().rev().zip(in_shape.iter()).any(|(a, b)| a != b) {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }

    Ok(vec![transpose_gather(in_shape, out_shape)])
}

pub fn stack(in_shapes: &[ShapeVec], out_shape: &ShapeVec) -> Result<Vec<Gather>> {
    for s in in_shapes {
        if s != out_shape[0..out_shape.len()-2] {
            return Err(ErrorKind::ShapeMismatch(out_shape.to_owned()))
        }
    }
    stack_gather(in_shapes, out_shape.as_slice())
}

pub fn rot_idx_r(in_shapes: &[ShapeVec],
                 out_shape: &[Ix], rot: Ix) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }
    let in_shape = in_shapes[0].as_slice();
    let split = in_shape.len() - rot;
    if in_shape[split..] != out_shape[..rot] || in_shape[..split] != out_shape[rot..] {
        let mut expected_out = in_shape.to_vec();
        expected_out.rotate_right(rot);
        return Err(ErrorKind::ShapeMismatch(out_shape.to_vec(),
                                            expected_out)
                   .into());
    }

    let gather =
        Gather::new(out_shape, |idxs| {
            let mut idxs = ShapeVec::from_slice(idxs);
            // The shape is rotated right, here we need the inverse
            // operation to get the original indices
            idxs.rotate_left(rot);
            (0, to_opt_ix(&idxs, in_shape))
        }, format!("rot_idx{{r={}}}", rot));
    Ok(vec![gather])
}

pub fn rot_idx_l(in_shapes: &[ShapeVec],
                 out_shape: &[Ix], rot: Ix) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }
    let in_shape = in_shapes[0].as_slice();
    let split = in_shape.len() - rot;
    if in_shape[..rot] != out_shape[split..] || in_shape[rot..] != out_shape[..split] {
        let mut expected_out = in_shape.to_vec();
        expected_out.rotate_left(rot);
        return Err(ErrorKind::ShapeMismatch(out_shape.to_vec(),
                                            expected_out)
                   .into());
    }

    let gather =
        Gather::new(out_shape, |idxs| {
            let mut idxs = ShapeVec::from_slice(idxs);
            idxs.rotate_right(rot);
            (0, to_opt_ix(&idxs, in_shape))
        }, format!("rot_idx{{l={}}}", rot));
    Ok(vec![gather])
}

#[derive(Debug)]
pub struct SearchStep {
    pub op: Operation,
    pub matrix: Option<TransitionMatrix>,
    // min and max, if computed and distinct
    pub copy_bounds: Option<(Vec<u32>, Vec<u32>)>,
    pub prune: bool,
}

impl SearchStep {
    pub fn new(op: Operation, prune: bool) -> Self {
        Self {op , matrix: None, copy_bounds: None, prune }
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
        let summary = super::summarize(&gather).unwrap();
        assert_eq!(&gather[0], &summary);
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
        use crate::state::Gather;
        let map = std::collections::BTreeMap::new();
        let cond_keep: Vec<Gather> =
            super::select::cond_keep(&[4, 3], &[0, 1, -1],
                                     &map).unwrap();
        let summary = super::summarize(&cond_keep);
        let identity = super::identity_gather(&[4, 3]);
        assert_eq!(summary, Some(identity));
    }
}
