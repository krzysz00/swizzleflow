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
pub mod reg_select;
pub mod load;

use crate::state::Gather;
use crate::misc::{ShapeVec,time_since,regularize_float_mat,intersect_matrices,
                  MergeSpot};
use crate::transition_matrix::{TransitionMatrix, build_or_load_matrix,
                               TransitionMatrixOps};
use crate::errors::*;

use smallvec::SmallVec;

use std::borrow::Cow;
use std::path::Path;
use std::time::Instant;

use ndarray::{Array2,Ix};

use itertools::Itertools;

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

    pub fn to_name(&self, merge: Option<MergeSpot>) -> String {
        let in_strings: SmallVec<[String; 4]> = self.in_shape.iter().map(|v| v.to_string()).collect();
        let out_strings: SmallVec<[String; 4]> = self.out_shape.iter().map(|v| v.to_string()).collect();
        if let Some(MergeSpot { lane, total_size }) = merge {
            format!("[{}/{}]{}-{}-{}", lane, total_size,
                    out_strings.join(","), self.name, in_strings.join(","))
        }
        else {
            format!("{}-{}-{}", out_strings.join(","), self.name, in_strings.join(","))
        }
    }
}

pub fn identity_gather(shape: &[Ix]) -> Gather {
    Gather::new(shape.len(), shape, |idxs, ops| ops.extend(idxs), "id")
}

pub fn identity(shape: &[Ix]) -> Result<OpSetKind> {
    Ok(OpSetKind::Gathers(vec![identity_gather(shape)]))
}

#[derive(Debug)]
pub struct SynthesisLevel {
    pub ops: OpSet,
    pub matrix: Option<TransitionMatrix>,
    pub lane: usize,
    // eventually redundant
    pub prune: bool,
}

impl SynthesisLevel {
    pub fn new(ops: OpSet, lane: usize, prune: bool) -> Self {
        Self {ops , matrix: None, lane, prune }
    }
}

pub fn add_matrices(directory: &Path, levels: &mut [SynthesisLevel],
                    max_lanes: usize) -> Result<()> {
    use crate::transition_matrix::density;

    let mut first_prunes = vec![levels.len(); max_lanes];
    for (idx, l) in levels.iter().enumerate().rev() {
        if l.prune {
            first_prunes[l.lane] = idx;
        }
    }

    let mut our_path = directory.to_path_buf();
    our_path.push("dummy");

    let outmost_shape = levels[levels.len() - 1].ops.out_shape.clone();

    let mut names = vec![String::new(); max_lanes];
    let mut merges: Vec<Option<MergeSpot>> = vec![None; max_lanes];
    let mut prev_mats: Vec<Option<Array2<f32>>> = vec![None; max_lanes];

    for (_idx, level) in levels.iter_mut().enumerate().rev()
        .filter(|(i, l)| *i >= first_prunes[l.lane])
    {
        match level.ops.ops {
            OpSetKind::Gathers(ref _swiz) => {
                let lane = level.lane;
                let name = level.ops.to_name(merges[lane]);

                if !names[lane].is_empty() {
                    names[lane].push('_');
                }
                names[lane].push_str(&name);

                our_path.set_file_name(names[lane].clone());

                if our_path.exists() {
                    let start = Instant::now();
                    let mat = TransitionMatrix::load_matrix(our_path.as_path())?;
                    let load_time = time_since(start);
                    println!("load:{} [{}]", our_path.display(), load_time);
                    prev_mats[lane] = Some(mat.to_f32_mat());
                    if level.prune {
                        level.matrix = Some(mat);
                    }
                }
                else {
                    let mut basis_path = directory.to_path_buf();
                    basis_path.push(name);
                    let basis_matrix = build_or_load_matrix(&level.ops, &basis_path,
                                                            merges[lane])?.to_f32_mat();
                    match &mut prev_mats[lane] {
                        Some(prev) => {
                            let mut output = Array2::<f32>::zeros((prev.shape()[0], basis_matrix.shape()[1]));
                            let start = Instant::now();
                            ndarray::linalg::general_mat_mul(1.0, prev, &basis_matrix, 0.0, &mut output);
                            let time = time_since(start);
                            regularize_float_mat(&mut output);
                            std::mem::swap(&mut output, prev);
                            std::mem::drop(output);

                            let our_form = TransitionMatrix::from_f32_mat(prev, &outmost_shape, &level.ops.in_shape);
                            println!("mul:{} density({}) [{}]", names[lane], density(&our_form), time);
                            our_form.store_matrix(&our_path)?;
                            if level.prune {
                                level.matrix = Some(our_form);
                            }
                        }
                        None => {
                            // Here, we've just generated the basis matrix
                            println!("Using newly-built {}", our_path.display());
                            if level.prune {
                                level.matrix = Some(TransitionMatrix::from_f32_mat(
                                    &basis_matrix, &level.ops.out_shape, &level.ops.in_shape));
                            }
                            prev_mats[lane] = Some(basis_matrix);
                        }
                    }
                }
                merges[lane] = None;
            },
            OpSetKind::Merge(ref from, to) => {
                if level.prune {
                    println!("WARNING: pruning on merges doesn't actually do anything");
                }
                // We assume that two non-folding merges aren't next to each other
                if !level.ops.fused_fold {
                    if let Some(ms) = merges[to] {
                        panic!("Chained fusion-less folds at {}, ({:?})", to, ms)
                    }
                    let size = from.len();
                    for (idx, lane) in from.iter().copied().enumerate() {
                        merges[lane] = Some((idx, size).into());
                    }
                }
                for lane in from.iter().copied() {
                    if lane != to {
                        names[lane] = names[to].clone();
                        prev_mats[lane] = prev_mats[to].clone();
                    }
                }
            },
            OpSetKind::Split(into, ref copies) => {
                let name = format!("&({})",
                                   copies.iter().map(|i| names[*i].clone()).join(","));
                our_path.set_file_name(name.clone());
                let mat = if our_path.exists() {
                    let start = Instant::now();
                    let load = TransitionMatrix::load_matrix(our_path.as_path())?;
                    let load_time = time_since(start);
                    println!("load:{} [{}]", our_path.display(), load_time);
                    let ret = load.to_f32_mat();
                    if level.prune {
                        level.matrix = Some(load);
                    }
                    ret
                }
                else {
                    let mut gather = prev_mats[copies[0]].as_ref().unwrap().clone();
                    let start = Instant::now();
                    for idx in copies.iter().copied() {
                        intersect_matrices(&mut gather, prev_mats[idx].as_ref().unwrap());
                    }
                    let intersect_time = time_since(start);
                    let our_form = TransitionMatrix::from_f32_mat(&gather, &outmost_shape,
                                                                  &level.ops.in_shape);
                    println!("intersect:{} density({}) [{}]", our_path.display(),
                             density(&our_form), intersect_time);
                    our_form.store_matrix(&our_path)?;
                    if level.prune {
                        level.matrix = Some(our_form);
                    }
                    gather
                };

                for idx in copies.iter().copied() {
                    prev_mats[idx] = None;
                    names[idx].clear();
                }
                prev_mats[into] = Some(mat);
                names[into] = name;
            }
        }
    }
    Ok(())
}

pub fn remove_matrices(levels: &mut [SynthesisLevel]) {
    for level in levels {
        std::mem::drop(level.matrix.take());
    }
}
