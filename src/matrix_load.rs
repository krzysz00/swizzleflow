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

use crate::errors::*;

use crate::operators::{OpSet, OpSetKind, SynthesisLevel, merge_adapter_gather};
use crate::transition_matrix::{TransitionMatrix, build_or_load_matrix,
                               TransitionMatrixOps, density};
use crate::misc::{time_since};

use std::collections::HashMap;
use std::path::{Path,PathBuf};
use std::time::Instant;

use ndarray::{Array2,Ix};

use itertools::Itertools;
use itertools::iproduct;

fn union_matrices(a: &mut TransitionMatrix, b: &TransitionMatrix) {
    let slots = a.slots();
    if slots != b.slots() {
        panic!("Length mismatches in matrix union {:?} {:?}", slots, b.slots());
    }
    let current = slots.0;
    let target = slots.1;
    for (c1, c2, t1, t2) in iproduct![(0..current), (0..current),
                                      (0..target), (0..target)] {
        if b.get_idxs(c1, c2, t1, t2) {
            a.set_idxs(c1, c2, t1, t2, true);
        }
    }
}

fn get_basis_mat<'a>(bases: &'a mut HashMap<String, Array2<f32>>,
                 directory: &Path, name: &str,
                 ops: &OpSet) -> Result<&'a Array2<f32>> {
    if !bases.contains_key(name) {
        let mut basis_path = directory.to_path_buf();
        basis_path.push(name);
        let ret = build_or_load_matrix(ops, &basis_path)?.to_f32_mat();
        bases.insert(name.to_owned(), ret);
    }
    Ok(bases.get(name).unwrap())
}

fn load_matrix(path: &Path) -> Result<TransitionMatrix> {
    let start = Instant::now();
    let mat = TransitionMatrix::load_matrix(path)?;
    let load_time = time_since(start);
    println!("load:{} [{}]", path.display(), load_time);
    Ok(mat)
}

fn add_matrix(ops: &OpSet, lane: usize,
              path: &mut PathBuf,
              names: &mut [String], prev_mats: &mut [Option<TransitionMatrix>],
              bases: &mut HashMap<String, Array2<f32>>,
              outmost_shape: &[Ix]) -> Result<()> {
    let name = ops.to_name();

    if !names[lane].is_empty() {
        // Drop redundant indexing
        names[lane].truncate(names[lane].rfind('-').expect("dash separating input length"));
        names[lane].push('_');
    }
    names[lane].push_str(&name);

    path.set_file_name(names[lane].clone());

    if path.exists() {
        let mat = load_matrix(path.as_path())?;
        prev_mats[lane] = Some(mat);
    }
    else {
        let basis_matrix = get_basis_mat(bases,
                                         path.parent().unwrap(), &name,
                                         ops)?;
        match prev_mats[lane] {
            Some(ref mut prev) => {
                let float = prev.to_f32_mat();
                let mut output = Array2::<f32>::zeros((float.shape()[0], basis_matrix.shape()[1]));
                let start = Instant::now();
                ndarray::linalg::general_mat_mul(1.0, &float, &basis_matrix, 0.0, &mut output);
                let time = time_since(start);

                let mut our_form = TransitionMatrix::from_f32_mat(&output, &outmost_shape, &ops.in_shape);
                println!("mul:{} density({}) [{}]", names[lane], density(&our_form), time);
                our_form.store_matrix(&path)?;
                std::mem::swap(&mut our_form, prev);
            },
            None => {
                // Here, we've just generated the basis matrix
                println!("Using newly-built {}", path.display());
                prev_mats[lane] =
                    Some(TransitionMatrix::from_f32_mat(
                        &basis_matrix, &ops.out_shape, &ops.in_shape));
            }
        }
    }
    Ok(())
}

pub fn add_matrices(directory: &Path, levels: &mut [SynthesisLevel],
                    max_lanes: usize) -> Result<()> {
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
    let mut bases = HashMap::<String, Array2<f32>>::new();
    let mut prev_mats: Vec<Option<TransitionMatrix>> = vec![None; max_lanes];

    for (idx, level) in levels.iter_mut().enumerate().rev()
        .filter(|(i, l)| *i >= first_prunes[l.lane])
    {
        if level.prune {
            level.matrix = prev_mats[level.lane].clone();
        }

        let lane = level.lane;
        if idx == first_prunes[lane] {
            prev_mats[lane] = None;
            continue;
        }

        match level.ops.ops {
            OpSetKind::Gathers(ref _swiz) => {
                add_matrix(&level.ops, lane, &mut our_path, &mut names, &mut prev_mats,
                           &mut bases, &outmost_shape)?;
            },
            OpSetKind::Merge(ref from, to) => {
                if level.ops.fused_fold {
                    for lane in from.iter().copied() {
                        if lane != to {
                            names[lane] = names[to].clone();
                            prev_mats[lane] = prev_mats[to].clone();
                        }
                    }
                }
                else {
                    let out_shape = &level.ops.out_shape;
                    let in_shape = &level.ops.in_shape;
                    for lane in from.iter().copied() {
                        let gather = vec![merge_adapter_gather(out_shape, lane)];
                        let name = gather[0].name.clone();
                        let opset = OpSetKind::Gathers(gather);
                        let ops = OpSet::new(name, opset,
                                             in_shape.clone(), out_shape.clone(),
                                             false);
                        add_matrix(&ops, lane, &mut our_path, &mut names,
                                   &mut prev_mats, &mut bases, &outmost_shape)?;
                    }
                }
            },
            OpSetKind::Split(into, ref copies) => {
                if level.prune {
                    panic!("Pruning right after a split isn't well-defined, aborting");
                }
                let name = format!("&({})",
                                   copies.iter().map(|i| names[*i].clone()).join(","));
                our_path.set_file_name(name.clone());
                let mat = if our_path.exists() {
                    let start = Instant::now();
                    let ret = TransitionMatrix::load_matrix(our_path.as_path())?;
                    let load_time = time_since(start);
                    println!("load:{} [{}]", our_path.display(), load_time);
                    ret
                }
                else {
                    let mut gather = prev_mats[copies[0]].as_ref().unwrap().clone();
                    let start = Instant::now();
                    for idx in copies[1..].iter().copied() {
                        union_matrices(&mut gather, prev_mats[idx].as_ref().unwrap());
                    }
                    let union_time = time_since(start);
                    println!("union:{} density({}) [{}]", our_path.display(),
                             density(&gather), union_time);
                    gather.store_matrix(&our_path)?;
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
