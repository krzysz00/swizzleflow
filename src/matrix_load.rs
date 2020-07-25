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

use crate::operators::{OpSet, OpSetKind, SynthesisLevel, stack_adapter_gather};
use crate::matrix::RowSparseMatrix;
use crate::transition_matrix::{TransitionMatrix, build_mat,
                               density};
use crate::multiply::transition_mul;
use crate::misc::{time_since,COLLECT_STATS};

use std::collections::HashMap;
use std::path::{Path,PathBuf};
use std::time::Instant;

use itertools::Itertools;
use itertools::iproduct;

fn stats(tag: &str, path: &Path, matrix: &TransitionMatrix, dur: f64) {
    if COLLECT_STATS {
        let (in_slots, out_slots) = matrix.slots();
        println!("{}:{} n_ones={}; n_elems={}; in_slots={}; out_slots={}; density={}; time={};",
                 tag, path.display(),
                 matrix.n_ones(), matrix.n_elements(),
                 in_slots, out_slots, density(matrix),
                 dur);
    }
    else {
        println!("{}:{} density={}; time={};",
                 tag, path.display(),
                 density(matrix), dur);
    }
}

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
            a.set_idxs(c1, c2, t1, t2,
                       true);
        }
    }
}

fn build_or_load_basis_mat(ops: &OpSet, path: impl AsRef<Path>) -> Result<TransitionMatrix> {
    let path = path.as_ref();
    if path.exists() {
        let start = Instant::now();
        let ret = TransitionMatrix::load_matrix(path)?;
        let dur = time_since(start);
        stats("load", path, &ret, dur);
        Ok(ret)
    }
    else {
        let start = Instant::now();
        let matrix = build_mat::<RowSparseMatrix>(ops);
        let dur = time_since(start);
        matrix.store_matrix(path)?;
        stats("build", path, &matrix, dur);
        Ok(matrix)
    }
}

fn get_basis_mat<'a>(bases: &'a mut HashMap<String, TransitionMatrix>,
                     directory: &Path, name: &str,
                     ops: &OpSet) -> Result<&'a TransitionMatrix> {
    if !bases.contains_key(name) {
        let mut basis_path = directory.to_path_buf();
        basis_path.push(name);
        let ret = build_or_load_basis_mat(ops, &basis_path)?;
        bases.insert(name.to_owned(), ret);
    }
    Ok(bases.get(name).unwrap())
}

fn load_matrix(path: &Path) -> Result<TransitionMatrix> {
    let start = Instant::now();
    let mat = TransitionMatrix::load_matrix(path)?;
    let load_time = time_since(start);
    stats("load", path, &mat, load_time);
    Ok(mat)
}

fn add_matrix(ops: &OpSet, lane: usize,
              path: &mut PathBuf,
              names: &mut [String], prev_mats: &mut [Option<TransitionMatrix>],
              bases: &mut HashMap<String, TransitionMatrix>) -> Result<()> {
    // TODO: actually propagate forward the true target shape
    if ops.prunes_like_identity() && prev_mats[lane].is_some() {
        prev_mats[lane] = Some(prev_mats[lane].as_ref().unwrap()
                               .reinterpret_current_shape(ops.in_shape.clone()))
    }
    else {
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
                    let start = Instant::now();
                    let mut output = transition_mul(basis_matrix, prev);
                    let time = time_since(start);

                    stats("mul", path.as_ref(), &output, time);
                    output.store_matrix(&path)?;
                    std::mem::swap(&mut output, prev);
                },
                None => {
                    // Here, we've just generated the basis matrix
                    println!("Using newly-built {}", path.display());
                    prev_mats[lane] = Some(basis_matrix.clone());
                }
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

    let mut names = vec![String::new(); max_lanes];
    let mut bases = HashMap::<String, TransitionMatrix>::new();
    let mut prev_mats: Vec<Option<TransitionMatrix>> = vec![None; max_lanes];

    for (idx, level) in levels.iter_mut().enumerate().rev()
        .filter(|(i, l)| *i >= first_prunes[l.lane])
    {
        let lane = level.lane;
        if level.prune {
            level.matrix = prev_mats[lane].clone();
        }

        if idx == first_prunes[lane] {
            prev_mats[lane] = None;
            continue;
        }

        match level.ops.ops {
            OpSetKind::Gathers(ref _swiz, _) => {
                add_matrix(&level.ops, lane, &mut our_path, &mut names, &mut prev_mats,
                           &mut bases)?;
            },
            OpSetKind::Stack(ref from, to) => {
                for lane in from.iter().copied() {
                    if lane != to {
                        names[lane] = names[to].clone();
                        prev_mats[lane] = prev_mats[to].clone();
                    }
                }
                if !level.ops.has_fold() {
                    let out_shape = &level.ops.out_shape;
                    let in_shape = &level.ops.in_shape;
                    for (column, lane) in from.iter().copied().enumerate() {
                        let gather = vec![stack_adapter_gather(out_shape, column)];
                        let name = gather[0].name.clone();
                        let opset = OpSetKind::new_gathers(gather);
                        let ops = OpSet::new(name, opset,
                                             in_shape.clone(), out_shape.clone(),
                                             None);
                        add_matrix(&ops, lane, &mut our_path, &mut names,
                                   &mut prev_mats, &mut bases)?;
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
                    stats("load", our_path.as_ref(), &ret, load_time);
                    ret
                }
                else {
                    let mut gather = prev_mats[copies[0]].as_ref().unwrap().clone();
                    let start = Instant::now();
                    for idx in copies[1..].iter().copied() {
                        union_matrices(&mut gather, prev_mats[idx].as_ref().unwrap());
                    }
                    let union_time = time_since(start);
                    stats("union", our_path.as_ref(),
                          &gather, union_time);
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
