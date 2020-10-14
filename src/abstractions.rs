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

use crate::misc::ShapeVec;
use crate::state::{OpType, Operation, Block};
use crate::matrix::{DenseMatrix, RowSparseMatrix};
use crate::transition_matrix::{TransitionMatrix, build_mat,
                               block_translate_matrix,
                               op_matrix_name, density};
use crate::multiply::{transition_mul, transition_add};
use crate::misc::{time_since, COLLECT_STATS};

use std::borrow::{Borrow, Cow};
use std::cmp::{min, max};
use std::collections::HashMap;
use std::path::{Path,PathBuf};
use std::time::Instant;

use itertools::Itertools;

#[derive(Clone, Debug, Default)]
pub struct Abstractions {
    pub pairs_matrix: Option<TransitionMatrix>,
    // min and max, if computed and distinct
    pub copy_bounds: Option<(Vec<u32>, Vec<u32>)>,
}

fn stats<T: std::fmt::Display>(tag: &str, path_or_tag: T,
                               matrix: &TransitionMatrix, dur: f64) {
    if COLLECT_STATS {
        let (in_slots, out_slots) = matrix.slots();
        println!("{}:{} n_ones={}; n_elems={}; in_slots={}; out_slots={}; density={}; time={};",
                 tag, path_or_tag,
                 matrix.n_ones(), matrix.n_elements(),
                 in_slots, out_slots, density(matrix),
                 dur);
    }
    else {
        println!("{}:{} density={}; time={};",
                 tag, path_or_tag,
                 density(matrix), dur);
    }
}

fn build_or_load_basis_mat(ops: &Operation, out_shape: &[Option<ShapeVec>],
                           _name: &str,
                           path: impl AsRef<Path>) -> Result<TransitionMatrix> {
    let path = path.as_ref();
    if path.exists() {
        let start = Instant::now();
        let ret = TransitionMatrix::load_matrix(path)?;
        let dur = time_since(start);
        stats("load", path.display(), &ret, dur);
        Ok(ret)
    }
    else {
        let start = Instant::now();
        // This does the right thing for literals and subgraphs
        let matrix = build_mat::<RowSparseMatrix>(ops, out_shape);
        let dur = time_since(start);
        matrix.store_matrix(path)?;
        stats("build", path.display(), &matrix, dur);
        Ok(matrix)
    }
}

fn get_basis_mat<'a>(bases: &'a mut HashMap<String, TransitionMatrix>,
                     basis_path: &mut PathBuf, name: &str, out_shape: &[Option<ShapeVec>],
                     ops: &Operation) -> Result<&'a TransitionMatrix> {
    if !bases.contains_key(name) {
        let mut basis_path = basis_path.clone();
        basis_path.set_file_name(name);
        let ret = build_or_load_basis_mat(ops, out_shape, name, basis_path.as_path())?;
        bases.insert(name.to_owned(), ret);
    }
    Ok(bases.get(name).unwrap())
}

fn load_matrix(path: &Path) -> Result<TransitionMatrix> {
    let start = Instant::now();
    let mat = TransitionMatrix::load_matrix(path)?;
    let load_time = time_since(start);
    stats("load", path.display(), &mat, load_time);
    Ok(mat)
}

fn hash_file_name(name: &str, path: &mut PathBuf) {
    use ring::digest::{digest, SHA256};
    let hash = digest(&SHA256, name.as_bytes());
    // Should be enough there
    let encoded = hex::encode(&(hash.as_ref()[0..5]));
    println!("file_map:{} raw={}", encoded, name);
    path.set_file_name(encoded);
}

fn add_cap(basis_name: &str, total_name: &str, path: &mut PathBuf,
           matrix: &TransitionMatrix,
           input_lanes: &[usize], input_shape: &[Option<ShapeVec>],
           output_lanes: &[usize], output_shape: &[Option<ShapeVec>])
           -> Result<TransitionMatrix> {
    hash_file_name(&total_name, path);
    if path.exists() {
        load_matrix(path)
    } else {
        let start_build = Instant::now();
        let basis_mat = block_translate_matrix::<RowSparseMatrix>(
            input_lanes, input_shape, output_lanes, output_shape);
        let dur_build = time_since(start_build);
        stats("build", &basis_name, &basis_mat, dur_build);

        let start_mul = Instant::now();
        let total_mat = transition_mul(&basis_mat, &matrix);
        let dur_mul = time_since(start_mul);
        stats("mul", path.display(), &total_mat, dur_mul);
        total_mat.store_matrix(&path)?;
        Ok(total_mat)
    }
}

fn add_matrices_rec(path: &mut PathBuf,
                    bases: &mut HashMap<String, TransitionMatrix>,
                    ops: &mut [Operation],
                    target_shape: &[Option<ShapeVec>], need_all: bool,
                    init_mat: TransitionMatrix,
                    mut name: String) -> Result<Option<(TransitionMatrix, String)>> {
    if ops.is_empty() { return Ok(None); }

    let mut matrix = init_mat;
    let mut latest_shape = target_shape.to_vec();

    let first_prune = if need_all { 0 } else { ops.iter().take_while(|l| !l.prune).count() };
    if first_prune >= ops.len()  { return Ok(None); }
    let range = first_prune..ops.len();
    for (idx, op) in ops[range].iter_mut().enumerate().rev()
    {
        if op.prune {
            op.abstractions.pairs_matrix = Some(matrix.clone());
        }
        let (block_mat, block_name) = if let Some(block) = op.op.block_mut() {
            let need_all = op.in_lanes.len() > 0;

            let exit_mat_name = format!(".[{}<{}]exit_{}", block.out_lane,
                                        op.out_lane, op.op_name);
            let recur_mat_name = format!("{}.{}", exit_mat_name, name);
            (&[block.out_lane], &block.out_shape,
             &[op.out_lane], latest_shape.as_slice());
            let recur_mat = add_cap(&exit_mat_name, &recur_mat_name, path, &matrix,
                                    &[block.out_lane], block.out_shape.as_slice(),
                                    &[op.out_lane], latest_shape.borrow())?;

            if let Some((res_mat, res_name)) =
                add_matrices_rec(path, bases,
                                 &mut block.ops, &block.out_shape,
                                 need_all, recur_mat, recur_mat_name)?
            {
                let post_mat_name = format!(
                    "post_{}{{{}}}", op.op_name,
                    op.in_lanes.iter().map(|i| i.to_string()).join(","));
                let capped_mat_name = format!("+{}.{}", post_mat_name, res_name);
                let range_vec: Vec<usize> = (0..op.in_lanes.len()).collect();
                let capped_mat = add_cap(&post_mat_name, &capped_mat_name, path, &res_mat,
                                         &op.in_lanes, &op.lane_in_shapes,
                                         &range_vec, &block.ops[0].lane_in_shapes)?;
                (Some(capped_mat), Cow::from(capped_mat_name))
            }
            else {
                (None, "".into())
            }
        }
        else {
            (None, "".into())
        };
        // No need to multiply further, we won't need the matrix for before
        // the first step where we apply pruning
        if idx == 0 && !need_all { () }
        else if op.prunes_like_identity() {
            matrix = matrix.reinterpret_current_shapes(op.lane_in_shapes.as_ref());
        }
        else {
            let basis_name = op_matrix_name(&op, latest_shape.borrow());
            name.insert(0, '.');
            name.insert_str(0, &block_name);
            name.insert_str(0, &basis_name);
            hash_file_name(&name, path);
            if path.exists() {
                matrix = load_matrix(path.as_path())?;
            }
            else {
                let basis = get_basis_mat(bases, path, &basis_name,
                                          latest_shape.borrow(), op)?;

                let start = Instant::now();
                matrix = transition_mul(&basis, &matrix);
                let time = time_since(start);
                stats("mul", path.display(), &matrix, time);

                if let Some(block_mat) = block_mat {
                    let start = Instant::now();
                    transition_add(&mut matrix, &block_mat);
                    let time = time_since(start);
                    stats("add", path.display(), &matrix, time);
                }
                matrix.store_matrix(&path)?;
            }
        }
        latest_shape.clear();
        latest_shape.extend_from_slice(op.lane_in_shapes.borrow());
    }
    Ok(if need_all { Some((matrix, name)) } else { None })
}

pub fn add_matrices(directory: &Path, block: &mut Block) -> Result<()> {
    let mut path = directory.to_path_buf();
    path.push("dummy");
    let mut bases = HashMap::<String, TransitionMatrix>::new();

    // Identity matrix for the output row
    let initial_mat = block_translate_matrix::<DenseMatrix>(
        &[block.out_lane], &block.out_shape,
        &[block.out_lane], &block.out_shape);
    add_matrices_rec(&mut path, &mut bases, &mut block.ops, &block.out_shape,
                     false, initial_mat, String::new())
        .map(|_| ())
}

fn remove_matrices_rec(ops: &mut [Operation]) {
    for op in ops {
        std::mem::drop(op.abstractions.pairs_matrix.take());
        if let Some(b) = op.op.block_mut() {
            remove_matrices_rec(&mut b.ops);
        }
    }
}

pub fn remove_matrices(block: &mut Block) {
    remove_matrices_rec(&mut block.ops)
}

fn init_copy_bounds(out_lane: usize, out_shape: &[Option<ShapeVec>],
                    copy_from: Option<(&[u32], &[u32])>)
                    -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut mins: Vec<Vec<u32>> = out_shape.iter().enumerate()
        .map(|(i, ma)| ma.as_ref().map_or_else(
            || vec![],
            |shape| { vec![if i == out_lane { 1u32 } else { 0u32};
                           shape.iter().copied().product()] }))
        .collect();
    let mut maxs: Vec<Vec<u32>> = out_shape.iter().enumerate()
        .map(|(i, ma)| ma.as_ref().map_or_else(
            || vec![],
            |shape| { vec![if i == out_lane { 1u32 } else { 0u32};
                           shape.iter().copied().product()] }))
        .collect();
    if let Some((old_mins, old_maxs)) = copy_from {
        mins[out_lane] = old_mins.to_vec();
        maxs[out_lane] = old_maxs.to_vec();
    }
    (mins, maxs)
}

pub fn add_copy_bounds_rec(ops: &mut [Operation],
                           need_all: bool,
                           init_next_mins: Vec<Vec<u32>>,
                           init_next_maxs: Vec<Vec<u32>>)
                           -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    // The next_s hold the counts for the input of the step after the current one,
    // aka, the counts for the output of the current step
    if ops.is_empty() { return Ok((vec![], vec![])); }

    let start = Instant::now();
    let first_prune = if need_all { 0 }
        else { ops.iter().take_while(|l| !l.prune).count() };
    if first_prune >= ops.len() - 1 { return Ok((vec![], vec![])); }

    let mut next_mins = init_next_mins;
    let mut next_maxs = init_next_maxs;

    for op in ops[first_prune..].iter_mut().rev() {
        let mut mins: Vec<Vec<u32>> = op.lane_in_lens.iter().copied().
            map(|len| vec![u32::MAX; len]).collect();
        let mut maxs: Vec<Vec<u32>> = op.lane_in_lens.iter().copied().
            map(|len| vec![0u32; len]).collect();
        match op.op {
            OpType::Apply { ref fns, summary: ref _summary } => {
                for gather in fns {
                    let (this_min, this_max) = gather.min_max_copies(
                        op, &next_mins, &next_maxs);
                    mins.iter_mut().zip(this_min.into_iter())
                        .for_each(|(va, ea)| va.iter_mut().zip(ea.into_iter())
                                  .for_each(|(v, e)| *v = min(*v, e)));
                    maxs.iter_mut().zip(this_max.into_iter())
                        .for_each(|(va, ea)| va.iter_mut().zip(ea.into_iter())
                                  .for_each(|(v, e)| *v = max(*v, e)));
                }
            },
            OpType::Literal(_) => {
                for i in 0..op.lane_in_lens.len() {
                    if i != op.out_lane && !op.drop_lanes.contains(&i) {
                        mins[i] = next_mins[i].clone();
                        maxs[i] = next_maxs[i].clone();
                    }
                    else {
                        // drop lanes and the output are cleared
                        for e in mins[i].iter_mut() {
                            *e = 0;
                        }
                    }
                }
            },
            OpType::Subprog(ref mut new_block) => {
                let rec_need_all = op.in_lanes.len() > 0;
                // Clear minimums to 0 to mimic gather behavion
                for arr in mins.iter_mut() {
                    for e in arr.iter_mut() {
                        *e = 0;
                    }
                }

                let (init_mins, init_maxs) =
                    init_copy_bounds(new_block.out_lane, new_block.out_shape.as_slice(),
                    Some((&next_mins[op.out_lane], &next_maxs[op.out_lane])));
                let (their_mins, their_maxs) =
                    add_copy_bounds_rec(&mut new_block.ops, rec_need_all,
                                        init_mins, init_maxs)?;
                if rec_need_all {
                    for (their_lane, our_lane) in op.in_lanes.iter().copied().enumerate() {
                        mins[our_lane] = their_mins[their_lane].clone();
                        maxs[our_lane] = their_maxs[their_lane].clone();
                    }
                }
                for l in op.preserved_lanes.iter().copied() {
                    for i in 0..op.lane_in_lens[l] {
                        mins[l][i] += next_mins[l][i];
                        maxs[l][i] += next_maxs[l][i];
                    }
                }
            }
        }
        if op.prune && (mins != next_mins || maxs != next_maxs) {
            op.abstractions.copy_bounds = Some((next_mins.concat(), next_maxs.concat()));
        }
        next_mins = mins;
        next_maxs = maxs;
    }
    let dur = time_since(start);
    println!("copy_counts:this time={};", dur);
    Ok((next_mins, next_maxs))
}

pub fn add_copy_bounds(block: &mut Block) -> Result<()> {
    let (init_mins, init_maxs) =
        init_copy_bounds(block.out_lane, &block.out_shape, None);
    add_copy_bounds_rec(&mut block.ops, false, init_mins, init_maxs)
        .map(|(_, _)| ())
}
