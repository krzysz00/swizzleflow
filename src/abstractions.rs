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
use crate::state::{OpType, Operation};
use crate::matrix::RowSparseMatrix;
use crate::transition_matrix::{TransitionMatrix, build_mat,
                               op_matrix_name, density};
use crate::multiply::transition_mul;
use crate::misc::{time_since, COLLECT_STATS};

use std::borrow::Borrow;
use std::cmp::{min, max};
use std::collections::HashMap;
use std::path::{Path,PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct Abstractions {
    pub pairs_matrix: Option<TransitionMatrix>,
    // min and max, if computed and distinct
    pub copy_bounds: Option<(Vec<u32>, Vec<u32>)>,
}

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

fn build_or_load_basis_mat(ops: &Operation, out_shape: &[Option<ShapeVec>],
                           _name: &str,
                           path: impl AsRef<Path>) -> Result<TransitionMatrix> {
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
        // This does the right thing for literals and subgraphs
        let matrix = build_mat::<RowSparseMatrix>(ops, out_shape);
        let dur = time_since(start);
        matrix.store_matrix(path)?;
        stats("build", path, &matrix, dur);
        Ok(matrix)
    }
}

fn get_basis_mat<'a>(bases: &'a mut HashMap<String, TransitionMatrix>,
                     directory: &Path, name: &str, out_shape: &[Option<ShapeVec>],
                     ops: &Operation) -> Result<&'a TransitionMatrix> {
    if !bases.contains_key(name) {
        let mut basis_path = directory.to_path_buf();
        basis_path.push(name);
        let ret = build_or_load_basis_mat(ops, out_shape, name, basis_path.as_path())?;
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

fn hash_file_name(name: &str, path: &mut PathBuf) {
    use ring::digest::{digest, SHA256};
    let hash = digest(&SHA256, name.as_bytes());
    // Should be enough there
    let encoded = hex::encode(&(hash.as_ref()[0..5]));
    println!("file_map:{} raw={}", encoded, name);
    path.set_file_name(encoded);
}

pub fn add_matrices(directory: &Path, ops: &mut [Operation],
                    target_shape: &[Option<ShapeVec>]) -> Result<()> {
    if ops.is_empty() { return Ok(()); }

    let mut path = directory.to_path_buf();
    path.push("dummy");
    let mut bases = HashMap::<String, TransitionMatrix>::new();

    let (mut name, mut matrix, mut latest_shape) = {
        let last_op = &ops[ops.len() - 1];
        let name = op_matrix_name(last_op, target_shape);
        let matrix = get_basis_mat(&mut bases, directory, &name,
                                   target_shape, last_op)?;
        path.set_file_name(name.clone());
        (name, matrix.clone(), last_op.lane_in_shapes.clone())
    };

    let first_prune = ops.iter().take_while(|l| !l.prune).count();
    // Skip last level, we handled it
    if first_prune >= ops.len() - 1 { return Ok(()); }
    let range = first_prune..(ops.len()-1);
    for (idx, op) in ops[range].iter_mut().enumerate().rev()
    {
        if op.prune {
            op.abstractions.pairs_matrix = Some(matrix.clone());
        }
        // No need to multiply further, we won't need the matrix for before
        // the first step where we apply pruning
        if idx == 0 { () }
        else if op.prunes_like_identity() {
            matrix = matrix.reinterpret_current_shapes(op.lane_in_shapes.as_ref());
        }
        else {
            let basis_name = op_matrix_name(&op, latest_shape.borrow());
            name.insert(0, '.');
            name.insert_str(0, &basis_name);
            hash_file_name(&name, &mut path);
            if path.exists() {
                matrix = load_matrix(path.as_path())?;
            }
            else {
                let basis = get_basis_mat(&mut bases, directory, &basis_name,
                                          latest_shape.borrow(), op)?;
                let start = Instant::now();
                matrix = transition_mul(&basis, &matrix);
                let time = time_since(start);

                stats("mul", path.as_ref(), &matrix, time);
                matrix.store_matrix(&path)?;
            }
        }
        latest_shape.clear();
        latest_shape.extend_from_slice(op.lane_in_shapes.borrow());
    }
    Ok(())
}

pub fn remove_matrices(ops: &mut [Operation]) {
    for op in ops {
        std::mem::drop(op.abstractions.pairs_matrix.take());
    }
}

pub fn add_copy_bounds(ops: &mut [Operation], out_shape: &[Option<ShapeVec>]) -> Result<()> {
    // The next_s hold the counts for the input of the step after the current one,
    // aka, the counts for the output of the current step
    if ops.is_empty() { return Ok(()); }

    let start = Instant::now();
    let mut next_mins: Vec<Vec<u32>> = out_shape.iter()
        .map(|ma| ma.as_ref().map_or_else(
            || vec![],
            |shape| vec![1u32; shape.iter().copied().product()]))
        .collect();
    let mut next_maxs: Vec<Vec<u32>> = out_shape.iter()
        .map(|ma| ma.as_ref().map_or_else(
            || vec![],
            |shape| vec![1u32; shape.iter().copied().product()]))
        .collect();
    let first_prune = ops.iter().take_while(|l| !l.prune).count();
    if first_prune >= ops.len() - 1 { return Ok(()); }

    for op in ops[first_prune..].iter_mut().rev() {
        let mut mins: Vec<Vec<u32>> = op.lane_in_lens.iter().copied().
            map(|len| vec![u32::MAX; len]).collect();
        let mut maxs: Vec<Vec<u32>> = op.lane_in_lens.iter().copied().
            map(|len| vec![0u32; len]).collect();
        match &op.op {
            OpType::Apply { fns, summary: _summary } => {
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
            OpType::Take(_) => {
                for i in 0..op.lane_in_lens.len() {
                    if i != op.out_lane && !op.drop_lanes.contains(&i) {
                        mins[i] = next_mins[i].clone();
                        maxs[i] = next_maxs[i].clone();
                    }
                }
            },
        }
        if op.prune && (mins != next_mins || maxs != next_maxs) {
            op.abstractions.copy_bounds = Some((next_mins.concat(), next_maxs.concat()));
        }
        next_mins = mins;
        next_maxs = maxs;
    }
    let dur = time_since(start);
    println!("copy_counts:this time={};", dur);
    Ok(())
}