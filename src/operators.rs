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
use crate::misc::{ShapeVec,time_since,regularize_float_mat};
use crate::transition_matrix::{TransitionMatrix, build_or_load_matrix, TransitionMatrixOps};
use crate::errors::*;

use smallvec::SmallVec;

use std::borrow::Cow;

use std::path::Path;
use std::time::Instant;

use ndarray::Array2;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpSet {
    pub name: Cow<'static, str>,
    pub ops: Vec<Gather>,
    pub in_shape: ShapeVec,
    pub out_shape: ShapeVec,
    pub fused_fold: bool,
}

impl OpSet {
    pub fn new<T>(name: T, ops: Vec<Gather>, in_shape: ShapeVec, out_shape: ShapeVec,
                  fused_fold: bool) -> Self
    where T: Into<Cow<'static, str>> {
        Self { name: name.into(), ops, in_shape, out_shape, fused_fold }
    }

    pub fn add_fused_fold(&mut self) {
        if !self.fused_fold {
            self.fused_fold = true;
            self.out_shape.pop();
        }
    }

    pub fn to_name(&self) -> String {
        let in_strings: SmallVec<[String; 4]> = self.in_shape.iter().map(|v| v.to_string()).collect();
        let out_strings: SmallVec<[String; 4]> = self.out_shape.iter().map(|v| v.to_string()).collect();
        format!("{}-{}-{}", out_strings.join(","), self.name, in_strings.join(","))
    }
}

#[derive(Debug)]
pub struct SynthesisLevel {
    pub ops: OpSet,
    pub matrix: Option<TransitionMatrix>,
    // eventually redundant
    pub prune: bool,
}

impl SynthesisLevel {
    pub fn new(ops: OpSet, prune: bool) -> Self {
        Self {ops , matrix: None, prune }
    }
}

pub fn add_matrices(directory: &Path, levels: &mut [SynthesisLevel]) -> Result<()> {
    use crate::transition_matrix::density;
    // Error checking
    for subslice in levels.windows(2) {
        if subslice[0].ops.out_shape != subslice[1].ops.in_shape {
            return Err(ErrorKind::ShapeMismatch(subslice[0].ops.out_shape.to_vec(),
                                                subslice[1].ops.in_shape.to_vec()).into())
        }
    }

    let n_levels = levels.len();
    let first_prune = levels.iter().enumerate()
        .filter_map(|(i, v)| if v.prune { Some(i) } else { None })
        .next().unwrap_or(n_levels);
    let mut our_path = directory.to_path_buf();
    our_path.push("dummy");

    let outmost_shape = levels[n_levels - 1].ops.out_shape.clone();

    let mut previous_matrix: Option<Array2<f32>> = None;

    let mut names = String::new();
    for (_idx, level) in levels.iter_mut().enumerate().rev().take_while(|(i, _)| *i >= first_prune) {
        let name = level.ops.to_name();

        if !names.is_empty() {
            names.push('_');
        }
        names.push_str(&name);

        our_path.set_file_name(names.clone());

        if our_path.exists() {
            let start = Instant::now();
            let mat = TransitionMatrix::load_matrix(our_path.as_path())?;
            let load_time = time_since(start);
            println!("load:{} [{}]", our_path.display(), load_time);
            previous_matrix = Some(mat.to_f32_mat());
            if level.prune {
                level.matrix = Some(mat);
            }
        }
        else {
            let mut basis_path = directory.to_path_buf();
            basis_path.push(name);
            let basis_matrix = build_or_load_matrix(&level.ops, &basis_path)?.to_f32_mat();
            match &mut previous_matrix {
                Some(prev) => {
                    let mut output = Array2::<f32>::zeros((prev.shape()[0], basis_matrix.shape()[1]));
                    let start = Instant::now();
                    ndarray::linalg::general_mat_mul(1.0, prev, &basis_matrix, 0.0, &mut output);
                    let time = time_since(start);
                    regularize_float_mat(&mut output);
                    std::mem::swap(&mut output, prev);
                    std::mem::drop(output);

                    let our_form = TransitionMatrix::from_f32_mat(prev, &outmost_shape, &level.ops.in_shape);
                    println!("mul:{} density({}) [{}]", names, density(&our_form), time);
                    our_form.store_matrix(&our_path)?;
                    if level.prune {
                        level.matrix = Some(our_form);
                    }
                }
                None => {
                    // Here, we've just generated the basis matrix
                    println!("Using newly-build {}", our_path.display());
                    if level.prune {
                        level.matrix = Some(TransitionMatrix::from_f32_mat(&basis_matrix, &level.ops.out_shape, &level.ops.in_shape));
                    }
                    previous_matrix = Some(basis_matrix);
                }
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
