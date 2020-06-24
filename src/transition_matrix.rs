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
use ndarray::{Ix};

use std::io::{Write,Read,BufReader,BufWriter};
use std::path::Path;

use crate::operators::OpSet;
use crate::matrix::{MatrixOps, Matrix};
use crate::misc::{ShapeVec,open_file,create_file,
                  write_length_tagged_idxs,
                  read_length_tagged_idxs};
use crate::errors::*;

#[derive(Clone, Debug)]
pub struct GenTransitionMatrix<T: MatrixOps> {
    target_shape: ShapeVec,
    current_shape: ShapeVec,
    current_len: usize,
    target_len: usize,
    pub mat: T,
}

pub type TransitionMatrix = GenTransitionMatrix<Matrix>;

fn to_index(c1: &[Ix], c2: &[Ix], shape: &ShapeVec) -> Ix {
    let mut ret = 0;
    for (v, scale) in c1.iter().zip(shape.iter()).chain(
        c2.iter().zip(shape.iter())) {
        // ret starts at 0 so the initial case is fine
        ret = (ret * scale) + v
    }
    ret
}

#[inline(always)]
fn to_raw_index(c1: Ix, c2: Ix, shape: Ix) -> Ix {
    c2 + (shape * c1)
}

impl<T: MatrixOps> GenTransitionMatrix<T> {
    pub fn new(current_shape: &[Ix], target_shape: &[Ix], mat: T) -> Self {
        let current_len: usize = current_shape.iter().copied().product();
        let target_len: usize = target_shape.iter().copied().product();
        let n_rows = current_len.pow(2);
        let n_cols = target_len.pow(2);
        assert_eq!((n_rows, n_cols), mat.dims());
        Self { mat, current_len, target_len,
               current_shape: ShapeVec::from_slice(current_shape),
               target_shape: ShapeVec::from_slice(target_shape) }
    }

    pub fn new_with<F>(current_shape: &[Ix], target_shape: &[Ix],
                       mat_fn: F) -> Self
        where F: FnOnce(usize, usize) -> T {
        let current_len: usize = current_shape.iter().copied().product();
        let target_len: usize = target_shape.iter().copied().product();
        let n_rows = current_len.pow(2);
        let n_cols = target_len.pow(2);
        let mat = mat_fn(n_rows, n_cols);
        Self { mat, current_len, target_len,
               current_shape: ShapeVec::from_slice(current_shape),
               target_shape: ShapeVec::from_slice(target_shape) }

    }

    pub fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self {
        Self::new_with(current_shape, target_shape, |m, n| T::empty(m, n))
    }

    pub fn with_row_size_hint(current_shape: &[Ix], target_shape: &[Ix], hint: usize) -> Self {
        Self::new_with(current_shape, target_shape,
                       move |m, n| T::with_row_size_hint(m, n, hint))
    }

    pub fn general_with_row_size_hint(current_shape: &[Ix],
                                      target_shape: &[Ix], hint: usize)
                                      -> TransitionMatrix {
        TransitionMatrix::new_with(current_shape, target_shape,
                                   move |m, n| T::with_row_size_hint(m, n, hint).into())

    }

    pub fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        let i = to_index(current1, current2, &self.current_shape);
        let j = to_index(target1, target2, &self.target_shape);
        self.mat.get(i, j)
    }

    pub fn get_cur_pos(&self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix]) -> bool {
        let i = to_raw_index(current1, current2, self.current_len);
        let j = to_index(target1, target2, &self.target_shape);
        self.mat.get(i, j)
    }

    pub fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool {
        let i = to_raw_index(current1, current2, self.current_len);
        let j = to_raw_index(target1, target2, self.target_len);
        self.mat.get(i, j)
    }

    pub fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        let i = to_index(current1, current2, &self.current_shape);
        let j = to_index(target1, target2, &self.target_shape);
        self.mat.set(i, j, value)
    }

    pub fn set_cur_pos(&mut self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix], value: bool) {
        let i = to_raw_index(current1, current2, self.current_len);
        let j = to_index(target1, target2, &self.target_shape);
        self.mat.set(i, j, value)
    }

    pub fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool) {
        let i = to_raw_index(current1, current2, self.current_len);
        let j = to_raw_index(target1, target2, self.target_len);
        self.mat.set(i, j, value)
    }

    pub fn slots(&self) -> (usize, usize) {
        (self.current_len, self.target_len)
    }

    pub fn get_current_shape(&self) -> &[Ix] {
        self.current_shape.as_slice()
    }
    pub fn get_target_shape(&self) -> &[Ix] {
        self.target_shape.as_slice()
    }

    pub fn n_ones(&self) -> usize {
        self.mat.n_ones()
    }
    pub fn n_elements(&self) -> usize {
        self.mat.n_elements()
    }

    pub fn reinterpret_current_shape(&self, new_shape: ShapeVec) -> Self {
        let mut ret = self.clone();
        ret.current_len = new_shape.iter().copied().product();
        ret.current_shape = new_shape;

        if ret.current_len != self.current_len {
            panic!("Incompatible input lengths in reinterpret: {} -> {}",
                   self.current_len, ret.current_len);
        }
        ret
    }

    pub fn write<U: Write>(&self, io: &mut U) -> Result<()> {
        write_length_tagged_idxs(io, &self.current_shape)?;
        write_length_tagged_idxs(io, &self.target_shape)?;
        self.mat.write(io)
    }

    pub fn read<U: Read>(io: &mut U) -> Result<Self> {
        let current_shape = read_length_tagged_idxs(io)?;
        let target_shape = read_length_tagged_idxs(io)?;
        let current_len: usize = current_shape.iter().copied().product();
        let target_len: usize = target_shape.iter().copied().product();
        let mat = T::read(io)?;
        Ok(Self { current_shape, target_shape,
                  current_len, target_len, mat })

    }

}

pub fn build_mat<T: MatrixOps>(ops: &OpSet) -> TransitionMatrix {
    let out_shape = ops.out_shape.to_owned();

    let in_slots: usize = ops.in_shape.iter().product();
    let in_bound = in_slots as isize;

    // Get actual last dimension
    let fold_dim = ops.fold_dim.map(|x| x.get()).unwrap_or(1);
    let has_fold = ops.has_fold();

    let gathers = ops.ops.gathers().unwrap();
    let n_ops = gathers.len();

    let mut ret = GenTransitionMatrix::<T>::general_with_row_size_hint(
        &ops.in_shape, &out_shape,n_ops);
    for op in gathers {
        for (output1, input1) in op.data.into_iter().copied()
            .enumerate().filter(|&(_, i)| i >= 0 && i < in_bound) {
                for (output2, input2) in op.data.into_iter().copied()
                    .enumerate().filter(|&(_, i)| i >= 0 && i < in_bound) {
                        // Remove fold dimension if needed
                        let output1 = if has_fold {
                            output1 / fold_dim
                        } else {
                            output1
                        };
                        let output2 = if has_fold {
                            output2 / fold_dim
                        } else {
                            output2
                        };
                        ret.set_idxs(input1 as usize, input2 as usize,
                                     output1, output2,
                                     true);
                    }
            }
    }
    ret
}

impl TransitionMatrix {
    pub fn load_matrix(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = open_file(path)?;
        let mut reader = BufReader::new(file);
        Self::read(&mut reader).chain_err(|| ErrorKind::MatrixLoad(path.to_owned()))
    }

    pub fn store_matrix(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = create_file(path)?;
        let mut writer = BufWriter::new(file);
        self.write(&mut writer)
    }

}


pub fn density<T: MatrixOps>(matrix: &GenTransitionMatrix<T>) -> f64 {
    (matrix.n_ones() as f64) / (matrix.n_elements() as f64)
}
