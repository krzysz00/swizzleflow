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

use crate::state::{OpType, Operation};
use crate::matrix::{MatrixOps, Matrix};
use crate::misc::{ShapeVec, OffsetsVec, shapes_to_offsets,
                  open_file, create_file,
                  write_length_tagged_idxs,
                  read_length_tagged_idxs};
use crate::errors::*;

use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};
use smallvec::SmallVec;

use itertools::{Itertools,iproduct};

#[derive(Clone, Debug)]
pub struct GenTransitionMatrix<T: MatrixOps> {
    target_shapes: Vec<ShapeVec>,
    current_shapes: Vec<ShapeVec>,
    current_lane_offs: OffsetsVec,
    target_lane_offs: OffsetsVec,
    current_len: Ix,
    target_len: Ix,
    pub mat: T,
}

pub type TransitionMatrix = GenTransitionMatrix<Matrix>;

fn to_index_part(idx: &[Ix], shape: &ShapeVec) -> Ix {
    let mut ret = 0;
    for (v, scale) in idx.iter().zip(shape.iter()) {
        // ret starts at 0 so the initial case is fine
        ret = (ret * scale) + v
    }
    ret
}

#[inline(always)]
fn to_index(lane1: Ix, idx1: Ix, lane2: Ix, idx2: Ix,
            offsets: &[Ix], len: Ix) -> Ix {
    (idx2 + offsets[lane2]) + (len * (idx1 + offsets[lane1]))
}

impl<T: MatrixOps> GenTransitionMatrix<T> {
    pub fn new(current_shapes: &[Option<ShapeVec>], target_shapes: &[Option<ShapeVec>], mat: T) -> Self {
        let current_lane_offs = shapes_to_offsets(current_shapes);
        let target_lane_offs = shapes_to_offsets(target_shapes);
        let current_shapes = current_shapes.iter().map(
            |x| x.as_ref().map_or_else(|| ShapeVec::new(), |s| s.clone())).collect();
        let target_shapes = target_shapes.iter().map(
            |x| x.as_ref().map_or_else(|| ShapeVec::new(), |s| s.clone())).collect();
        let current_len = current_lane_offs[current_lane_offs.len()-1];
        let target_len = target_lane_offs[target_lane_offs.len()-1];
        let n_rows = current_len.pow(2);
        let n_cols = target_len.pow(2);
        assert_eq!((n_rows, n_cols), mat.dims());
        Self { mat, current_shapes, target_shapes,
               current_len, target_len,
               current_lane_offs, target_lane_offs, }
    }

    pub fn new_no_option(current_shapes: &[ShapeVec], target_shapes: &[ShapeVec], mat: T) -> Self {
        let option_current_shapes = current_shapes.iter()
            .map(|s| if s.is_empty() { None } else { Some(s.to_owned()) })
            .collect::<Vec<Option<ShapeVec>>>();
        let option_target_shapes = target_shapes.iter()
            .map(|s| if s.is_empty() { None } else { Some(s.to_owned()) })
            .collect::<Vec<Option<ShapeVec>>>();
        let current_lane_offs = shapes_to_offsets(&option_current_shapes);
        let target_lane_offs = shapes_to_offsets(&option_target_shapes);
        let current_len = current_lane_offs[current_lane_offs.len()-1];
        let target_len = target_lane_offs[target_lane_offs.len()-1];
        let n_rows = current_len.pow(2);
        let n_cols = target_len.pow(2);
        assert_eq!((n_rows, n_cols), mat.dims());
        Self { mat,
               current_shapes: current_shapes.to_owned(),
               target_shapes: target_shapes.to_owned(),
               current_len, target_len,
               current_lane_offs, target_lane_offs, }
    }

    pub fn new_with<F>(current_shapes: &[Option<ShapeVec>], target_shapes: &[Option<ShapeVec>],
                       mat_fn: F) -> Self
    where F: FnOnce(usize, usize) -> T {
        let current_lane_offs = shapes_to_offsets(current_shapes);
        let target_lane_offs = shapes_to_offsets(target_shapes);
        let current_shapes = current_shapes.iter().map(
            |x| x.as_ref().map_or_else(|| ShapeVec::new(), |s| s.clone())).collect();
        let target_shapes = target_shapes.iter().map(
            |x| x.as_ref().map_or_else(|| ShapeVec::new(), |s| s.clone())).collect();
        let current_len = current_lane_offs[current_lane_offs.len()-1];
        let target_len = target_lane_offs[target_lane_offs.len()-1];
        let n_rows = current_len.pow(2);
        let n_cols = target_len.pow(2);
        let mat = mat_fn(n_rows, n_cols);
        Self { mat, current_shapes, target_shapes,
               current_len, target_len,
               current_lane_offs, target_lane_offs, }
    }

    pub fn empty(current_shapes: &[Option<ShapeVec>],
                 target_shapes: &[Option<ShapeVec>]) -> Self {
        Self::new_with(current_shapes, target_shapes, |m, n| T::empty(m, n))
    }

    pub fn with_row_size_hint(current_shapes: &[Option<ShapeVec>],
                              target_shapes: &[Option<ShapeVec>], hint: usize) -> Self {
        Self::new_with(current_shapes, target_shapes,
                       move |m, n| T::with_row_size_hint(m, n, hint))
    }

    pub fn general_with_row_size_hint(current_shapes: &[Option<ShapeVec>],
                                      target_shapes: &[Option<ShapeVec>], hint: usize)
                                      -> TransitionMatrix {
        TransitionMatrix::new_with(current_shapes, target_shapes,
                                   move |m, n| T::with_row_size_hint(m, n, hint).into())

    }

    pub fn get(&self, (current_lane1, current1): (Ix, &[Ix]),
               (current_lane2, current2): (Ix, &[Ix]),
               (target_lane1, target1): (Ix, &[Ix]),
               (target_lane2, target2): (Ix, &[Ix])) -> bool {
        let i1 = to_index_part(current1, &self.current_shapes[current_lane1]);
        let i2 = to_index_part(current2, &self.current_shapes[current_lane2]);
        let i = to_index(current_lane1, i1, current_lane2, i2,
                         &self.current_lane_offs, self.current_len);
        let j1 = to_index_part(target1, &self.target_shapes[target_lane1]);
        let j2 = to_index_part(target2, &self.target_shapes[target_lane2]);
        let j = to_index(target_lane1, j1, target_lane2, j2,
                         &self.target_lane_offs, self.target_len);
        self.mat.get(i, j)
    }

    pub fn get_cur_pos(&self, (current_lane1, current1): (Ix, Ix),
                       (current_lane2, current2): (Ix, Ix),
                       (target_lane1, target1): (Ix, &[Ix]),
                       (target_lane2, target2): (Ix, &[Ix])) -> bool {
        let i = to_index(current_lane1, current1, current_lane2, current2,
                         &self.current_lane_offs, self.current_len);
        let j1 = to_index_part(target1, &self.target_shapes[target_lane1]);
        let j2 = to_index_part(target2, &self.target_shapes[target_lane2]);
        let j = to_index(target_lane1, j1, target_lane2, j2,
                         &self.target_lane_offs, self.target_len);
        self.mat.get(i, j)
    }

    pub fn get_idxs(&self, (current_lane1, current1): (Ix, Ix),
                    (current_lane2, current2): (Ix, Ix),
                    (target_lane1, target1): (Ix, Ix),
                    (target_lane2, target2): (Ix, Ix)) -> bool {
        let i = to_index(current_lane1, current1, current_lane2, current2,
                         &self.current_lane_offs, self.current_len);
        let j = to_index(target_lane1, target1, target_lane2, target2,
                         &self.target_lane_offs, self.target_len);
        self.mat.get(i, j)
    }

    #[inline]
    pub fn get_raw_idxs(&self, current1: Ix, current2: Ix,
                        target1: Ix, target2: Ix) -> bool {
        let i = current2 + self.current_len * current1;
        let j = target2 + self.target_len * target1;
        self.mat.get(i, j)
    }

    pub fn set(&mut self, (current_lane1, current1): (Ix, &[Ix]),
               (current_lane2, current2): (Ix, &[Ix]),
               (target_lane1, target1): (Ix, &[Ix]),
               (target_lane2, target2): (Ix, &[Ix]), value: bool)  {
        let i1 = to_index_part(current1, &self.current_shapes[current_lane1]);
        let i2 = to_index_part(current2, &self.current_shapes[current_lane2]);
        let i = to_index(current_lane1, i1, current_lane2, i2,
                         &self.current_lane_offs, self.current_len);
        let j1 = to_index_part(target1, &self.target_shapes[target_lane1]);
        let j2 = to_index_part(target2, &self.target_shapes[target_lane2]);
        let j = to_index(target_lane1, j1, target_lane2, j2,
                         &self.target_lane_offs, self.target_len);
        self.mat.set(i, j, value)
    }

    pub fn set_cur_pos(&mut self, (current_lane1, current1): (Ix, Ix),
                       (current_lane2, current2): (Ix, Ix),
                       (target_lane1, target1): (Ix, &[Ix]),
                       (target_lane2, target2): (Ix, &[Ix]), value: bool) {
        let i = to_index(current_lane1, current1, current_lane2, current2,
                         &self.current_lane_offs, self.current_len);
        let j1 = to_index_part(target1, &self.target_shapes[target_lane1]);
        let j2 = to_index_part(target2, &self.target_shapes[target_lane2]);
        let j = to_index(target_lane1, j1, target_lane2, j2,
                         &self.target_lane_offs, self.target_len);
        self.mat.set(i, j, value)
    }

    #[inline]
    pub fn set_idxs(&mut self, (current_lane1, current1): (Ix, Ix),
                    (current_lane2, current2): (Ix, Ix),
                    (target_lane1, target1): (Ix, Ix),
                    (target_lane2, target2): (Ix, Ix), value: bool) {
        let i = to_index(current_lane1, current1, current_lane2, current2,
                         &self.current_lane_offs, self.current_len);
        let j = to_index(target_lane1, target1, target_lane2, target2,
                         &self.target_lane_offs, self.target_len);
        self.mat.set(i, j, value)
    }

    #[inline]
    pub fn set_raw_idxs(&mut self, current1: Ix, current2: Ix,
                        target1: Ix, target2: Ix, value: bool) {
        let i = current2 + self.current_len * current1;
        let j = target2 + self.target_len * target1;
        self.mat.set(i, j, value)
    }


    pub fn slots(&self) -> (usize, usize) {
        (self.current_len, self.target_len)
    }
    pub fn n_ones(&self) -> usize {
        self.mat.n_ones()
    }
    pub fn n_elements(&self) -> usize {
        self.mat.n_elements()
    }

    pub fn get_current_shapes(&self) -> &[ShapeVec] {
        self.current_shapes.as_slice()
    }

    pub fn get_target_shapes(&self) -> &[ShapeVec] {
        self.target_shapes.as_slice()
    }

    pub fn reinterpret_current_shapes(&self, new_shapes: &[Option<ShapeVec>]) -> Self {
        let mut ret = self.clone();
        ret.current_lane_offs = shapes_to_offsets(new_shapes);
        ret.current_shapes = new_shapes.iter().map(
            |x| x.as_ref().map_or_else(|| ShapeVec::new(), |s| s.clone())).collect();
        ret.current_len = ret.current_lane_offs[ret.current_lane_offs.len()-1];

        if ret.current_len != self.current_len {
            panic!("Incompatible input lengths in reinterpret: {} -> {}",
                   self.current_len, ret.current_len);
        }
        ret
    }

    pub fn write<U: Write>(&self, io: &mut U) -> Result<()> {
        io.write_u64::<LittleEndian>(self.current_shapes.len() as u64)?;
        for shape in self.current_shapes.iter() {
            write_length_tagged_idxs(io, shape)?;
        }
        io.write_u64::<LittleEndian>(self.target_shapes.len() as u64)?;
        for shape in self.target_shapes.iter() {
            write_length_tagged_idxs(io, shape)?;
        }
        self.mat.write(io)
    }

    pub fn read<U: Read>(io: &mut U) -> Result<Self> {
        let n_current_lanes = io.read_u64::<LittleEndian>()?;
        let current_shapes: Result<Vec<ShapeVec>> =
            (0..n_current_lanes).map(
                |_| read_length_tagged_idxs(io).map_err(Error::from))
            .collect();
        let current_shapes = current_shapes?;
        let n_target_lanes = io.read_u64::<LittleEndian>()?;
        let target_shapes: Result<Vec<ShapeVec>> =
            (0..n_target_lanes).map(
                |_| read_length_tagged_idxs(io).map_err(Error::from))
            .collect();
        let target_shapes = target_shapes?;
        let mat = T::read(io)?;
        Ok(Self::new_no_option(&current_shapes, &target_shapes, mat))
    }
}

pub fn build_mat<T: MatrixOps>(op: &Operation, out_shape: &[Option<ShapeVec>])
                               -> TransitionMatrix {
    let fold_dim = op.fold_len.map_or(1, |x| x.get());
    let has_fold = op.fold_len.is_some();
    let out_lane = op.out_lane;

    // Preserved lanes need to be passed through
    // Note, something like v = range(a, b) preserves all the lanes except the one
    // that's about to contain v
    let preserved_lanes = op.preserved_lanes.iter().copied()
        .map(|i| (i, op.lane_in_lens[i]))
        .collect::<SmallVec<[(usize, usize); 4]>>();
    let fns =
        match &op.op {
            OpType::Literal(_) => &[],
            // If we can summarize a basis, use that summary to save time
            OpType::Apply { fns: _fns, summary: Some(s) } =>
                ref_slice::ref_slice(s),
            OpType::Apply { fns , summary: None } =>
                fns,
            OpType::Subprog(_) => {
                &[]
            }
        };
    let n_args = op.in_lanes.len();
    let arg_lanes = &op.in_lanes;

    let n_ops = fns.len();

    let mut ret = GenTransitionMatrix::<T>::general_with_row_size_hint(
        &op.lane_in_shapes, &out_shape, n_ops);
    for fun in fns {
        for (output1, (a1, e1)) in fun.data.into_iter().copied().enumerate()
            .filter(|&(_, (a, e))| a < n_args && e >= 0
                    && (e as usize) < op.lane_in_lens[arg_lanes[a]])
        {
            let output1 = (out_lane, if has_fold {
                output1 / fold_dim
            } else {
                output1
            });
            let input1 = (arg_lanes[a1], e1 as usize);
            for (output2, (a2, e2)) in fun.data.into_iter().copied().enumerate()
                .filter(|&(_, (a, e))| a < n_args && e >= 0
                        && (e as usize) < op.lane_in_lens[arg_lanes[a]])
            {
                // Remove fold dimension if needed
                let output2 = (out_lane, if has_fold {
                    output2 / fold_dim
                } else {
                    output2
                });
                let input2 = (arg_lanes[a2], e2 as usize);
                ret.set_idxs(input1, input2, output1, output2, true);
            }

            // Add implicit identity function on preserved lanes
            for &(l, len) in preserved_lanes.iter() {
                for i in 0..len {
                    ret.set_idxs(input1, (l, i), output1, (l, i), true);
                    ret.set_idxs((l, i), input1, (l, i), output1, true);
                }
            }
        }
    }
    // Scribble identity in the appropriate places
    for &(l1, len1) in preserved_lanes.iter() {
        for &(l2, len2) in preserved_lanes.iter() {
            for (i1, i2) in iproduct![0..len1, 0..len2] {
                ret.set_idxs((l1, i1), (l2, i2), (l1, i1), (l2, i2), true)
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

fn shapes_to_string(shapes: &[Option<ShapeVec>]) -> String {
    shapes.iter()
        .map(|ms|
             ms.as_ref().map_or_else(|| String::new(), |s| s.iter().map(|i| i.to_string()).join("x")))
        .join(",")
}

// The matrix that moves input_lanes to output_lanes, clearing all others
pub fn block_translate_matrix<T: MatrixOps>(
    input_lanes: &[usize], input_shape: &[Option<ShapeVec>],
    output_lanes: &[usize], output_shape: &[Option<ShapeVec>]) -> TransitionMatrix
{
    assert_eq!(input_lanes.len(), output_lanes.len());

    let mut ret = GenTransitionMatrix::<T>::general_with_row_size_hint(
        input_shape, output_shape, input_lanes.len());
    let in_lane_lens = input_lanes.iter().copied()
        .map(|il| input_shape[il].as_ref().
             map_or(0usize, |a| a.iter().copied().product()))
        .collect::<SmallVec<[usize; 4]>>();
    let out_lane_lens = output_lanes.iter().copied()
        .map(|il| output_shape[il].as_ref().
             map_or(0usize, |a| a.iter().copied().product()))
        .collect::<SmallVec<[usize; 4]>>();
    assert_eq!(in_lane_lens, out_lane_lens);
    for (lane1, (il1, ol1)) in input_lanes.iter().copied()
        .zip(output_lanes.iter().copied()).enumerate()
    {
        let len1 = in_lane_lens[lane1];
        for (lane2, (il2, ol2)) in input_lanes.iter().copied()
            .zip(output_lanes.iter().copied()).enumerate()
        {
            let len2 = in_lane_lens[lane2];
            for i in 0..len1 {
                for j in 0..len2 {
                    ret.set_idxs((il1, i), (il2, j), (ol1, i), (ol2, j), true);
                }
            }
        }
    }
    ret
}

pub fn op_matrix_name(op: &Operation, out_shape: &[Option<ShapeVec>]) -> String {
    format!(
        "{}-[{}>{}{}{}]{}-{}",
        shapes_to_string(&op.lane_in_shapes),
        op.in_lanes.iter().map(|i| i.to_string()).join(","),
        op.out_lane,
        if op.drop_lanes.is_empty() { "" } else { " -"},
        op.drop_lanes.iter().map(|i| i.to_string()).join("-"),
        if op.op.is_take() { "" } else { &op.op_name },
        shapes_to_string(out_shape)
    )
}

pub fn density<T: MatrixOps>(matrix: &GenTransitionMatrix<T>) -> f64 {
    (matrix.n_ones() as f64) / (matrix.n_elements() as f64)
}
