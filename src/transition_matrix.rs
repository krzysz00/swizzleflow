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
use ndarray::{Ix,Array2,Dimension};
use std::io::{Write,Read};
use std::io;
use std::path::Path;
use std::time::Instant;

use crate::operators::OpSet;
use crate::misc::{EPSILON,ShapeVec,time_since,open_file,create_file,MergeSpot};
use crate::errors::*;

use bit_vec::BitVec;

use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};

use smallvec::SmallVec;

pub trait TransitionMatrixOps: Sized + std::fmt::Debug {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool;
    fn get_pos(&self, pos: Ix) -> bool;
    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool);
    fn set_pos(&mut self, pos: Ix, value: bool);
    fn write<T: Write>(&self, io: &mut T) -> Result<()>;
    fn read<T: Read>(io: &mut T) -> Result<Self>;
    fn empty(len: usize, out_shape: &[Ix], in_shape: &[Ix]) -> Self;
    // The shape is [target1, target2, current1, current2]
    fn to_f32_mat(&self) -> Array2<f32>;
    fn from_f32_mat(mat: &Array2<f32>, out_shape: &[Ix], in_shape: &[Ix]) -> Self;

    fn n_ones(&self) -> usize;
    fn n_elements(&self) -> usize;
}

fn to_index(i1: &[Ix], i2: &[Ix], ishape: &[Ix], j1: &[Ix], j2: &[Ix], jshape: &[Ix]) -> Ix {
    let mut ret = 0;
    for (is, ss) in [(i1, ishape), (i2, ishape), (j1, jshape), (j2, jshape)].into_iter() {
        for (v, scale) in is.iter().zip(ss.iter()) {
            // ret starts at 0 so the initial case is fine
            ret = (ret * scale) + v
        }
    }
    ret
}

fn write_length_tagged_idxs<T: Write>(io: &mut T, data: &[Ix]) -> io::Result<()> {
    io.write_u64::<LittleEndian>(data.len() as u64)?;
    for i in data {
        io.write_u64::<LittleEndian>(*i as u64)?;
    }
    Ok(())
}

fn read_length_tagged_idxs<T: Read>(io: &mut T) -> io::Result<ShapeVec> {
    let length = io.read_u64::<LittleEndian>()? as usize;
    let mut buffer = ShapeVec::with_capacity(length);
    for _ in 0..length {
        buffer.push(io.read_u64::<LittleEndian>()? as usize);
    }
    Ok(buffer)
}

#[derive(Debug)]
pub struct DenseTransitionMatrix {
    data: BitVec,
    target_shape: ShapeVec,
    current_shape: ShapeVec,
}

impl DenseTransitionMatrix {
    fn new(data: BitVec, target_shape: ShapeVec, current_shape: ShapeVec) -> Self {
        Self {data, target_shape, current_shape}
    }
}

impl TransitionMatrixOps for DenseTransitionMatrix {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        self.data[to_index(target1, target2, &self.target_shape,
                           current1, current2, &self.current_shape)]
    }

    fn get_pos(&self, pos: Ix) -> bool {
        self.data[pos]
    }

    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        let index = to_index(target1, target2, &self.target_shape,
                             current1, current2, &self.current_shape);
        self.data.set(index, value);
    }

    fn set_pos(&mut self, pos: Ix, value: bool) {
        self.data.set(pos, value);
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        write_length_tagged_idxs(io, &self.target_shape)?;
        write_length_tagged_idxs(io, &self.current_shape)?;
        io.write_u64::<LittleEndian>(self.data.len() as u64)?;
        io.write_all(&self.data.to_bytes()).map_err(|e| e.into())
    }

    fn read<T: Read>(io: &mut T) -> Result<Self> {
        let target_shape = read_length_tagged_idxs(io)?;
        let current_shape = read_length_tagged_idxs(io)?;
        let len = io.read_u64::<LittleEndian>()? as usize;
        let io_len = (len + 7) / 8;
        let mut bytes = Vec::with_capacity(io_len);
        unsafe {
            // We need an unitialized pile of storage
            bytes.set_len(io_len);
        }
        io.read_exact(&mut bytes)?;

        let mut bits = BitVec::from_bytes(&bytes);
        bits.truncate(len);

        Ok(Self::new(bits, target_shape, current_shape))
    }

    fn empty(len: usize, out_shape: &[Ix], in_shape: &[Ix]) -> Self {
        let bits = BitVec::from_elem(len, false);
        Self::new(bits, ShapeVec::from_slice(out_shape), ShapeVec::from_slice(in_shape))
    }

    fn to_f32_mat(&self) -> Array2<f32> {
        let floats = self.data.iter().map(|b| if b { 1.0 } else { 0.0 }).collect();
        let target_slots: Ix = self.target_shape.iter().product();
        let current_slots: Ix = self.current_shape.iter().product();
        Array2::from_shape_vec((target_slots.pow(2), current_slots.pow(2)), floats).unwrap()
    }

    fn from_f32_mat(mat: &Array2<f32>, out_shape: &[Ix], in_shape: &[Ix]) -> Self {
        let bits = mat.as_slice().unwrap().iter().map(|f| !(f.abs() < EPSILON)).collect();
        Self::new(bits, ShapeVec::from_slice(out_shape), ShapeVec::from_slice(in_shape))
    }

    fn n_ones(&self) -> usize {
        self.data.iter().filter(|x| *x).count()
    }

    fn n_elements(&self) -> usize {
        self.data.len()
    }
}

fn in_bounds(index: ndarray::ArrayView1<Ix>, bounds: &[Ix]) -> bool {
    index.into_iter().zip(bounds.iter()).all(move |(i, v)| i < v)
}

pub fn build_mat<T: TransitionMatrixOps>(ops: &OpSet, merge: Option<MergeSpot>) -> T {
    let mut out_shape = ops.out_shape.to_owned();
    // Folds are already handled
    if let Some(ms) = merge {
        out_shape.push(ms.total_size);
    }

    let out_slots: usize = out_shape.iter().product();
    let in_slots: usize = ops.in_shape.iter().product();
    let bounds = &ops.in_shape;
    let len = (out_slots.pow(2)) * (in_slots.pow(2));
    let fold = ops.fused_fold;
    let merge_lane = merge.map(|ms| ms.lane);

    // empty takes out, in, unlike set and their friends
    let mut ret = T::empty(len, &out_shape, &ops.in_shape);
    let gathers = ops.ops.gathers().unwrap();
    for op in gathers {
        let axis_num = op.data.ndim() - 1;
        let output_shape = &op.data.shape()[0..axis_num];
        for (input1, output1) in op.data.genrows().into_iter()
            .zip(ndarray::indices(output_shape)).filter(|&(i, _)| in_bounds(i, bounds)) {
                for (input2, output2) in op.data.genrows().into_iter()
                    .zip(ndarray::indices(output_shape)).filter(|&(i, _)| in_bounds(i, bounds)) {
                        let mut out1: SmallVec<[usize; 6]> =
                            SmallVec::from_slice(output1.slice());
                        let mut out2: SmallVec<[usize; 6]> =
                            SmallVec::from_slice(output2.slice());
                        if fold {
                            out1.pop();
                            out2.pop();
                        }
                        if let Some(l) = merge_lane {
                            out1.push(l);
                            out2.push(l);
                        }
                        ret.set(input1.as_slice().unwrap(), input2.as_slice().unwrap(),
                                &out1, &out2,
                                true);
                    }
            }
    }
    ret
}

#[derive(Debug)]
pub enum TransitionMatrix {
    Dense(DenseTransitionMatrix),
}

const DENSE_MATRIX_TAG: u8 = 1;
const FILE_MARKER: &[u8; 8] = b"SWIZFLOW";

impl TransitionMatrixOps for TransitionMatrix {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get(current1, current2, target1, target2)
        }
    }

    fn get_pos(&self, pos: Ix) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get_pos(pos)
        }
    }

    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set(current1, current2, target1, target2, value)
        }
    }

    fn set_pos(&mut self, pos: Ix, value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set_pos(pos, value)
        }
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        io.write_all(FILE_MARKER)?;
        match self {
            TransitionMatrix::Dense(d) => {
                io.write_u8(DENSE_MATRIX_TAG)?;
                Ok(d.write(io)?)
            }
        }
    }

    fn read<T: Read>(io: &mut T) -> Result<Self> {
        let mut header: [u8; 8] = [0; 8];
        io.read_exact(&mut header)?;
        if &header != FILE_MARKER {
            return Err(ErrorKind::NotAMatrix.into());
        }
        let tag = io.read_u8()?;
        match tag {
            DENSE_MATRIX_TAG => {
                let mat = DenseTransitionMatrix::read(io)?;
                Ok(TransitionMatrix::Dense(mat))
            }
            e => Err(ErrorKind::UnknownMatrixType(e).into())
        }
    }

    fn empty(len: usize, out_shape: &[Ix], in_shape: &[Ix]) -> Self {
        TransitionMatrix::Dense(DenseTransitionMatrix::empty(len, out_shape, in_shape))
    }

    fn to_f32_mat(&self) -> Array2<f32> {
        match self {
            TransitionMatrix::Dense(d) => d.to_f32_mat()
        }
    }

    fn from_f32_mat(mat: &Array2<f32>, out_shape: &[Ix], in_shape: &[Ix]) -> Self {
       TransitionMatrix::Dense(DenseTransitionMatrix::from_f32_mat(mat, out_shape, in_shape))
    }

    fn n_ones(&self) -> usize {
        match self {
            TransitionMatrix::Dense(d) => d.n_ones()
        }
    }

    fn n_elements(&self) -> usize {
        match self {
            TransitionMatrix::Dense(d) => d.n_elements()
        }
    }
}

impl TransitionMatrix {
    pub fn load_matrix(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = open_file(path)?;
        Self::read(&mut file).chain_err(|| ErrorKind::MatrixLoad(path.to_owned()))
    }

    pub fn store_matrix(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = create_file(path)?;
        self.write(&mut file)
    }
}

impl From<DenseTransitionMatrix> for TransitionMatrix {
    fn from(d: DenseTransitionMatrix) -> Self {
        TransitionMatrix::Dense(d)
    }
}

pub fn build_or_load_matrix(ops: &OpSet, path: impl AsRef<Path>,
                            merge: Option<MergeSpot>) -> Result<TransitionMatrix> {
    let path = path.as_ref();
    if path.exists() {
        let start = Instant::now();
        let ret = TransitionMatrix::load_matrix(path);
        let dur = time_since(start);
        println!("load:{} [{}]", path.display(), dur);
        ret
    }
    else {
        let start = Instant::now();
        let matrix = TransitionMatrix::Dense(build_mat(ops, merge));
        matrix.store_matrix(path)?;
        let dur = time_since(start);
        println!("build:{} density({}) [{}]", path.display(), density(&matrix), dur);
        Ok(matrix)
    }
}

pub fn density<T: TransitionMatrixOps>(matrix: &T) -> f64 {
    (matrix.n_ones() as f64) / (matrix.n_elements() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::swizzle::{simple_fans,OpAxis};
    use smallvec::smallvec;
    use tempfile::tempfile;

    #[test]
    fn correct_length_trove_rows() {
        let small_fans = simple_fans(&[3, 4], OpAxis::Rows).unwrap();
        let opset = OpSet::new("sFr", small_fans, smallvec![3, 4],
                               smallvec![3, 4], false);
        let matrix: DenseTransitionMatrix = build_mat(&opset, None);
        assert_eq!(matrix.n_ones(), 488);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn correct_length_big_matrix() {
        use crate::operators::swizzle::simple_rotations;
        let big_rots = simple_rotations(&[4, 32], OpAxis::Columns).unwrap();
        let opset = OpSet::new("sRr", big_rots, smallvec![4, 32],
                               smallvec![4, 32], false);
        let big_matrix: DenseTransitionMatrix = build_mat(&opset, None);
        assert_eq!(big_matrix.n_ones(), 246272);
    }

    #[test]
    fn test_write_round_trip() {
        use ndarray::Array2;
        use std::io::{Seek,SeekFrom};

        const M: usize = 3;
        const N: usize = 8;

        let size = (M * N).pow(2);
        let floats: Vec<f32> = (0..size.pow(2)).map(|x| if x % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let floats = Array2::from_shape_vec((size, size), floats).unwrap();
        let matrix = DenseTransitionMatrix::from_f32_mat(
            &floats, &[M, N], &[M, N]);

        let mut file = tempfile().unwrap();
        matrix.write(&mut file).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let matrix2 = DenseTransitionMatrix::read(&mut file).unwrap();
        assert_eq!(matrix.data, matrix2.data);
        assert_eq!(matrix.current_shape, matrix2.current_shape);
        assert_eq!(matrix.target_shape, matrix2.target_shape);
    }
}
