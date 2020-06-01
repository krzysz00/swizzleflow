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
use std::io::{Write,Read,BufReader,BufWriter};
use std::io;
use std::path::Path;

use crate::operators::OpSet;
use crate::misc::{EPSILON,ShapeVec,open_file,create_file};
use crate::errors::*;

use bit_vec::BitVec;

use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};

use smallvec::SmallVec;

pub trait TransitionMatrixOps: Sized + std::fmt::Debug + std::clone::Clone {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool;
    fn get_cur_pos(&self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix]) -> bool;
    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool;
    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool);
    fn set_cur_pos(&mut self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix], value: bool);
    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool);
    fn write<T: Write>(&self, io: &mut T) -> Result<()>;
    fn read<T: Read>(io: &mut T) -> Result<Self>;
    fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self;

    fn to_f32_mat(&self) -> Array2<f32>;
    fn from_f32_mat(mat: &Array2<f32>, current_shape: &[Ix], target_shape: &[Ix]) -> Self;

    fn get_target_shape(&self) -> &[Ix];
    fn get_current_shape(&self) -> &[Ix];

    // Slots are current/input, target/output
    fn slots(&self) -> (usize, usize);
    fn matrix_dims(&self) -> (usize, usize) {
        let (current_slots, target_slots) = self.slots();
        (target_slots.pow(2), current_slots.pow(2))
    }

    fn reinterpret_current_shape(&self, new_shape: ShapeVec) -> Self;

    fn n_ones(&self) -> usize;
    fn n_elements(&self) -> usize;
}

fn to_index(c1: &[Ix], c2: &[Ix], shape: &[Ix]) -> Ix {
    let mut ret = 0;
    for (v, scale) in c1.iter().zip(shape.iter()).chain(
        c2.iter().zip(shape.iter())) {
        // ret starts at 0 so the initial case is fine
        ret = (ret * scale) + v
    }
    ret
}

fn to_raw_index(c1: Ix, c2: Ix, shape: Ix) -> Ix {
    c2 + (shape * c1)
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

#[derive(Clone, Debug)]
pub struct DenseTransitionMatrix {
    data: Vec<BitVec>,
    target_shape: ShapeVec,
    current_shape: ShapeVec,
    current_len: usize,
    target_len: usize,
}

impl DenseTransitionMatrix {
    fn new(data: Vec<BitVec>, target_shape: ShapeVec, current_shape: ShapeVec) -> Self {
        let current_len: usize = current_shape.iter().product();
        let target_len: usize = target_shape.iter().product();
        Self {data, target_shape, current_shape, current_len, target_len}
    }

    pub fn update_row(&mut self, i1: Ix, i2: Ix,
                      other: &Self, k1: Ix, k2: Ix) {
        let our_row = to_raw_index(i1, i2, self.target_len);
        let their_row = to_raw_index(k1, k2, other.target_len);
        self.data[our_row].or(&other.data[their_row]);
    }
}

impl TransitionMatrixOps for DenseTransitionMatrix {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        let row = to_index(target1, target2, &self.target_shape);
        let column = to_index(current1, current2, &self.current_shape);
        self.data[row][column]
    }

    fn get_cur_pos(&self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix]) -> bool {
        let row = to_index(target1, target2, &self.target_shape);
        let column = to_raw_index(current1, current2, self.current_len);
        self.data[row][column]
    }

    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool {
        let row = to_raw_index(target1, target2, self.target_len);
        let column = to_raw_index(current1, current2, self.current_len);
        self.data[row][column]
    }

    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        let row = to_index(target1, target2, &self.target_shape);
        let column = to_index(current1, current2, &self.current_shape);
        self.data[row].set(column, value);
    }

    fn set_cur_pos(&mut self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix], value: bool) {
        let row = to_index(target1, target2, &self.target_shape);
        let column = to_raw_index(current1, current2, self.current_len);
        self.data[row].set(column, value);
    }

    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool) {
        let row = to_raw_index(target1, target2, self.target_len);
        let column = to_raw_index(current1, current2, self.current_len);
        self.data[row].set(column, value);
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        write_length_tagged_idxs(io, &self.target_shape)?;
        write_length_tagged_idxs(io, &self.current_shape)?;
        io.write_u64::<LittleEndian>(self.data.len() as u64)?;
        for row in &self.data {
            io.write_u64::<LittleEndian>(row.len() as u64)?;
            let storage = row.storage();
            io.write_u64::<LittleEndian>(storage.len() as u64)?;
            for item in storage.iter().copied() {
                io.write_u32::<LittleEndian>(item)?;
            }
        }
        Ok(())
    }

    fn read<T: Read>(io: &mut T) -> Result<Self> {
        let target_shape = read_length_tagged_idxs(io)?;
        let current_shape = read_length_tagged_idxs(io)?;
        let n_rows = io.read_u64::<LittleEndian>()? as usize;
        let data: Result<Vec<BitVec>> = (0..n_rows).map(|_| {
            let len = io.read_u64::<LittleEndian>()? as usize;
            let io_len = io.read_u64::<LittleEndian>()? as usize;
            let mut bits = BitVec::new();
            unsafe {
                bits.set_len(len);
                let storage = bits.storage_mut();
                storage.reserve(io_len);
                for _ in 0..io_len {
                    storage.push(io.read_u32::<LittleEndian>()?);
                }
            }
            Ok(bits)
        }).collect();
        let data = data?;

        Ok(Self::new(data, target_shape, current_shape))
    }

    fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self {
        let out_slots: usize = target_shape.iter().copied().product();
        let in_slots: usize = current_shape.iter().copied().product();
        let n_rows = out_slots.pow(2);
        let n_cols = in_slots.pow(2);
        let data = vec![BitVec::from_elem(n_cols, false); n_rows];
        Self::new(data, ShapeVec::from_slice(target_shape), ShapeVec::from_slice(current_shape))
    }

    fn to_f32_mat(&self) -> Array2<f32> {
        let floats = self.data.iter().flat_map(
            |v| v.iter().map(|b| if b { 1.0 } else { 0.0 })).collect();
        let dims = self.matrix_dims();
        Array2::from_shape_vec(dims, floats).unwrap()
    }

    fn from_f32_mat(mat: &Array2<f32>, current_shape: &[Ix], target_shape: &[Ix]) -> Self {
        let n_columns = current_shape.iter().copied().product::<usize>().pow(2);
        let data = mat.as_slice().unwrap().chunks(n_columns)
            .map(|s| s.iter().map(|f| !(f.abs() < EPSILON)).collect())
            .collect();
        Self::new(data, ShapeVec::from_slice(target_shape), ShapeVec::from_slice(current_shape))
    }

    fn get_target_shape(&self) -> &[Ix] {
        self.target_shape.as_slice()
    }

    fn get_current_shape(&self) -> &[Ix] {
        self.current_shape.as_slice()
    }

    fn reinterpret_current_shape(&self, new_shape: ShapeVec) -> Self {
        let ret = Self::new(self.data.clone(), self.target_shape.clone(),
                            new_shape);
        if ret.current_len != self.current_len {
            panic!("Incompatible input lengths in reinterpret: {} -> {}",
                   self.current_len, ret.current_len);
        }
        ret
    }

    fn slots(&self) -> (usize, usize) {
        (self.current_len, self.target_len)
    }

    fn n_ones(&self) -> usize {
        self.data.iter().map(
            |v| v.blocks().map(|b| b.count_ones()).sum::<u32>() as usize)
            .sum()
    }

    fn n_elements(&self) -> usize {
        self.current_len.pow(2) * self.target_len.pow(2)
    }
}

pub fn build_mat<T: TransitionMatrixOps>(ops: &OpSet) -> T {
    let out_shape = ops.out_shape.to_owned();

    let in_slots: usize = ops.in_shape.iter().product();
    let in_bound = in_slots as isize;
    let fold = ops.fused_fold;

    let mut ret = T::empty(&ops.in_shape, &out_shape);
    let gathers = ops.ops.gathers().unwrap();
    for op in gathers {
        let output_shape = op.data.shape();
        for (input1, output1) in op.data.into_iter().copied()
            .zip(ndarray::indices(output_shape)).filter(|&(i, _)| i >= 0 && i < in_bound) {
                for (input2, output2) in op.data.into_iter().copied()
                    .zip(ndarray::indices(output_shape)).filter(|&(i, _)| i >= 0 && i < in_bound) {
                        let mut out1: SmallVec<[usize; 6]> =
                            SmallVec::from_slice(output1.slice());
                        let mut out2: SmallVec<[usize; 6]> =
                            SmallVec::from_slice(output2.slice());
                        if fold {
                            out1.pop();
                            out2.pop();
                        }
                        ret.set_cur_pos(input1 as usize, input2 as usize,
                                        &out1, &out2,
                                        true);
                    }
            }
    }
    ret
}

#[derive(Clone, Debug)]
pub enum TransitionMatrix {
    Dense(DenseTransitionMatrix),
}

const DENSE_MATRIX_TAG: u8 = 3;
const FILE_MARKER: &[u8; 8] = b"SWIZFLOW";

impl TransitionMatrixOps for TransitionMatrix {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get(current1, current2, target1, target2)
        }
    }

    fn get_cur_pos(&self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix]) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get_cur_pos(current1, current2, target1, target2)
        }
    }

    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get_idxs(current1, current2, target1, target2)
        }
    }

    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set(current1, current2, target1, target2, value)
        }
    }

    fn set_cur_pos(&mut self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix], value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set_cur_pos(current1, current2, target1, target2, value)
        }
    }

    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set_idxs(current1, current2, target1, target2, value)
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

    fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self {
        TransitionMatrix::Dense(DenseTransitionMatrix::empty(current_shape, target_shape))
    }

    fn to_f32_mat(&self) -> Array2<f32> {
        match self {
            TransitionMatrix::Dense(d) => d.to_f32_mat()
        }
    }

    fn from_f32_mat(mat: &Array2<f32>, current_shape: &[Ix], target_shape: &[Ix]) -> Self {
       TransitionMatrix::Dense(DenseTransitionMatrix::from_f32_mat(mat, current_shape, target_shape))
    }

    fn get_target_shape(&self) -> &[Ix] {
        match self {
            TransitionMatrix::Dense(d) => d.get_target_shape()
        }
    }

    fn get_current_shape(&self) -> &[Ix] {
        match self {
            TransitionMatrix::Dense(d) => d.get_current_shape()
        }
    }

    fn reinterpret_current_shape(&self, new_shape: ShapeVec) -> Self {
        match self {
            TransitionMatrix::Dense(d) =>
                TransitionMatrix::Dense(d.reinterpret_current_shape(new_shape)),
        }
    }

    fn slots(&self) -> (usize, usize) {
        match self {
            TransitionMatrix::Dense(d) => d.slots()
        }
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

impl From<DenseTransitionMatrix> for TransitionMatrix {
    fn from(d: DenseTransitionMatrix) -> Self {
        TransitionMatrix::Dense(d)
    }
}

pub fn density<T: TransitionMatrixOps>(matrix: &T) -> f64 {
    (matrix.n_ones() as f64) / (matrix.n_elements() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempfile;

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
