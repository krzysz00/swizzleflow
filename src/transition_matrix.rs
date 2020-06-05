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

use fnv::FnvHashSet;

use std::io::{Write,Read,BufReader,BufWriter};
use std::io;
use std::path::Path;

use crate::state::to_ix;
use crate::operators::OpSet;
use crate::misc::{ShapeVec,open_file,create_file};
use crate::errors::*;

use bit_vec::BitVec;

use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};

pub trait TransitionMatrixOps: Sized + std::fmt::Debug + std::clone::Clone {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        let c_shape = self.get_current_shape();
        let t_shape = self.get_target_shape();
        self.get_idxs(to_ix(current1, c_shape), to_ix(current2, c_shape),
                      to_ix(target1, t_shape), to_ix(target2, t_shape))
    }
    fn get_cur_pos(&self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix]) -> bool {
        let t_shape = self.get_target_shape();
        self.get_idxs(current1, current2,
                      to_ix(target1, t_shape), to_ix(target2, t_shape))

    }
    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool;
    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        let (c1, c2, t1, t2) = {
            let c_shape = self.get_current_shape();
            let t_shape = self.get_target_shape();
            (to_ix(current1, c_shape), to_ix(current2, c_shape),
             to_ix(target1, t_shape), to_ix(target2, t_shape))
        };
        self.set_idxs(c1, c2, t1, t2, value)

    }
    fn set_cur_pos(&mut self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix], value: bool) {
        let (t1, t2) = {
            let t_shape = self.get_target_shape();
            (to_ix(target1, t_shape), to_ix(target2, t_shape))
        };
        self.set_idxs(current1, current2, t1, t2, value)

    }
    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool);
    fn write<T: Write>(&self, io: &mut T) -> Result<()>;
    fn read<T: Read>(io: &mut T) -> Result<Self>;
    fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self;
    fn with_row_size_hint(current_shape: &[Ix], target_shape: &[Ix], _hint: usize) -> Self {
        Self::empty(current_shape, target_shape)
    }

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

#[allow(dead_code)]
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
        let our_row = to_raw_index(i1, i2, self.current_len);
        let their_row = to_raw_index(k1, k2, other.current_len);
        self.data[our_row].or(&other.data[their_row]);
    }
}

impl TransitionMatrixOps for DenseTransitionMatrix {
    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool {
        let row = to_raw_index(current1, current2, self.current_len);
        let column = to_raw_index(target1, target2, self.target_len);
        self.data[row][column]
    }

    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool) {
        let row = to_raw_index(current1, current2, self.current_len);
        let column = to_raw_index(target1, target2, self.target_len);
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
        let in_slots: usize = current_shape.iter().copied().product();
        let out_slots: usize = target_shape.iter().copied().product();
        let n_rows = in_slots.pow(2);
        let n_cols = out_slots.pow(2);
        let data = vec![BitVec::from_elem(n_cols, false); n_rows];
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

pub type SparseMatRow = FnvHashSet<(Ix, Ix)>;
#[derive(Clone, Debug)]
pub struct RowSparseTransitionMatrix {
    data: Vec<SparseMatRow>,
    target_shape: ShapeVec,
    current_shape: ShapeVec,
    current_len: usize,
    target_len: usize,
}

impl RowSparseTransitionMatrix {
    fn new(data: Vec<SparseMatRow>,
           current_shape: ShapeVec, target_shape: ShapeVec) -> Self {
        let current_len: usize = current_shape.iter().product();
        let target_len: usize = target_shape.iter().product();
        Self {data, target_shape, current_shape, current_len, target_len}
    }

    pub fn view_row_set_elems(&self, current1: Ix, current2: Ix) -> &SparseMatRow {
        &self.data[to_raw_index(current1, current2, self.current_len)]
    }
}

impl TransitionMatrixOps for RowSparseTransitionMatrix {
    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool {
        let row = to_raw_index(current1, current2, self.current_len);
        self.data[row].contains(&(target1, target2))
    }

    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool) {
        let row = to_raw_index(current1, current2, self.current_len);
        let column = (target1, target2);
        if value {
            self.data[row].insert(column);
        }
        else {
            self.data[row].remove(&column);
        }
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        write_length_tagged_idxs(io, &self.target_shape)?;
        write_length_tagged_idxs(io, &self.current_shape)?;
        io.write_u64::<LittleEndian>(self.data.len() as u64)?;
        for row in &self.data {
            io.write_u64::<LittleEndian>(row.len() as u64)?;
            for (i1, i2) in row.iter().copied() {
                io.write_u32::<LittleEndian>(i1 as u32)?;
                io.write_u32::<LittleEndian>(i2 as u32)?;
            }
        }
        Ok(())
    }

    fn read<T: Read>(io: &mut T) -> Result<Self> {
        let target_shape = read_length_tagged_idxs(io)?;
        let current_shape = read_length_tagged_idxs(io)?;
        let n_rows = io.read_u64::<LittleEndian>()? as usize;
        let data: Result<Vec<SparseMatRow>> = (0..n_rows).map(|_| {
            let len = io.read_u64::<LittleEndian>()? as usize;
            let mut ret = SparseMatRow::with_capacity_and_hasher(len, Default::default());
            for _ in 0..len {
                let v1 = io.read_u32::<LittleEndian>()? as usize;
                let v2 = io.read_u32::<LittleEndian>()? as usize;
                ret.insert((v1, v2));
            }
            Ok(ret)
        }).collect();
        let data = data?;

        Ok(Self::new(data, current_shape, target_shape))
    }

    fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self {
        let in_slots: usize = current_shape.iter().copied().product();
        let n_rows = in_slots.pow(2);
        let data = vec![SparseMatRow::default(); n_rows];
        Self::new(data, ShapeVec::from_slice(current_shape), ShapeVec::from_slice(target_shape))
    }

    fn with_row_size_hint(current_shape: &[Ix], target_shape: &[Ix], hint: usize) -> Self {
        let in_slots: usize = current_shape.iter().copied().product();
        let n_rows = in_slots.pow(2);
        let data = (0..n_rows)
            .map(|_|SparseMatRow::with_capacity_and_hasher(hint, Default::default()))
            .collect();
        Self::new(data, ShapeVec::from_slice(current_shape), ShapeVec::from_slice(target_shape))
    }

    fn get_target_shape(&self) -> &[Ix] {
        self.target_shape.as_slice()
    }

    fn get_current_shape(&self) -> &[Ix] {
        self.current_shape.as_slice()
    }

    fn reinterpret_current_shape(&self, new_shape: ShapeVec) -> Self {
        let ret = Self::new(self.data.clone(),
                            new_shape, self.target_shape.clone());
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
        self.data.iter().map(|v| v.len()).sum()
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

    let gathers = ops.ops.gathers().unwrap();
    let n_ops = gathers.len();

    // Get actual last dimension
    let fold_dim = gathers.get(0)
        .and_then(|o| o.data.shape().get(out_shape.len()).copied())
        .unwrap_or(1);

    let mut ret = T::with_row_size_hint(&ops.in_shape, &out_shape, n_ops);
    for op in gathers {
        for (output1, input1) in op.data.into_iter().copied()
            .enumerate().filter(|&(_, i)| i >= 0 && i < in_bound) {
                for (output2, input2) in op.data.into_iter().copied()
                    .enumerate().filter(|&(_, i)| i >= 0 && i < in_bound) {
                        // Remove fold dimension if needed
                        let output1 = if fold {
                            output1 / fold_dim
                        } else {
                            output1
                        };
                        let output2 = if fold {
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

#[derive(Clone, Debug)]
pub enum TransitionMatrix {
    Dense(DenseTransitionMatrix),
    RowSparse(RowSparseTransitionMatrix),
}

const DENSE_MATRIX_TAG: u8 = 3;
const ROW_SPARSE_MATRIX_TAG: u8 = 4;
const FILE_MARKER: &[u8; 8] = b"SWIZFLOW";

impl TransitionMatrixOps for TransitionMatrix {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get(current1, current2, target1, target2),
            TransitionMatrix::RowSparse(s) => s.get(current1, current2, target1, target2),
        }
    }

    fn get_cur_pos(&self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix]) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get_cur_pos(current1, current2, target1, target2),
            TransitionMatrix::RowSparse(s) => s.get_cur_pos(current1, current2, target1, target2),
        }
    }

    fn get_idxs(&self, current1: Ix, current2: Ix, target1: Ix, target2: Ix) -> bool {
        match self {
            TransitionMatrix::Dense(d) => d.get_idxs(current1, current2, target1, target2),
            TransitionMatrix::RowSparse(s) => s.get_idxs(current1, current2, target1, target2),
        }
    }

    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set(current1, current2, target1, target2, value),
            TransitionMatrix::RowSparse(s) => s.set(current1, current2, target1, target2, value),
        }
    }

    fn set_cur_pos(&mut self, current1: Ix, current2: Ix, target1: &[Ix], target2: &[Ix], value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set_cur_pos(current1, current2, target1, target2, value),
            TransitionMatrix::RowSparse(s) => s.set_cur_pos(current1, current2, target1, target2, value),

        }
    }

    fn set_idxs(&mut self, current1: Ix, current2: Ix, target1: Ix, target2: Ix, value: bool) {
        match self {
            TransitionMatrix::Dense(d) => d.set_idxs(current1, current2, target1, target2, value),
            TransitionMatrix::RowSparse(s) => s.set_idxs(current1, current2, target1, target2, value),
        }
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        io.write_all(FILE_MARKER)?;
        match self {
            TransitionMatrix::Dense(d) => {
                io.write_u8(DENSE_MATRIX_TAG)?;
                Ok(d.write(io)?)
            },
            TransitionMatrix::RowSparse(s) => {
                io.write_u8(ROW_SPARSE_MATRIX_TAG)?;
                Ok(s.write(io)?)
            },
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
            ROW_SPARSE_MATRIX_TAG => {
                let mat = RowSparseTransitionMatrix::read(io)?;
                Ok(TransitionMatrix::RowSparse(mat))
            }
            e => Err(ErrorKind::UnknownMatrixType(e).into())
        }
    }

    fn empty(current_shape: &[Ix], target_shape: &[Ix]) -> Self {
        TransitionMatrix::Dense(DenseTransitionMatrix::empty(current_shape, target_shape))
    }

    fn get_target_shape(&self) -> &[Ix] {
        match self {
            TransitionMatrix::Dense(d) => d.get_target_shape(),
            TransitionMatrix::RowSparse(s) => s.get_target_shape(),
        }
    }

    fn get_current_shape(&self) -> &[Ix] {
        match self {
            TransitionMatrix::Dense(d) => d.get_current_shape(),
            TransitionMatrix::RowSparse(s) => s.get_current_shape(),
        }
    }

    fn reinterpret_current_shape(&self, new_shape: ShapeVec) -> Self {
        match self {
            TransitionMatrix::Dense(d) =>
                TransitionMatrix::Dense(d.reinterpret_current_shape(new_shape)),
            TransitionMatrix::RowSparse(s) =>
                TransitionMatrix::RowSparse(s.reinterpret_current_shape(new_shape)),
        }
    }

    fn slots(&self) -> (usize, usize) {
        match self {
            TransitionMatrix::Dense(d) => d.slots(),
            TransitionMatrix::RowSparse(s) => s.slots(),
        }
    }

    fn n_ones(&self) -> usize {
        match self {
            TransitionMatrix::Dense(d) => d.n_ones(),
            TransitionMatrix::RowSparse(s) => s.n_ones(),
        }
    }

    fn n_elements(&self) -> usize {
        match self {
            TransitionMatrix::Dense(d) => d.n_elements(),
            TransitionMatrix::RowSparse(s) => s.n_elements(),
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

    pub fn to_dense(&self) -> Self {
        match self {
            TransitionMatrix::Dense(d) => TransitionMatrix::Dense(d.clone()),
            TransitionMatrix::RowSparse(s) => {
                let mut ret = DenseTransitionMatrix::empty(s.get_current_shape(),
                                                           s.get_target_shape());
                let (m, _) = s.slots();
                for m1 in 0..m {
                    for m2 in 0..m {
                        for (n1, n2) in s.view_row_set_elems(m1, m2).iter().copied() {
                            ret.set_idxs(m1, m2, n1, n2, true);
                        }
                    }
                }
                TransitionMatrix::Dense(ret)
            }
        }
    }
}

impl From<DenseTransitionMatrix> for TransitionMatrix {
    fn from(d: DenseTransitionMatrix) -> Self {
        TransitionMatrix::Dense(d)
    }
}

impl From<RowSparseTransitionMatrix> for TransitionMatrix {
    fn from(s: RowSparseTransitionMatrix) -> Self {
        TransitionMatrix::RowSparse(s)
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
        use std::io::{Seek,SeekFrom};

        const M: usize = 3;
        const N: usize = 8;

        let dim = M * N;
        let size = dim.pow(2);
        let mut matrix = DenseTransitionMatrix::empty(&[M, N], &[M, N]);
        for x in 0..size.pow(2) {
            if x % 4 == 0 {
                let current = x / size;
                let target = x % size;
                matrix.set_idxs(current / dim, current % dim,
                                target / dim, target % dim, true);
            }
        }

        let mut file = tempfile().unwrap();
        matrix.write(&mut file).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let matrix2 = DenseTransitionMatrix::read(&mut file).unwrap();
        assert_eq!(matrix.data, matrix2.data);
        assert_eq!(matrix.current_shape, matrix2.current_shape);
        assert_eq!(matrix.target_shape, matrix2.target_shape);
    }
}
