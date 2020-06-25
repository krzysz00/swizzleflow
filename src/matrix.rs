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

use rustc_hash::FxHashSet;

use std::convert::{Into, From};
use std::io::{Write,Read};

use bit_vec::BitVec;

use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};

use crate::errors::*;

pub trait MatrixOps: Sized + std::fmt::Debug + std::clone::Clone + Into<Matrix> {
    fn empty(m: usize, n: usize) -> Self;
    fn with_row_size_hint(m: usize, n: usize, _hint: usize) -> Self {
        Self::empty(m, n)
    }

    fn get(&self, i: Ix, j: Ix) -> bool;
    fn set(&mut self, i: Ix, j: Ix, value: bool);

    fn write<T: Write>(&self, io: &mut T) -> Result<()>;
    fn read<T: Read>(io: &mut T) -> Result<Self>;

    // (m, n)
    fn dims(&self) -> (usize, usize);

    fn n_ones(&self) -> usize;
    fn n_elements(&self) -> usize {
        let (m, n) = self.dims();
        m * n
    }
}

#[derive(Clone, Debug)]
pub struct DenseMatrix {
    data: Vec<BitVec>,
    m: usize,
    n: usize,
}

impl DenseMatrix {
    pub fn update_row(&mut self, i: Ix,
                      other: &Self, k: Ix) {
        self.data[i].or(&other.data[k]);
    }
}

impl MatrixOps for DenseMatrix {
    fn empty(m: usize, n: usize) -> Self {
        let data = vec![BitVec::from_elem(n, false); m];
        Self { data, m, n }
    }

    fn get(&self, i: Ix, j: Ix) -> bool {
        self.data[i][j]
    }

    fn set(&mut self, i: Ix, j: Ix, value: bool) {
        self.data[i].set(j, value)
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        assert_eq!(self.data.len(), self.m);
        if self.m > 0 && self.n != self.data[0].len() {
            panic!("Row length mismatch {} != {}", self.n, self.data[0].len());
        }

        io.write_u64::<LittleEndian>(self.m as u64)?;
        io.write_u64::<LittleEndian>(self.n as u64)?;
        for row in &self.data {
            let storage = row.storage();
            io.write_u64::<LittleEndian>(storage.len() as u64)?;
            for item in storage.iter().copied() {
                io.write_u32::<LittleEndian>(item)?;
            }
        }
        Ok(())
    }

    fn read<T: Read>(io: &mut T) -> Result<Self> {
        let m = io.read_u64::<LittleEndian>()? as usize;
        let n = io.read_u64::<LittleEndian>()? as usize;
        let data: Result<Vec<BitVec>> = (0..m).map(|_| {
            let io_len = io.read_u64::<LittleEndian>()? as usize;
            let mut bits = BitVec::new();
            unsafe {
                bits.set_len(n);
                let storage = bits.storage_mut();
                storage.reserve(io_len);
                for _ in 0..io_len {
                    storage.push(io.read_u32::<LittleEndian>()?);
                }
            }
            Ok(bits)
        }).collect();
        let data = data?;
        Ok(Self { data, m, n })
    }

    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    fn n_ones(&self) -> usize {
        self.data.iter().map(
            |v| v.blocks().map(|b| b.count_ones()).sum::<u32>() as usize)
            .sum()
    }
}

pub type SparseMatRow = FxHashSet<Ix>;
#[derive(Clone, Debug)]
pub struct RowSparseMatrix {
    data: Vec<SparseMatRow>,
    m: usize,
    n: usize,
}

impl RowSparseMatrix {
    pub fn view_row_set_elems(&self, i: Ix,) -> &SparseMatRow {
        &self.data[i]
    }
}

impl MatrixOps for RowSparseMatrix {
    fn empty(m: usize, n: usize) -> Self {
        let data = vec![SparseMatRow::default(); m];
        Self { data, m, n }
    }

    fn with_row_size_hint(m: usize, n: usize, hint: usize) -> Self {
        let data = (0..m)
            .map(|_|SparseMatRow::with_capacity_and_hasher(hint, Default::default()))
            .collect();
        Self { data, m, n }
    }

    fn get(&self, i: Ix, j: Ix) -> bool {
        self.data[i].contains(&j)
    }

    fn set(&mut self, i: Ix, j: Ix, value: bool) {
        if value {
            self.data[i].insert(j);
        }
        else {
            self.data[i].remove(&j);
        }
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        io.write_u64::<LittleEndian>(self.m as u64)?;
        io.write_u64::<LittleEndian>(self.n as u64)?;
        for row in &self.data {
            io.write_u64::<LittleEndian>(row.len() as u64)?;
            for j in row.iter().copied() {
                io.write_u64::<LittleEndian>(j as u64)?;
            }
        }
        Ok(())
    }

    fn read<T: Read>(io: &mut T) -> Result<Self> {
        let m = io.read_u64::<LittleEndian>()? as usize;
        let n = io.read_u64::<LittleEndian>()? as usize;
        let data: Result<Vec<SparseMatRow>> = (0..m).map(|_| {
            let len = io.read_u64::<LittleEndian>()? as usize;
            let mut ret = SparseMatRow::with_capacity_and_hasher(len, Default::default());
            for _ in 0..len {
                let j = io.read_u64::<LittleEndian>()? as usize;
                ret.insert(j);
            }
            Ok(ret)
        }).collect();
        let data = data?;

        Ok(Self { data, m, n })
    }

    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    fn n_ones(&self) -> usize {
        self.data.iter().map(|v| v.len()).sum()
    }

}
#[derive(Clone, Debug)]
pub enum Matrix {
    Dense(DenseMatrix),
    RowSparse(RowSparseMatrix),
}

const DENSE_MATRIX_TAG: u8 = 3;
const ROW_SPARSE_MATRIX_TAG: u8 = 4;
const FILE_MARKER: &[u8; 8] = b"SWIZFLOW";

impl MatrixOps for Matrix {
    fn empty(i: usize, j: usize) -> Self {
        Matrix::Dense(DenseMatrix::empty(i, j))
    }

    fn get(&self, i: Ix, j: Ix) -> bool {
        match self {
            Matrix::Dense(d) => d.get(i, j),
            Matrix::RowSparse(s) => s.get(i, j),
        }
    }

    fn set(&mut self, i: Ix, j: Ix, value: bool) {
        match self {
            Matrix::Dense(d) => d.set(i, j, value),
            Matrix::RowSparse(s) => s.set(i, j, value),
        }
    }

    fn write<T: Write>(&self, io: &mut T) -> Result<()> {
        io.write_all(FILE_MARKER)?;
        match self {
            Matrix::Dense(d) => {
                io.write_u8(DENSE_MATRIX_TAG)?;
                Ok(d.write(io)?)
            },
            Matrix::RowSparse(s) => {
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
                let mat = DenseMatrix::read(io)?;
                Ok(Matrix::Dense(mat))
            }
            ROW_SPARSE_MATRIX_TAG => {
                let mat = RowSparseMatrix::read(io)?;
                Ok(Matrix::RowSparse(mat))
            }
            e => Err(ErrorKind::UnknownMatrixType(e).into())
        }
    }

    fn dims(&self) -> (usize, usize) {
        match self {
            Matrix::Dense(d) => d.dims(),
            Matrix::RowSparse(s) => s.dims(),
        }
    }

    fn n_ones(&self) -> usize {
        match self {
            Matrix::Dense(d) => d.n_ones(),
            Matrix::RowSparse(s) => s.n_ones(),
        }
    }
}

impl From<DenseMatrix> for Matrix {
    fn from(d: DenseMatrix) -> Self {
        Matrix::Dense(d)
    }
}

impl From<RowSparseMatrix> for Matrix {
    fn from(s: RowSparseMatrix) -> Self {
        Matrix::RowSparse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempfile;

    #[test]
    fn test_write_round_trip() {
        use std::io::{Seek,SeekFrom};

        const M: usize = 576;
        const N: usize = 479;

        let mut matrix = DenseMatrix::empty(M, N);
        for i in 0..M {
            for j in 0..N {
                if (i * j) % 4 == 0 {
                    matrix.set(i, j, true);
                }
            }
        }

        let mut file = tempfile().unwrap();
        matrix.write(&mut file).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let matrix2 = DenseMatrix::read(&mut file).unwrap();
        assert_eq!(matrix.data, matrix2.data);
        assert_eq!(matrix2.m, M);
        assert_eq!(matrix2.n, N);
    }
}
