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
use ndarray::{Ix,Array2,Axis,Dimension};
use std::io::{Write,Read};
use std::io;

use crate::operators::Operators;
use crate::misc::{EPSILON,ShapeVec};

use bit_vec::BitVec;

use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};


// The strides need to start with 1
pub trait TransitionMatrix: Sized {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool;
    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool);
    fn write<T: Write>(&self, io: &mut T) -> io::Result<()>;
    fn read<T: Read>(io: &mut T) -> io::Result<Self>;
    fn empty(len: usize, out_shape: &[Ix], in_shape: &[Ix]) -> Self;
    // The shape is [target1, target2, current1, current2]
    fn to_f32_mat(&self) -> Array2<f32>;
    fn from_f32_mat(mat: &Array2<f32>, out_shape: &[Ix], in_shape: &[Ix]) -> Self;

    fn n_ones(&self) -> usize;
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

#[derive(Clone,Debug)]
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

impl TransitionMatrix for DenseTransitionMatrix {
    fn get(&self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix]) -> bool {
        self.data[to_index(target1, target2, &self.target_shape,
                           current1, current2, &self.current_shape)]
    }

    fn set(&mut self, current1: &[Ix], current2: &[Ix], target1: &[Ix], target2: &[Ix], value: bool) {
        let index = to_index(target1, target2, &self.target_shape,
                             current1, current2, &self.current_shape);
        self.data.set(index, value);
    }

    fn write<T: Write>(&self, io: &mut T) -> io::Result<()> {
        write_length_tagged_idxs(io, &self.target_shape)?;
        write_length_tagged_idxs(io, &self.current_shape)?;
        io.write_u64::<LittleEndian>(self.data.len() as u64)?;
        io.write_all(&mut self.data.to_bytes())
    }

    fn read<T: Read>(io: &mut T) -> io::Result<Self> {
        let target_shape = read_length_tagged_idxs(io)?;
        let current_shape = read_length_tagged_idxs(io)?;
        let len = io.read_u64::<LittleEndian>()? as usize;
        let io_len = (len + 7) % 8;
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
        let bits = mat.as_slice().unwrap().iter().map(|f| f.abs() < EPSILON).collect();
        Self::new(bits, ShapeVec::from_slice(out_shape), ShapeVec::from_slice(in_shape))
    }

    fn n_ones(&self) -> usize {
        self.data.iter().filter(|x| *x).count()
    }
}

pub fn build_mat<T: TransitionMatrix>(ops: &Operators) -> T {
    let out_slots: usize = ops.out_shape.iter().product();
    let in_slots: usize = ops.in_shape.iter().product();
    let len = (out_slots.pow(2)) * (in_slots.pow(2));
    let mut ret = T::empty(len, &ops.out_shape, &ops.in_shape);
    for op in &ops.ops {
        let axis_num = op.data.ndim() - 1;
        let output_shape = &op.data.shape()[0..axis_num];
        for (input1, output1) in op.data.lanes(Axis(axis_num)).into_iter()
            .zip(ndarray::indices(output_shape)) {
                for (input2, output2) in op.data.lanes(Axis(axis_num)).into_iter()
                    .zip(ndarray::indices(output_shape)) {
                        ret.set(input1.as_slice().unwrap(), input2.as_slice().unwrap(),
                                output1.slice(), output2.slice(),
                                true);
                    }
            }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swizzle_ops::{simple_fans,OpAxis};

    #[test]
    fn correct_length_trove_rows() {
        let small_fans = simple_fans(3, 4, OpAxis::Rows);
        let matrix: DenseTransitionMatrix = build_mat(&small_fans);
        assert_eq!(matrix.n_ones(), 488);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn correct_length_big_matrix() {
        use crate::swizzle_ops::simple_rotations;
        let big_matrix: DenseTransitionMatrix = build_mat(&simple_rotations(4, 32, OpAxis::Columns));
        assert_eq!(big_matrix.n_ones(), 246272);
    }
}
