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
use crate::state::{Gather,to_opt_ix};
use super::OpSetKind;

use crate::errors::*;

use ndarray::Ix;

use std::collections::HashSet;

use smallvec::SmallVec;

use itertools::iproduct;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Eq,
    Neq,
    Lt,
    Leq,
    Gt,
    Geq,
}

impl Op {
    fn name(self) -> &'static str {
        use Op::*;
        match self {
            Eq => "=", Neq => "!=",
            Lt => "<", Leq => "<=",
            Gt => ">", Geq => ">=",
        }
    }

    fn perform<T: Eq + Ord>(self, v1: T, v2: T) -> bool {
        use Op::*;
        match self {
            Eq => v1 == v2,
            Neq => v1 != v2,
            Lt => v1 < v2,
            Leq => v1 <= v2,
            Gt => v1 > v2,
            Geq => v1 >= v2,
        }
    }
}

pub fn reg_select_gather(shape: &[Ix], operand1: usize, operand2: usize, c: isize,
                         op: Op) -> Gather {
    let name = format!("select({} {} {} + {})", operand1, op.name(), operand2, c);
    let copy_idx = shape.len() - 1;

    let mut in_shape = shape.to_vec();
    in_shape[copy_idx] = 2;
    Gather::new(shape,
                |idxs: &[Ix]| {
                    let mut storage: SmallVec<[usize; 4]> =
                        SmallVec::from_slice(&idxs[0..copy_idx]);
                    let op1 = idxs[operand1] as isize;
                    let op2 = idxs[operand2] as isize + c;
                    if op.perform(op1, op2) {
                        storage.push(0);
                    }
                    else {
                        storage.push(1);
                    }
                    to_opt_ix(&storage, &in_shape)
                }, name)
}

pub fn cond_keep_gather(shape: &[Ix], operand1: usize, operand2: usize, c: isize,
                        op: Op) -> Gather {
    let name = format!("keep_if({} {} {} + {})", operand1, op.name(), operand2, c);
    Gather::new(shape,
                |idxs: &[Ix]| {
                    if op.perform(idxs[operand1] as isize,
                                  idxs[operand2] as isize + c) {
                        to_opt_ix(idxs, shape)
                    }
                    else {
                        -1
                    }
                }, name)
}

pub fn reg_select_no_const(shape: &[Ix]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    ret.extend(iproduct!(0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(idx1, idx2, op)|
                    reg_select_gather(shape, idx1, idx2, 0, *op)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub fn reg_select(shape: &[Ix]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    let n = shape[0] as isize;
    ret.extend(iproduct!(0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq],
                         &[0, 1, -1, n, -n])
               .map(move |(idx1, idx2, op, c)|
                    reg_select_gather(shape, idx1, idx2, *c, *op)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub fn cond_keep_no_consts(shape: &[Ix]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    ret.extend(iproduct!(0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(idx1, idx2, op)|
                    cond_keep_gather(shape, idx1, idx2, 0, *op)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub fn cond_keep(shape: &[Ix]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    let n = shape[0] as isize;
    ret.extend(iproduct!(0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq],
                         &[0, 1, -1, n, -n])
               .map(move |(idx1, idx2, op, c)|
                    cond_keep_gather(shape, idx1, idx2, *c, *op)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}
