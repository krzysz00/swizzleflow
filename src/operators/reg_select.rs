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
use crate::state::Gather;
use super::OpSet;

use crate::misc::ShapeVec;
use crate::errors::*;

use ndarray::Ix;

use std::collections::HashSet;

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

pub fn reg_select(shape: &[Ix], operand1: usize, operand2: usize, op: Op) -> Gather {
    let name = format!("select({} {} {})", operand1, op.name(), operand2);
    let copy_idx = shape.len() - 1;
    Gather::new(shape.len(), shape,
                move |idxs: &[Ix], out: &mut Vec<Ix>| {
                    out.extend(&idxs[0..copy_idx]);
                    let op1 = idxs[operand1];
                    let op2 = idxs[operand2];
                    if op.perform(op1, op2) {
                        out.push(0);
                    }
                    else {
                        out.push(1);
                    }
                }, name)
}

pub fn reg_select_no_const(shape: &[Ix]) -> Result<OpSet> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    ret.extend(iproduct!(0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(idx1, idx2, op)| reg_select(shape, idx1, idx2, *op)));

    let out_shape = ShapeVec::from_slice(shape);
    let mut in_shape = out_shape.clone();
    in_shape[op_len - 1] = 2;

    let name = "regSelNC";
    Ok(OpSet::new(name, ret.into_iter().collect(), in_shape, out_shape, false))
}
