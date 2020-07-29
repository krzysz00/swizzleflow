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

use std::collections::{HashSet,BTreeMap};
use std::ops::{Add, Sub};

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Plus,
    Minus,
}

impl BinOp {
    fn name(self) -> &'static str {
        use BinOp::*;
        match self {
            Plus => "+", Minus => "-"
        }
    }

    fn perform<U: Copy, T: Copy + Add<Output = U>
               + Sub<Output = U>>(self, v1: T, v2: T) -> U {
        use BinOp::*;
        match self {
            Plus => v1 + v2,
            Minus => v1 - v2,
        }
    }
}

pub fn reg_select_gather(shape: &[Ix], operand1: usize, operand2: usize, c: isize, combine: BinOp,
                         op: Op) -> Gather {
    let name =
        if c != 0 {
            format!("keep_if(d{} {} {} {} d{})", operand1, op.name(), c, combine.name(), operand2)
        } else {
            format!("keep_if(d{} {} {}d{})", operand1, op.name(), combine.name(), operand2)
        };
    let copy_idx = shape.len() - 1;

    let mut in_shape = shape.to_vec();
    in_shape[copy_idx] = 2;
    Gather::new(shape,
                |idxs: &[Ix]| {
                    let mut storage: SmallVec<[usize; 4]> =
                        SmallVec::from_slice(&idxs[0..copy_idx]);
                    let op1 = idxs[operand1] as isize;
                    let op2 = combine.perform(c, idxs[operand2] as isize);
                    if op.perform(op1, op2) {
                        storage.push(0);
                    }
                    else {
                        storage.push(1);
                    }
                    to_opt_ix(&storage, &in_shape)
                }, name)
}

pub fn cond_keep_gather(shape: &[Ix], operand1: usize, operand2: usize, c: isize, combine: BinOp,
                        op: Op, restrict: &BTreeMap<usize, Ix>) -> Gather {
    let name =
        if c != 0 {
            format!("keep_if(d{} {} {} {} d{})", operand1, op.name(), c, combine.name(), operand2)
        } else {
            format!("keep_if(d{} {} {}d{})", operand1, op.name(), combine.name(), operand2)
        };
    Gather::new(shape,
                |idxs: &[Ix]| {
                    let v2 = combine.perform(c, idxs[operand2] as isize);
                    if restrict.iter().any(|(&d, &n)| idxs[d] != n) {
                        to_opt_ix(idxs, shape)
                    }
                    else if op.perform(idxs[operand1] as isize, v2) {
                        to_opt_ix(idxs, shape)
                    }
                    else {
                        -1
                    }
                }, name)
}

pub fn reg_select(shape: &[Ix], consts: &[isize]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    ret.extend(iproduct!(consts.iter(),
                         &[BinOp::Plus, BinOp::Minus],
                         0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(c, binop, idx1, idx2, op)|
                    reg_select_gather(shape, idx1, idx2, *c, *binop, *op)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub fn cond_keep(shape: &[Ix], consts: &[isize],
                 restrict: &BTreeMap<usize, usize>) -> Result<OpSetKind> {
    let mut ret = HashSet::new();
    let op_len = shape.len();

    let first_op = (0..op_len).filter(|d| !restrict.contains_key(d)).next()
        .expect("Conditional keep needs at least one dimension that can be used");
    let mut keep_all = cond_keep_gather(shape, first_op, first_op, 0, BinOp::Plus, Op::Eq, restrict);
    keep_all.name = "keep_if(true)".to_owned();
    ret.insert(keep_all);

    let mut keep_none = cond_keep_gather(shape, first_op, first_op, 0, BinOp::Plus, Op::Neq, restrict);
    keep_none.name = "keep_if(false)".to_owned();
    ret.insert(keep_none);

    ret.extend(iproduct!(consts.iter(),
                         &[BinOp::Plus, BinOp::Minus],
                         (0..op_len).filter(|d| !restrict.contains_key(d)),
                         (0..op_len).filter(|d| !restrict.contains_key(d)),
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(c, binop, idx1, idx2, op)|
                    cond_keep_gather(shape, idx1, idx2, *c, *binop, *op, restrict)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub type Operator = (usize, usize, isize, BinOp, Op);
pub fn general_select_gather(in_shape: &[Ix], out_shape: &[Ix],
                             conds: &[Operator], axis: usize) -> Gather {
    let mut name = "select(".to_owned();
    for (a, b, c, binop, op) in conds {
        let name_part = if *c != 0 {
            format!("keep_if(d{} {} {} {} d{})", a, op.name(), c, binop.name(), b)
        } else {
            format!("keep_if(d{} {} {}d{})", a, op.name(), binop.name(), b)
        };
        name.push_str(&name_part);
    }
    name.push(')');

    Gather::new(out_shape,
                |idxs: &[Ix]| {
                    let mut storage: SmallVec<[usize; 4]> = SmallVec::from_slice(&idxs);
                    storage[axis] = conds.len(); // fail through
                    for (i, (a, b, c, binop, op)) in conds.iter().enumerate() {
                        let op1 = idxs[*a] as isize;
                        let op2 = binop.perform(*c, idxs[*b] as isize);
                        if op.perform(op1, op2) {
                            storage[axis] = i;
                            return to_opt_ix(&storage, in_shape);
                        }
                    }
                    to_opt_ix(&storage, in_shape)
                }, name)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if n < 1 {
        panic!("Wanted {} < 1 conditions for a partition", n);
    }
    if n == 1 {
        (0..k).map(|i| vec![i]).collect()
    }
    else {
        let recurse = combinations(n - 1, k);
        let mut ret = Vec::with_capacity(recurse.len() * k);
        for i in 0..k {
            for v in &recurse {
                if !v.contains(&i) {
                    let mut clone = v.clone();
                    clone.push(i);
                    ret.push(clone);
                }
            }
        }
        ret
    }
}

pub fn general_select(in_shape: &[Ix], out_shape: &[Ix], axis: usize,
                      consts: &[isize], dims: &[Ix]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let n = in_shape[axis] - 1;
    let operators: Vec<Operator> =
        iproduct!(consts.iter(), &[BinOp::Plus, BinOp::Minus],
                  dims.iter(), dims.iter(),
                  &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
        .map(|(c, binop, a, b, op)| (*a, *b, *c, *binop, *op)).collect();
    let indexes = combinations(n, operators.len());
    let mut storage = Vec::new();

    ret.extend(indexes.iter().map(|ixs| {
        storage.extend(ixs.iter().copied().map(|i| operators[i]));
        let ret = general_select_gather(in_shape, out_shape, &storage, axis);
        storage.clear();
        ret
    }));
    Ok(ret.into_iter().collect::<Vec<_>>().into())
}
