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
    let name = format!("select(d{} {} d{} + {})", operand1, op.name(), operand2, c);
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
                        op: Op, restrict: &BTreeMap<usize, Ix>) -> Gather {
    let name = format!("keep_if(d{} {} d{} + {})", operand1, op.name(), operand2, c);
    Gather::new(shape,
                |idxs: &[Ix]| {
                    if restrict.iter().any(|(&d, &n)| idxs[d] != n) {
                        to_opt_ix(idxs, shape)
                    }
                    else if op.perform(idxs[operand1] as isize,
                                  idxs[operand2] as isize + c) {
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
                         0..op_len, 0..op_len,
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(c, idx1, idx2, op)|
                    reg_select_gather(shape, idx1, idx2, *c, *op)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub fn cond_keep(shape: &[Ix], consts: &[isize],
                 restrict: &BTreeMap<usize, usize>) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let op_len = shape.len();
    ret.extend(iproduct!(consts.iter(),
                         (0..op_len).filter(|d| !restrict.contains_key(d)),
                         (0..op_len).filter(|d| !restrict.contains_key(d)),
                         &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
               .map(move |(c, idx1, idx2, op)|
                    cond_keep_gather(shape, idx1, idx2, *c, *op, restrict)));

    Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub type Operator = (usize, usize, isize, Op);
pub fn general_select_gather(out_shape: &[Ix], in_shape: &[Ix],
                             conds: &[Operator], axis: usize) -> Gather {
    let mut name = "select(".to_owned();
    for (a, b, c, op) in conds {
        let name_part = format!("d{} {} d{} + {}, ", a, op.name(), b, c);
        name.push_str(&name_part);
    }
    name.push(')');

    Gather::new(out_shape,
                |idxs: &[Ix]| {
                    let mut storage: SmallVec<[usize; 4]> = SmallVec::from_slice(&idxs);
                    storage[axis] = conds.len(); // fail through
                    for (i, (a, b, c, op)) in conds.iter().enumerate() {
                        let op1 = idxs[*a] as isize;
                        let op2 = idxs[*b] as isize + *c;
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

pub fn general_select(out_shape: &[Ix], in_shape: &[Ix], axis: usize,
                      consts: &[isize], dims: &[Ix]) -> Result<OpSetKind> {
    let mut ret = HashSet::new();

    let n = in_shape[axis] - 1;
    let operators: Vec<Operator> =
        iproduct!(consts.iter(), dims.iter(), dims.iter(),
                  &[Op::Eq, Op::Neq, Op::Lt, Op::Leq, Op::Gt, Op::Geq])
        .map(|(c, a, b, op)| (*a, *b, *c, *op)).collect();
    let indexes = combinations(n, operators.len());
    let mut storage = Vec::new();

    ret.extend(indexes.iter().map(|ixs| {
        storage.extend(ixs.iter().copied().map(|i| operators[i]));
        let ret = general_select_gather(out_shape, in_shape, &storage, axis);
        storage.clear();
        ret
    }));
    Ok(ret.into_iter().collect::<Vec<_>>().into())
}
