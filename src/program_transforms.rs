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
use crate::errors::*;

use crate::misc::{extending_set, ShapeVec};
use crate::state::{Operation, Value, Domain, ProgState};
use crate::parser::{Statement, OpType};
use crate::operators::SearchStep;

use std::collections::{HashMap,BinaryHeap};
use std::cmp::Reverse;

use ndarray::ArrayD;

use smallvec::{SmallVec};

use itertools::Itertools;

fn remap_refs(stmt: &mut Statement, map: &[usize]) {
    for i in stmt.args.iter_mut() {
        *i = map[*i];
    }
    for i in stmt.used_at.iter_mut() {
        *i = map[*i];
    }
}

pub fn bring_literals_up(parsed: Vec<Statement>) -> Vec<Statement> {
    let n_ops = parsed.len();

    let (inits, mut ops) = parsed.into_iter().enumerate()
        .partition::<Vec<_>, _>(|(_, s)| s.op.is_initial());

    // Shut off pruning on final operation
    let last_op = ops.len() - 1;
    ops[last_op].1.prune = false;
    let mut map = vec![None; n_ops];
    for (new, &(old, _)) in inits.iter().enumerate() {
        map[old] = Some(new);
    }
    let offset = inits.len();
    for (new, &(old, _)) in ops.iter().enumerate() {
        map[old] = Some(new + offset);
    }

    let map: Option<Vec<usize>> = map.into_iter().collect();
    let map = map.expect("no operations to be dropped");

    (inits.into_iter().chain(ops.into_iter()))
        .map(|(_, mut s)| { remap_refs(&mut s, &map); s})
        .collect::<Vec<Statement>>()
}

fn next_free_lane(free: &mut BinaryHeap<Reverse<usize>>, max_lanes: &mut usize) -> usize {
    if let Some(Reverse(lane)) = free.pop() {
        lane
    }
    else {
        let ret = *max_lanes;
        *max_lanes += 1;
        ret
    }
}

pub fn to_program(statements: Vec<Statement>) -> (Vec<Option<ArrayD<Value>>>, Vec<Operation>, usize) {
    // Assigning lanes is equivalent to register allocation
    // Except that we get to conjure up more registers whenever we want
    let n_statements = statements.len();
    let mut lanes = Vec::with_capacity(n_statements);
    let mut levels = Vec::with_capacity(n_statements);

    let mut lane_shapes = Vec::new();
    let mut var_names = Vec::with_capacity(n_statements);
    let mut max_lanes = 0;

    let n_initials = statements.iter().take_while(|s| s.op.is_initial()).count();
    let mut initials = Vec::with_capacity(n_initials);
    let mut ops = Vec::with_capacity(n_statements - n_initials);

    let mut lane_ends: HashMap<usize, SmallVec<[usize; 2]>> = HashMap::new();
    let mut free_lanes = BinaryHeap::new();
    for (idx, Statement { op, var, mut args,
                          in_shapes: _in_shapes, out_shape,
                          name, used_at, prune}) in statements.into_iter().enumerate() {
        if let Some(vars) = lane_ends.get(&idx) {
            for v in vars.iter().copied() {
                free_lanes.push(Reverse(lanes[v]))
            }
        }
        match op {
            OpType::Initial(literal) => {
                let lane = next_free_lane(&mut free_lanes, &mut max_lanes);
                lanes.push(lane);
                let level = literal.fold(usize::MAX, |x, v| std::cmp::min(x, v.level()));
                levels.push(level);
                extending_set(&mut initials, lane, Some(literal));
                var_names.push(var);
                assert!(args.is_empty());
            },
            OpType::Gathers(fns, fold_len) => {
                let arg_string = format!("({})", args.iter().copied()
                                         .map(|a| var_names[a].clone()).join(", "));
                let level = args.iter().copied().map(|a| levels[a]).min()
                    .unwrap_or(1) + if fold_len.is_some() { 1 } else { 0 };
                levels.push(level);

                args.iter_mut().for_each(|i| *i = lanes[*i]);

                let lane = next_free_lane(&mut free_lanes, &mut max_lanes);
                lanes.push(lane);
                var_names.push(var.clone());

                let mut drop_lanes = Vec::new();
                if let Some(vars) = lane_ends.get(&idx) {
                    drop_lanes.extend(vars.iter().copied()
                                      .map(|v| lanes[v])
                                      .filter(|&l| l != lane));
                }
                let op = Operation::new(fns, fold_len,
                                        lane_shapes.clone(), out_shape.clone(),
                                        var, arg_string, name,
                                        args, lane, drop_lanes,
                                        prune, level);
                ops.push(op);
            }
        };
        let lane = lanes[idx];
        extending_set(&mut lane_shapes, lane, Some(out_shape));
        if let Some(last_live) = used_at.iter().copied().max() {
            lane_ends.entry(last_live)
                .or_insert_with(|| SmallVec::<[usize; 2]>::new())
                .push(idx);
        }
        else {
            free_lanes.push(Reverse(lane));
        }
        if let Some(vars) = lane_ends.remove(&idx) {
            for v in vars {
                let their_lane = lanes[v];
                if their_lane != lane {
                    lane_shapes[lanes[v]] = None;
                }
            }
        }
    }
    for _ in initials.len()..max_lanes {
        initials.push(None);
    }
    for o in ops.iter_mut() {
        o.extend_shapes(max_lanes);
    }
    (initials, ops, max_lanes)
}

pub fn to_search_problem<'d>(
    domain: &'d Domain,
    // Initials have been padded with Nones
    initials: &[Option<ArrayD<Value>>], operations: &[Operation],
    target: ArrayD<Value>)
    -> Result<(ProgState<'d, 'static>, Vec<SearchStep>,
               ProgState<'d, 'static>, Vec<Option<ShapeVec>>)>
{
    let steps = operations.iter().map(|o| SearchStep::new(o.clone())).collect();
    let init = ProgState::new_from_spec(domain, initials.to_owned(), "[init];");
    let last_op = &operations[operations.len()-1];
    let target_shape = ShapeVec::from_slice(target.shape());
    let mut target_vec = vec![None; last_op.lane_in_lens.len()];
    target_vec[operations[operations.len()-1].out_lane] = Some(target);
    let target_state = ProgState::new_from_spec(domain, target_vec, "[target]");
    let mut last_op_shape = last_op.lane_in_shapes.clone();
    last_op_shape[last_op.out_lane] = Some(target_shape);
    let expected_shape = target_state.state.iter()
        .map(|ma| ma.as_ref().as_ref().map(|a| ShapeVec::from_slice(a.shape())))
        .collect();
    if last_op_shape != expected_shape {
        return Err(ErrorKind::BadGoalShape(last_op_shape, expected_shape).into());
    }
    Ok((init, steps, target_state, last_op_shape))
}
