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
use crate::state::{OpType, Operation, Value, DomRef, Domain, ProgState};
use crate::parser;
use crate::parser::{Statement};

use std::collections::{HashMap,BinaryHeap,BTreeSet};
use std::cmp::Reverse;
use std::iter::FromIterator;

use ndarray::{ArrayD, Ix};

use smallvec::{SmallVec};

use itertools::Itertools;

#[derive(Clone, Debug)]
pub enum UniverseDef {
    // Literal refers to the array of literals,
    // all others to the array of universe definitions
    Literal(Ix),
    Union(SmallVec<[Ix; 3]>),
    Fold(Ix),
}

fn union_universes(args: Vec<&[DomRef]>) -> Vec<DomRef> {
    args.into_iter().flat_map(|s| s.iter()).copied().dedup().collect()
}

fn fold_universe(domain: &Domain, universe: &[DomRef]) -> Vec<DomRef> {
    let current_set = BTreeSet::from_iter(universe.iter().copied());
    let superterms: BTreeSet<DomRef> = universe.iter().copied()
        .flat_map(|idx| domain.imm_superterms(idx).iter().copied())
        .collect();
    superterms.into_iter()
        .filter(|idx| domain.subterms_all_within(*idx, &current_set))
        .collect()
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

pub fn to_program(statements: Vec<Statement>)
                  -> (Vec<ArrayD<Value>>, // literals,
                      Vec<Operation>, // operations
                      Vec<UniverseDef>, // universe definitons
                      Vec<Option<ShapeVec>>, // output shape
                      usize) // max lanes
{
    // Assigning lanes is equivalent to register allocation
    // Except that we get to conjure up more registers whenever we want
    let n_statements = statements.len();
    let mut lanes = Vec::with_capacity(n_statements);
    let mut universes = Vec::with_capacity(n_statements);

    let mut lane_shapes = Vec::new();
    let mut var_names = Vec::with_capacity(n_statements);
    let mut max_lanes = 0;

    let mut literals = Vec::new();
    let mut ops = Vec::with_capacity(n_statements);
    let mut universe_defs = Vec::new();

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
        let lane = next_free_lane(&mut free_lanes, &mut max_lanes);
        lanes.push(lane);
        var_names.push(var.clone());
        let (op, fold_len, arg_string, universe_idx) =
            match op {
                parser::OpType::Initial(literal) => {
                    let literal_idx = literals.len();
                    literals.push(literal);

                    let universe_idx = universe_defs.len();
                    universe_defs.push(UniverseDef::Literal(literal_idx));

                    assert!(args.is_empty());
                    (OpType::literal_idx(literal_idx), None,
                     "".to_owned(), universe_idx)
                },
                parser::OpType::Gathers(fns, fold_len) => {
                    let arg_string = format!("({})", args.iter().copied()
                                             .map(|a| var_names[a].clone()).join(", "));
                    let universe_idx = if args.len() > 1 {
                        let idx = universe_defs.len();
                        universe_defs.push(
                            UniverseDef::Union(args.iter().copied()
                                               .map(|a| universes[a]).collect()));
                        idx
                    } else { universes[args.get(0).copied()
                                       .expect("functions to have an argument")] };
                    let universe_idx =
                        if fold_len.is_some() {
                            let idx = universe_defs.len();
                            universe_defs.push(UniverseDef::Fold(universe_idx));
                            idx
                        } else { universe_idx };
                    // Arguments now refer to lanes, not variables
                    args.iter_mut().for_each(|i| *i = lanes[*i]);

                    (OpType::fns(fns, args.len()), fold_len, arg_string, universe_idx)
                }
            };
        universes.push(universe_idx);
        let mut drop_lanes = Vec::new();
        if let Some(vars) = lane_ends.get(&idx) {
            drop_lanes.extend(vars.iter().copied()
                              .map(|v| lanes[v])
                              .filter(|&l| l != lane));
        }
        let op = Operation::new(op, fold_len,
                                lane_shapes.clone(), out_shape.clone(),
                                var, arg_string, name,
                                args, lane, drop_lanes,
                                prune, universe_idx);
        ops.push(op);

        extending_set(&mut lane_shapes, lane, Some(out_shape));
        if let Some(last_live) = used_at.iter().copied().max() {
            lane_ends.entry(last_live)
                .or_insert_with(|| SmallVec::<[usize; 2]>::new())
                .push(idx);
        }
        else {
            if idx != n_statements - 1 {
                println!("WARNING - unused variable {}", ops[ops.len()-1].var);
            }
            lane_ends.entry(idx + 1)
                .or_insert_with(|| SmallVec::<[usize; 2]>::new())
                .push(idx);
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
    for o in ops.iter_mut() {
        o.extend_shapes(max_lanes);
    }

    let last_op = &ops[ops.len() - 1];
    let mut last_op_shape = last_op.lane_in_shapes.clone();
    last_op_shape[last_op.out_lane] = Some(last_op.out_shape.clone());
    for l in last_op.drop_lanes.iter().copied() {
        last_op_shape[l] = None;
    }

    (literals, ops, universe_defs, last_op_shape, max_lanes)
}

pub fn to_search_problem<'d>(
    domain: &'d Domain,
    // Initials have been padded with Nones
    literals: &[ArrayD<Value>],
    operations: &[Operation],
    target: ArrayD<Value>,
    expected_target_state_shape: &[Option<ShapeVec>],
    universe_defs: &[UniverseDef])
    -> Result<(Vec<ArrayD<DomRef>>, // literals
               ProgState<'d, 'static>, // target
               Vec<Vec<DomRef>>)> // universes
{
    let literals: Vec<ArrayD<DomRef>> =
        literals.iter().map(|a| domain.literal_to_refs(a.view())).collect();

    let last_op = &operations[operations.len()-1];
    let mut target_vec = vec![None; last_op.lane_in_lens.len()];
    target_vec[last_op.out_lane] = Some(target);
    let target_state = ProgState::new_from_spec(domain, target_vec, "[target]", None);
    let target_state_shape = target_state.state.iter()
        .map(|ma| ma.as_ref().as_ref().map(|a| ShapeVec::from_slice(a.shape())))
        .collect();
    if target_state_shape != expected_target_state_shape {
        return Err(ErrorKind::BadGoalShape(expected_target_state_shape.to_owned(),
                                           target_state_shape).into());
    }

    let mut universes: Vec<Vec<DomRef>> = Vec::with_capacity(universe_defs.len());
    for def in universe_defs {
        universes.push(
            match def {
                UniverseDef::Literal(idx) => {
                    literals[*idx]
                        .iter().copied()
                        .filter(|&x| x != crate::state::NOT_IN_DOMAIN
                                && x != crate::state::ZERO)
                        .collect::<Vec<DomRef>>()
                    },
                    UniverseDef::Union(ref args) => {
                        let args = args.iter().copied()
                            .map(|x| universes[x].as_ref()).collect();
                        union_universes(args)
                    },
                    UniverseDef::Fold(prev) => {
                        fold_universe(domain, &universes[*prev])
                    }
                });
        }

    Ok((literals, target_state, universes))
}
