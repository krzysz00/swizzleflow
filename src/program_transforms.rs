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
use crate::state::{OpType, Operation, Value, DomRef, Domain, ProgState, Block, BlockCopyMaps};
use crate::parser::{Statement, StmtType, Dependency, VarIdx};

use std::collections::{BinaryHeap,BTreeSet,BTreeMap};
use std::cmp::Reverse;
use std::iter::FromIterator;

use ndarray::{ArrayD, Ix};

use smallvec::{SmallVec};
use rustc_hash::FxHashMap;

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

fn to_var_idx(ix: VarIdx, n_deps: usize) -> usize {
    match ix {
        VarIdx::Dep(i) => i,
        VarIdx::Here(i) => i + n_deps
    }
}

fn to_program_rec(statements: Vec<Statement>,
                  n_blocks: &mut usize, global_counter: &mut usize,
                  literals: &mut Vec<ArrayD<Value>>,
                  universe_defs: &mut Vec<UniverseDef>,
                  dependencies: Vec<Dependency>,
                  dep_universes: Vec<usize>)
                  -> Block {
    // Assigning lanes is equivalent to register allocation
    // Except that we get to conjure up more registers whenever we want
    let block_num = *n_blocks;
    *n_blocks += 1;

    let n_deps = dependencies.len();
    let n_stmts = statements.len();
    let n_vars = n_deps + n_stmts;

    let mut ops = Vec::with_capacity(n_stmts);

    let mut lanes = Vec::with_capacity(n_vars);
    let mut universes = Vec::with_capacity(n_vars);

    let mut lane_shapes = Vec::new();
    let mut var_names = Vec::with_capacity(n_vars);
    let mut max_lanes = 0;

    let mut lane_ends: FxHashMap<usize, SmallVec<[usize; 2]>> = FxHashMap::default();
    let mut free_lanes = BinaryHeap::new();

    let stmt_is_block: Vec<bool> = statements.iter().map(|s| s.op.is_block()).collect();

    let block_args_maps = statements.iter().enumerate()
        .map(|(i, v)| (i, v.args.iter().copied().enumerate()
                       .map(|(idx, a)| (a, idx)).collect::<FxHashMap<_, _>>()))
        .collect::<FxHashMap<_, _>>();
    let mut blocks_to_update = FxHashMap::default();

    for (idx, (Dependency { parent_idx: _parent_idx,
                            used_at, var, shape },
               universe))
        in dependencies.into_iter().zip(dep_universes.into_iter()).enumerate()
    {
        let lane = next_free_lane(&mut free_lanes, &mut max_lanes);
        lanes.push(lane);
        var_names.push(var);
        universes.push(universe);

        extending_set(&mut lane_shapes, lane, Some(shape));
        // Args must be preserved to the end of the computation
        // so that abstractions are correctly computed
        lane_ends.entry(n_vars)
            .or_insert_with(|| SmallVec::<[usize; 2]>::new())
            .push(idx);

        for stmt in used_at.iter().copied() {
            for block in used_at.iter().copied().filter(|&i| stmt_is_block[i] && i > stmt) {
                blocks_to_update.entry(stmt)
                    .or_insert_with(BTreeMap::default)
                    .entry(block).or_insert_with(FxHashMap::default)
                    .insert(*block_args_maps.get(&block).unwrap()
                            .get(&VarIdx::Dep(idx)).unwrap()
                            , lane);

            }
        }
    }

    for (stmt_idx, Statement {
        op, var, args,
        in_shapes: _in_shapes, out_shape,
        name, used_at, prune}) in statements.into_iter().enumerate()
    {
        let var_idx = stmt_idx + n_deps;
        if let Some(vars) = lane_ends.get(&var_idx) {
            for v in vars.iter().copied() {
                free_lanes.push(Reverse(lanes[v]))
            }
        }
        // Arguments now refer to lanes, not variables
        let arg_lanes = args.iter().copied()
            .map(|i| lanes[to_var_idx(i, n_deps)])
            .collect::<Vec<_>>();

        let lane = next_free_lane(&mut free_lanes, &mut max_lanes);
        lanes.push(lane);
        var_names.push(var.clone());

        for stmt in used_at.iter().copied() {
            for block in used_at.iter().copied().filter(|&i| stmt_is_block[i] && i > stmt) {
                blocks_to_update.entry(stmt)
                    .or_insert_with(BTreeMap::default)
                    .entry(block).or_insert_with(FxHashMap::default)
                    .insert(*block_args_maps.get(&block).unwrap()
                            .get(&VarIdx::Here(stmt_idx)).unwrap(),
                            lane);
            }
        }

        let block_copy_maps: Option<crate::state::BlockCopyDat> =
            blocks_to_update.remove(&stmt_idx)
            .map(|block_map| {
                block_map.into_iter().map(|(block, push_map)| {
                    let pull_map = push_map.iter().map(|(&theirs, &ours)| (ours, theirs)).collect();
                    (block, BlockCopyMaps { push_map, pull_map })
                }).collect()});
        if let Some(block_updates) = block_copy_maps.as_ref() {
            println!("Update {} ({}) with {:?}", stmt_idx, var, block_updates);
        }
        let (op, fold_len, arg_string, universe_idx) =
            match op {
                StmtType::Initial(literal) => {
                    let literal_idx = literals.len();
                    literals.push(literal);

                    let universe_idx = universe_defs.len();
                    universe_defs.push(UniverseDef::Literal(literal_idx));

                    assert!(args.is_empty());
                    (OpType::literal_idx(literal_idx), None,
                     "".to_owned(), universe_idx)
                },
                StmtType::Gathers(fns, fold_len) => {
                    let arg_string = format!(
                        "({})", args.iter().copied()
                            .map(|a| var_names[to_var_idx(a, n_deps)].clone())
                            .join(", "));
                    let universe_idx = if args.len() > 1 {
                        let idx = universe_defs.len();
                        universe_defs.push(
                            UniverseDef::Union(args.iter().copied()
                                               .map(|a| universes[to_var_idx(a, n_deps)])
                                               .collect()));
                        idx
                    } else {
                        universes[to_var_idx(
                            args.get(0).copied()
                                .expect("functions to have an argument"), n_deps)]
                    };
                    let universe_idx =
                        if fold_len.is_some() {
                            let idx = universe_defs.len();
                            universe_defs.push(UniverseDef::Fold(universe_idx));
                            idx
                        } else { universe_idx };

                    (OpType::fns(fns, args.len()), fold_len, arg_string, universe_idx)
                }
                StmtType::Block { body, deps } => {
                    let arg_string = format!(
                        "({})", args.iter().copied()
                            .map(|a| var_names[to_var_idx(a, n_deps)].clone())
                            .join(", "));
                    let dep_universes = args.iter().copied()
                        .map(|i| universes[to_var_idx(i, n_deps)]).collect();
                    let block = to_program_rec(body, n_blocks,
                                               global_counter, literals,
                                               universe_defs, deps, dep_universes);
                    let universe_idx = block.ops.last()
                        .expect("Block to have statements in it")
                        .universe_idx;
                    (OpType::Subprog(block), None, arg_string, universe_idx)
                }
            };
        universes.push(universe_idx);

        let mut drop_lanes = Vec::new();
        if let Some(vars) = lane_ends.get(&var_idx) {
            drop_lanes.extend(vars.iter().copied()
                              .map(|v| lanes[v])
                              .filter(|&l| l != lane));
        }
        let global_idx = *global_counter;
        *global_counter += 1;

        let op = Operation::new(op, fold_len,
                                lane_shapes.clone(), out_shape.clone(),
                                var, arg_string, name,
                                arg_lanes, lane, drop_lanes,
                                prune, universe_idx, global_idx,
                                block_copy_maps);
        ops.push(op);

        extending_set(&mut lane_shapes, lane, Some(out_shape));
        if let Some(last_live) = used_at.iter().copied().max() {
            // used_at is on statements, we need vars
            lane_ends.entry(last_live + n_deps)
                .or_insert_with(|| SmallVec::<[usize; 2]>::new())
                .push(var_idx);
        }
        else {
            if var_idx != n_vars - 1 {
                println!("WARNING - unused variable {}", ops[ops.len()-1].var);
            }
            lane_ends.entry(var_idx + 1)
                .or_insert_with(|| SmallVec::<[usize; 2]>::new())
                .push(var_idx);
        }
        if let Some(vars) = lane_ends.remove(&var_idx) {
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

    let last_op = ops.last().expect("at least one operation");
    let out_lane = last_op.out_lane;
    let mut last_op_shape = last_op.lane_in_shapes.clone();
    last_op_shape[out_lane] = Some(last_op.out_shape.clone());
    for l in last_op.drop_lanes.iter().copied() {
        last_op_shape[l] = None;
    }

    println!("Block number {}: last shape {:?}", block_num, last_op_shape);
    Block::new(ops, block_num, max_lanes, out_lane, last_op_shape)
}

pub fn to_program(statements: Vec<Statement>)
                  -> (Vec<ArrayD<Value>>, // literals,
                      Block, // whole program
                      Vec<UniverseDef>, // universe definitons
                      usize, // number of blocks
                      usize) // number of operitons
{
    let mut n_blocks = 0;
    let mut global_count = 0;

    let mut literals = Vec::new();
    let mut universe_defs = Vec::new();

    let block = to_program_rec(statements,
                               &mut n_blocks, &mut global_count,
                               &mut literals, &mut universe_defs,
                               vec![], vec![]); // No dependencies
    (literals, block, universe_defs, n_blocks, global_count)
}

pub fn to_search_problem<'d>(
    domain: &'d Domain,
    // Initials have been padded with Nones
    literals: &[ArrayD<Value>],
    Block { ref ops,
            out_shape: ref expected_target_state_shape,
            max_lanes: _max_lanes, out_lane: _out_lane,
            block_num: _block_num }: &Block,
    target: ArrayD<Value>,
    universe_defs: &[UniverseDef])
    -> Result<(Vec<ArrayD<DomRef>>, // literals
               ProgState<'d, 'static>, // target
               Vec<Vec<DomRef>>)> // universes
{
    let expected_target_state_shape = expected_target_state_shape.as_slice();
    let literals: Vec<ArrayD<DomRef>> =
        literals.iter().map(|a| domain.literal_to_refs(a.view())).collect();

    let last_op = &ops[ops.len()-1];
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

fn linearize_program_rec<'u>(ops: &[Operation], ret: &mut Vec<Operation>) {
    for op in ops.into_iter() {
        if let Some(b) = op.op.block() {
            linearize_program_rec(&b.ops, ret);
        }
        assert_eq!(op.global_idx, ret.len());
        ret.push(op.clone());
    }
}

pub fn linearize_program(block: &Block) -> Vec<Operation> {
    let mut ret = Vec::new();
    linearize_program_rec(&block.ops, &mut ret);
    ret
}
