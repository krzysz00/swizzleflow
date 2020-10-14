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
use crate::misc::{time_since, COLLECT_STATS};
use crate::program_transforms::linearize_program;
use crate::state::{ProgState, DomRef, Operation, OpType, Block};
use crate::transition_matrix::{TransitionMatrix};

use std::collections::{HashMap,BTreeMap};

use std::time::Instant;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc};
use parking_lot::{RwLock, Mutex, MappedRwLockWriteGuard, MappedRwLockReadGuard,
                  RwLockUpgradableReadGuard, RwLockReadGuard, RwLockWriteGuard};
use crossbeam_channel::{Sender, Receiver};
use crossbeam_utils::thread::{Scope, ScopedJoinHandle};
use itertools::Itertools;
use itertools::iproduct;

use ndarray::ArrayD;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    All,
    First,
}

#[derive(Debug, Default)]
struct BlockResults<'d> {
    results: Arc<RwLock<(AtomicBool, Vec<ProgState<'d, 'static>>)>>,
}

impl<'d> BlockResults<'d> {
    pub fn iter<'a>(&'a self) ->
        (Option<Sender<Option<ProgState<'d, 'static>>>>, BlockResultsIter<'a, 'd>)
        where 'd: 'a,
    {
        let lock = self.results.upgradable_read();
        if !lock.0.fetch_or(true, Ordering::SeqCst) {
            let lock = RwLockUpgradableReadGuard::upgrade(lock);
            let lock = RwLockWriteGuard::map(lock, |(_, ref mut vec)| vec);
            let (send, recv) = crossbeam_channel::unbounded();
            let ret = BlockResultsIter::Initial { lock, channel: recv };
            (Some(send), ret)
        }
        else {
            let lock = RwLockUpgradableReadGuard::downgrade(lock);
            let lock = RwLockReadGuard::map(lock, |(_, ref vec)| vec);
            (None, BlockResultsIter::Later { lock, pos: 0 })
        }
    }
}

#[derive(Debug)]
enum BlockResultsIter<'a, 'd: 'a> {
    Initial { lock: MappedRwLockWriteGuard<'a, Vec<ProgState<'d, 'static>>>,
              channel: Receiver<Option<ProgState<'d, 'static>>> },
    Later { lock: MappedRwLockReadGuard<'a, Vec<ProgState<'d, 'static>>>, pos: usize },
}
impl<'a, 'd> std::iter::Iterator for BlockResultsIter<'a, 'd>
where 'd: 'a
{
    type Item = ProgState<'d, 'static>;
    fn next(&mut self) -> Option<ProgState<'d, 'static>> {
        match self {
            BlockResultsIter::Initial { lock, channel } => {
                channel.recv().ok().and_then(|v| {
                    v.as_ref().map(|a| lock.push(a.clone()));
                    v
                })
            },
            BlockResultsIter::Later { lock, pos } => {
                let ret = lock.get(*pos).cloned();
                *pos += 1;
                ret
            }
        }
    }
}

#[derive(Debug, Default)]
struct SearchStepStats {
    tested: AtomicUsize,
    pruned: AtomicUsize,
    pruned_copy_count: AtomicUsize,
    failed: AtomicUsize,
    cache_writes: AtomicUsize,
    cache_hits: AtomicUsize,
    in_solution: AtomicUsize,
    value_checks: Option<Arc<Mutex<BTreeMap<usize, usize>>>>,
}

impl SearchStepStats {
    pub fn new() -> Self {
        if COLLECT_STATS {
            let mut ret = Self::default();
            ret.value_checks = Some(Arc::new(Mutex::new(BTreeMap::new())));
            ret
        }
        else {
            Self::default()
        }
    }

    pub fn success(&self) {
        self.in_solution.fetch_add(1, Ordering::Relaxed);
    }

    pub fn checking(&self) {
        self.tested.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pruned(&self) {
        self.pruned.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pruned_copy_count(&self) {
        self.pruned_copy_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn cache_write(&self) {
        self.cache_writes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(not(feature = "stats"))]
    pub fn record_value_checks(&self, _count: usize) {}

    #[cfg(feature = "stats")]
    pub fn record_value_checks(&self, count: usize) {
        if let Some(ref lock) = self.value_checks {
            let mut map = lock.lock().unwrap();
            *map.entry(count).or_insert(0) += 1;
        }
    }
}

impl Clone for SearchStepStats {
    fn clone(&self) -> Self {
        Self { tested: AtomicUsize::new(self.tested.load(Ordering::SeqCst)),
               pruned: AtomicUsize::new(self.pruned.load(Ordering::SeqCst)),
               pruned_copy_count: AtomicUsize::new(self.pruned_copy_count.load(Ordering::SeqCst)),
               in_solution: AtomicUsize::new(self.in_solution.load(Ordering::SeqCst)),
               failed: AtomicUsize::new(self.failed.load(Ordering::SeqCst)),
               cache_writes: AtomicUsize::new(self.cache_writes.load(Ordering::SeqCst)),
               cache_hits: AtomicUsize::new(self.cache_hits.load(Ordering::SeqCst)),
               value_checks: self.value_checks.clone(),
        }
    }
}

impl Display for SearchStepStats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Let's force a memory fence here to be safe
        let tested = self.tested.load(Ordering::SeqCst);
        let pruned = self.pruned.load(Ordering::Relaxed);
        let pruned_copy_count = self.pruned_copy_count.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let in_solution = self.in_solution.load(Ordering::Relaxed);
        let cache_writes = self.cache_writes.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);

        let continued = tested - pruned - failed;

        write!(f, "tested={}; failed={}; pruned={}; copy_count={}; cache_writes={}; cache_hits={}; continued={}; in_solution={};",
               tested, failed, pruned, pruned_copy_count, cache_writes, cache_hits, continued, in_solution)?;

        if COLLECT_STATS {
            if let Some(ref lock) = self.value_checks {
                let map = lock.lock();
                write!(f, " value_checks=[{:?}];",
                       map.iter().format(", "))?;
            }
        }
        Ok(())
    }
}

type ResultMap<'d> = RwLock<HashMap<ProgState<'d, 'static>, bool>>;
type SearchResultCache<'d> = Arc<ResultMap<'d>>;

// Invariant: any level with pruning enabled has a corresponding pruning matrix available
fn viable<'d, 'l>(current: &ProgState<'d, 'l>, target: &ProgState<'d, 'static>,
                  matrix: &TransitionMatrix,
                  copy_bounds: Option<&(Vec<u32>, Vec<u32>)>,
                  terms: &[DomRef],
                  cache: &ResultMap<'d>, tracker: &SearchStepStats,
                  step: usize, print_pruned: bool, prune_fuel: usize) -> bool {
    let mut value_checks = 0;
    for (i, a) in terms.iter().copied().enumerate() {
        for b in (terms[i+1..]).iter().copied()
            .chain(std::iter::once(a)).take(prune_fuel)
        {
            value_checks += 1;
            for (t1, t2) in iproduct!(target.inv_state[a].iter().copied(),
                                      target.inv_state[b].iter().copied()) {
                let result = iproduct!(current.inv_state[a].iter().copied(),
                                       current.inv_state[b].iter().copied())
                    .any(|(c1, c2)| {
                        let v = matrix.get_raw_idxs(c1, c2, t1, t2);
                        v
                    });
                if !result {
                    // if did_lookup {
                    //     cache.write().unwrap().insert(current.clone(), false);
                    // }
                    tracker.pruned();
                    tracker.record_value_checks(value_checks);
                    if print_pruned {
                        println!("pruned @ {}\n{}", step, current);
                        println!("v1 = {}, v2 = {}, t1 = {}, t2 = {}",
                                 target.domain.get_value(a), target.domain.get_value(b),
                                 t1, t2);
                    }
                    return false;
                }
            }
        }
        if i == 0 {
            if let Some((mins, maxs)) = copy_bounds {
                for v in terms.iter().copied() {
                    let actual = target.inv_state[v].len() as u32;
                    let min_copies: u32 = current.inv_state[v].iter()
                        .copied().map(|i| mins[i]).sum();
                    let max_copies: u32 = current.inv_state[v].iter()
                        .copied().map(|i| maxs[i]).sum();
                    if min_copies > actual || max_copies < actual {
                        tracker.pruned();
                        tracker.pruned_copy_count();
                        if print_pruned {
                            println!("pruned (copies) @ {}\n{}", step, current);
                            println!("v = {}, actual = {}, min = {}, max = {}",
                                 target.domain.get_value(v), actual, min_copies, max_copies);
                        }
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn search<'d, 'l, 'f>(current: &ProgState<'d, 'l>, target: &ProgState<'d, 'static>,
                      channel: &Sender<Option<ProgState<'d, 'static>>>,
                      ops: &[Operation], current_step: usize,
                      universes: &'f [Vec<DomRef>], literals: &'f [ArrayD<DomRef>],
                      stats: &'f [SearchStepStats], blocks: &'f [BlockResults<'d>],
                      caches: &'f [SearchResultCache<'d>], is_root: bool,
                      mode: Mode, print: bool, print_pruned: bool, prune_fuel: usize) -> bool {

    if current_step == ops.len() {
        if is_root {
            let tracker = stats.last().unwrap();
            tracker.checking();
            if current == target {
                tracker.success();
                channel.send(Some(current.deep_clone())).expect("relying results to succeed");
                println!("solution:{}", &current.name);
                if print {
                    println!("success_path [step {}]\n{}", current_step, current);
                }
                return true;
            }
            else {
                tracker.failed();
                return false;
            }
        }
        else {
            // All viable block results continue
            channel.send(Some(current.deep_clone())).expect("relaying results to succeed");
            return true;
        }
    }

    let op = &ops[current_step];
    let global_step = op.global_idx;
    let tracker = &stats[global_step];
    let cache = caches[global_step].clone();

    let proceed = |new_state: &ProgState<'d, '_>| {
        tracker.checking();
        if op.prune {
            if !viable(new_state, target,
                       op.abstractions.pairs_matrix.as_ref()
                       .expect("Pruning matrix to be in place"),
                       op.abstractions.copy_bounds.as_ref(),
                       &universes[op.universe_idx],
                       cache.as_ref(), &tracker,
                       current_step, print_pruned, prune_fuel) {
                return false;
            }
        }
        let ret = search(new_state, target, channel, ops,
                         current_step + 1,
                         universes, literals, stats, blocks, caches,
                         is_root, mode, print, print_pruned, prune_fuel);
        if ret {
            tracker.success();
            if print {
                println!("success_path [step {}]\n{}", current_step, new_state);
            }
        }
        ret
    };
    let ret =
        match &op.op {
            OpType::Apply { fns, summary: _summary } => {
                match mode {
                    Mode::All => {
                        fns.iter().map(
                            |f| {
                                let res = current.gather_by(f, op);
                                proceed(&res)
                            })
                        // Yep, this is meant not to be short-circuiting
                            .fold(false, |acc, new| new || acc)
                    }
                    Mode::First => {
                        fns.iter().any(
                            |f| {
                                let res = current.gather_by(f, op);
                                proceed(&res)
                            })
                    }
                }
            },
            OpType::Literal(idx) => {
                let literal = &literals[*idx];
                let res = current.set_value(literal, &op.op_name, op);
                proceed(&res)
            },
            OpType::Subprog(block) => {
                let args = current.new_from_block_args(&op.in_lanes, block.max_lanes);
                crossbeam_utils::thread::scope(|scope| {
                    let (maybe_sender, mut iter) = blocks[block.block_num].iter();
                    let _handle = maybe_sender.map(|sender| {
                        scope.spawn(move |_| {
                            search(&args, target,
                                   &sender,  &block.ops, 0,
                                   universes, literals, stats, blocks, caches,
                                   // Force All to ensure sensible semantics
                                   false, Mode::All, print, print_pruned, prune_fuel);
                            sender.send(None).expect("Sending to work");
                        });
                    });

                    match mode {
                        Mode::All => {
                            iter.map(|s| {
                                let res = current.get_block_result(s, block.out_lane, op);
                                proceed(&res)
                            }).fold(false, |acc, new| new || acc)
                        },
                        Mode::First => {
                            iter.any(|s| {
                                let res = current.get_block_result(s, block.out_lane, op);
                                proceed(&res)
                            })
                        }
                    }
                }).expect("threads not to have panicked")
            },
        };
    ret
}

pub fn synthesize<'d, 'l>(
    target: &ProgState<'d, 'static>,
    block: &Block, universes: &[Vec<DomRef>], literals: &[ArrayD<DomRef>],
    n_blocks: usize, n_stmts: usize,
    mode: Mode, print: bool, print_pruned: bool, prune_fuel: usize,
    spec_name: &str) -> Vec<ProgState<'d, 'static>>
{
    let start = ProgState::empty(target.domain, block.max_lanes);

    let stats = (0..n_stmts+1).map(|_| SearchStepStats::new()).collect::<Vec<_>>();
    let caches: Vec<SearchResultCache>
        = (0..n_stmts).map(|_| Arc::new(RwLock::new(HashMap::new()))).collect();
    let blocks: Vec<_> = (0..n_blocks).map(|_| BlockResults::default()).collect();
    let (sender, iter) = blocks[0].iter();
    let sender = sender.expect("search to not have happened");
    let (ret, dur) =
        crossbeam_utils::thread::scope(|scope| {
            let start_time = Instant::now();
            let _handle = scope.spawn(
                |_| {
                    search(&start, target, &sender, &block.ops, 0,
                           universes, literals, &stats, &blocks, &caches,
                           true, mode, print, print_pruned, prune_fuel);
                    sender.send(None).expect("sending to work");
                });
            let ret = iter.collect::<Vec<_>>();
            let dur = time_since(start_time);
            (ret, dur)
        }).expect("search to succeed");

    let ops = linearize_program(block);
    assert_eq!(ops.len(), n_stmts);
    for (idx, stats) in (&stats).iter().enumerate() {
        if COLLECT_STATS {
            println!("stats:: n_syms={};",
                     ops.get(idx).map_or(0, |op| {
                         universes[op.universe_idx].len()
                     }));
        }
        println!("stats:{} var={}; op_name={}; pruning={}; {}", idx,
                 ops.get(idx).map_or(&"(last)".into(), |op| &op.var),
                 ops.get(idx).map_or(&"(last)".into(), |op| &op.op_name),
                 ops.get(idx).map_or(false, |op| op.prune),
                 stats);
    }
    println!("search:{} success={}; mode={:?}; prune_fuel={}; time={};",
             spec_name, !ret.is_empty(), mode, prune_fuel, dur);
    ret
}
