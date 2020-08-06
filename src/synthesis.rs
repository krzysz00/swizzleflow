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
use crate::state::{DomRef,ProgState};
use crate::transition_matrix::{TransitionMatrix};
use crate::operators::SynthesisLevel;
use crate::operators::OpSetKind;

use crate::misc::{time_since, COLLECT_STATS};

use std::collections::{HashMap,BTreeMap};

use std::time::Instant;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::iter::FromIterator;

use itertools::Itertools;
use itertools::iproduct;

use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    All,
    First,
}

#[derive(Debug, Default)]
struct SearchLevelStats {
    tested: AtomicUsize,
    pruned: AtomicUsize,
    pruned_copy_count: AtomicUsize,
    failed: AtomicUsize,
    in_solution: AtomicUsize,
    value_checks: Option<Arc<Mutex<BTreeMap<usize, usize>>>>,
}

impl SearchLevelStats {
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

    #[cfg(not(feature = "stats"))]
    pub fn record_value_checks(&self, _count: usize) {}

    #[cfg(feature = "stats")]
    pub fn record_value_checks(&self, count: usize) {
        use crate::misc::loghist;
        if let Some(ref lock) = self.value_checks {
            let mut map = lock.lock().unwrap();
            *map.entry(count).or_insert(0) += 1;
        }
    }
}

impl Clone for SearchLevelStats {
    fn clone(&self) -> Self {
        Self { tested: AtomicUsize::new(self.tested.load(Ordering::SeqCst)),
               pruned: AtomicUsize::new(self.pruned.load(Ordering::SeqCst)),
               pruned_copy_count: AtomicUsize::new(self.pruned_copy_count.load(Ordering::SeqCst)),
               in_solution: AtomicUsize::new(self.in_solution.load(Ordering::SeqCst)),
               failed: AtomicUsize::new(self.failed.load(Ordering::SeqCst)),
               value_checks: self.value_checks.clone(),
        }
    }
}

impl Display for SearchLevelStats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Let's force a memory fence here to be safe
        let tested = self.tested.load(Ordering::SeqCst);
        let pruned = self.pruned.load(Ordering::Relaxed);
        let pruned_copy_count = self.pruned_copy_count.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let in_solution = self.in_solution.load(Ordering::Relaxed);

        let continued = tested - pruned - failed;

        write!(f, "tested={}; failed={}; pruned={}; copy_count={}; continued={}; in_solution={};",
               tested, failed, pruned, pruned_copy_count, continued, in_solution)?;

        if COLLECT_STATS {
            if let Some(ref lock) = self.value_checks {
                let map = lock.lock().unwrap();
                write!(f, " value_checks=[{:?}];",
                       map.iter().format(", "))?;
            }
        }
        Ok(())
    }
}

type ResultMap<'d> = RwLock<HashMap<ProgState<'d>, bool>>;
type SearchResultCache<'d> = Arc<ResultMap<'d>>;
type States<'d, 'l> = SmallVec<[Option<&'l ProgState<'d>>; 4]>;

// Invariant: any level with pruning enabled has a corresponding pruning matrix available
fn viable<'d>(current: &ProgState<'d>, target: &ProgState<'d>, matrix: &TransitionMatrix,
              copy_bounds: Option<(&[u32], &[u32])>, expected_syms: &[DomRef],
              _cache: &ResultMap<'d>, tracker: &SearchLevelStats,
              level: usize, print_pruned: bool, prune_fuel: usize) -> bool {
    let mut value_checks = 0;
    for (i, a) in expected_syms.iter().copied().enumerate() {
        for b in (&expected_syms[(i+1)..]).iter().copied().chain(std::iter::once(a)).take(prune_fuel) {
            value_checks += 1;
            for (t1, t2) in iproduct!(target.inv_state[a].iter().copied(),
                                      target.inv_state[b].iter().copied()) {
                let result = iproduct!(current.inv_state[a].iter().copied(),
                                       current.inv_state[b].iter().copied())
                    .any(|(c1, c2)| {
                        let v = matrix.get_idxs(c1, c2, t1, t2);
                        v
                    });
                if !result {
                    // if did_lookup {
                    //     cache.write().unwrap().insert(current.clone(), false);
                    // }
                    tracker.pruned();
                    tracker.record_value_checks(value_checks);
                    if print_pruned {
                        println!("pruned @ {}\n{}", level, current);
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
                for v in expected_syms.iter().copied() {
                    let actual = target.inv_state[v].len() as u32;
                    let min_copies: u32 = current.inv_state[v].iter()
                        .copied().map(|i| mins[i]).sum();
                    let max_copies: u32 = current.inv_state[v].iter()
                        .copied().map(|i| maxs[i]).sum();
                    if min_copies > actual || max_copies < actual {
                        tracker.pruned();
                        tracker.pruned_copy_count();
                        if print_pruned {
                            println!("pruned (copies) @ {}\n{}", level, current);
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

fn copy_replacing<'d, 'm, 'l: 'm>(s: &States<'d, 'l>, idx: usize,
                                  elem: Option<&'m ProgState<'d>>) -> States<'d, 'm> {
    let mut ret: States<'d, 'm> = s.iter().copied().map(|e| e ).collect();
    ret[idx] = elem;
    ret
}

fn search<'d, 'l, 'f>(curr_states: States<'d, 'l>, target: &ProgState<'d>,
                      levels: &'f [SynthesisLevel], expected_syms: &'f [Vec<DomRef>],
                      current_level: usize,
                      stats: &'f [SearchLevelStats], mode: Mode,
                      caches: &'f [SearchResultCache<'d>],
                      print: bool, print_pruned: bool, prune_fuel: usize) -> bool {
    let tracker = &stats[current_level];

    if current_level == levels.len() {
        tracker.checking();
        let current = curr_states[0].unwrap();
        if current == target {
            tracker.success();
            println!("solution:{}", &current.name);
            if print {
                println!("success_path [level {} @ lane 0]\n{}", current_level, current);
            }
            return true;
        }
        else {
            tracker.failed();
            return false;
        }
    }

    let level = &levels[current_level];
    let lane = level.lane;
    let current: &ProgState<'d> = curr_states[lane].unwrap();
    let cache = caches[current_level].clone();
    let proceed = |r: &ProgState<'d>| {
        tracker.checking();
        if level.prune {
            if !viable(&r, target,
                       level.matrix.as_ref().unwrap(),
                       level.copy_bounds.as_ref()
                       .map(|(s, b)| (s.as_slice(), b.as_slice())),
                       &expected_syms[level.expected_syms],
                       cache.as_ref(), &tracker,
                       current_level, print_pruned, prune_fuel) {
                return false;
            }
        }
        let new_states = copy_replacing(&curr_states,
                                        lane, Some(&r));
        let ret = search(new_states, target, levels, expected_syms,
                         current_level + 1, stats, mode, caches,
                         print, print_pruned, prune_fuel);
        if ret {
            tracker.success();
            if print {
                println!("success_path [level {} @ lane {}]\n{}", current_level, lane, r)
            }
        }
        ret
    };

    let ops = &level.ops.ops; // Get at the actual vector of gathers
    let ret =
        match ops {
            OpSetKind::Gathers(gathers, _) => {
                if level.ops.has_fold() {
                    match mode {
                        Mode::All => {
                            gathers.iter().map(
                                |o| {
                                    let res = current.gather_fold_by(o);
                                    proceed(&res)
                                })
                                .fold(false, |acc, new| new || acc)
                        }
                        Mode::First => {
                            gathers.iter().any(
                                |o| {
                                    let res = current.gather_fold_by(o);
                                    proceed(&res)
                                })
                        }
                    }
                }
                else {
                    match mode {
                        Mode::All => {
                            // Yep, this is meant not to be short-circuiting
                            gathers.iter().map(
                                |o| {
                                    let res = current.gather_by(o);
                                    proceed(&res)
                                })
                                .fold(false, |acc, new| new || acc)
                        }
                        Mode::First => {
                            gathers.iter().any(
                                |o| {
                                    let res = current.gather_by(o);
                                    proceed(&res)
                                })
                        }
                    }
                }
            },
            OpSetKind::Stack(from, to) => {
                tracker.checking();
                let to = *to;
                let to_stack: SmallVec<[&ProgState; 6]> =
                    from.iter().copied().map(
                        |idx| curr_states[idx].unwrap()).collect();
                let next =
                    if level.ops.has_fold() {
                        ProgState::stack_folding(&to_stack)
                    } else {
                        ProgState::stack(&to_stack)
                    };
                let mut new_states = copy_replacing(&curr_states,
                                                    to, Some(&next));
                for i in from.iter().copied() {
                    if i != to {
                        new_states[i] = None
                    }
                }

                if level.prune {
                    if !viable(&next, target,
                               level.matrix.as_ref().unwrap(),
                               level.copy_bounds.as_ref()
                               .map(|(s, b)| (s.as_slice(), b.as_slice())),
                               &expected_syms[level.expected_syms],
                               cache.as_ref(), &tracker,
                               current_level, print_pruned, prune_fuel) {
                        return false;
                    }
                }

                let ret = search(new_states, target, levels, expected_syms,
                                 current_level + 1, stats, mode, caches,
                                 print, print_pruned, prune_fuel);
                if ret {
                    tracker.success();
                    if print {
                        println!("success_path [level {} @ lane {}]\n{}", current_level, lane, next)
                    }
                }
                ret
            }
            OpSetKind::Split(into, copies) => {
                tracker.checking();
                let mut new_states = curr_states.clone();
                let to_copy = curr_states[*into];
                for i in copies.iter().copied() {
                    new_states[i] = to_copy;
                }
                search(new_states, target, levels, expected_syms,
                       current_level + 1, stats, mode, caches,
                       print, print_pruned, prune_fuel)
            }
        };
    ret
}

pub fn synthesize(start: Vec<Option<ProgState>>, target: &ProgState,
                  levels: &[SynthesisLevel],
                  expected_syms: &[Vec<DomRef>],
                  mode: Mode,
                  print: bool,
                  print_pruned: bool,
                  prune_fuel: usize,
                  spec_name: &str) -> bool {
    let n_levels = levels.len();
    let stats = (0..n_levels+1).map(|_| SearchLevelStats::new()).collect::<Vec<_>>();
    let caches: Vec<SearchResultCache>
        = (0..n_levels).map(|_| Arc::new(RwLock::new(HashMap::new()))).collect();

    let states: States = SmallVec::from_iter(start.iter().map(|e| e.as_ref()));
    let start_time = Instant::now();
    let ret = search(states, target, levels, expected_syms, 0, &stats,
                     mode, &caches, print, print_pruned, prune_fuel);
    let dur = time_since(start_time);

    for (idx, stats) in (&stats).iter().enumerate() {
        if COLLECT_STATS {
            println!("stats:: n_syms={};",
                     levels.get(idx).map_or(0, |l| expected_syms[l.expected_syms].len()));
        }
        println!("stats:{} name={}; lane={}; pruning={}; {}", idx,
                 levels.get(idx).map_or(&"(last)".into(), |x| &x.ops.name),
                 levels.get(idx).map_or(0, |x| x.lane),
                 levels.get(idx).map_or(false, |l| l.prune),
                 stats);
    }
    println!("search:{} success={}; mode={:?}; prune_fuel={}; time={};",
             spec_name, ret, mode, prune_fuel, dur);
    ret
}
