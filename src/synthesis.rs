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
use crate::state::{ProgState};
use crate::transition_matrix::{TransitionMatrix};
use crate::operators::SearchStep;

use std::collections::{HashMap,BTreeMap};

use std::time::Instant;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Mutex};

use itertools::Itertools;
use itertools::iproduct;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    All,
    First,
}

#[derive(Debug, Default)]
struct SearchStepStats {
    tested: AtomicUsize,
    pruned: AtomicUsize,
    pruned_copy_count: AtomicUsize,
    failed: AtomicUsize,
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

type ResultMap<'d> = RwLock<HashMap<ProgState<'d, 'static>, bool>>;
type SearchResultCache<'d> = Arc<ResultMap<'d>>;

// Invariant: any level with pruning enabled has a corresponding pruning matrix available
fn viable<'d, 'l>(current: &ProgState<'d, 'l>, target: &ProgState<'d, 'static>,
                  matrix: &TransitionMatrix,
                  copy_bounds: Option<&(Vec<Vec<u32>>, Vec<Vec<u32>>)>,
                  Range {start: term_start, end: term_end }: Range<usize>,
                  _cache: &ResultMap<'d>, tracker: &SearchStepStats,
                  step: usize, print_pruned: bool, prune_fuel: usize) -> bool {
    let mut value_checks = 0;
    for (i, a) in (term_start..term_end).enumerate() {
        for b in (a+1..term_end).chain(std::iter::once(a)).take(prune_fuel) {
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
                        println!("pruned @ {}\n{}", step, current);
                        println!("v1 = {}, v2 = {}, t1 = ({}, {}), t2 = ({}, {})",
                                 target.domain.get_value(a), target.domain.get_value(b),
                                 t1.0, t1.1, t2.0, t2.1);
                    }
                    return false;
                }
            }
        }
        if i == 0 {
            if let Some((mins, maxs)) = copy_bounds {
                for v in term_start..term_end {
                    let actual = target.inv_state[v].len() as u32;
                    let min_copies: u32 = current.inv_state[v].iter()
                        .copied().map(|(a, e)| mins[a][e]).sum();
                    let max_copies: u32 = current.inv_state[v].iter()
                        .copied().map(|(a, e)| maxs[a][e]).sum();
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
                      steps: &'f [SearchStep], current_step: usize,
                      stats: &'f [SearchStepStats], mode: Mode,
                      caches: &'f [SearchResultCache<'d>],
                      print: bool, print_pruned: bool, prune_fuel: usize) -> bool {
    let tracker = &stats[current_step];

    if current_step == steps.len() {
        tracker.checking();
        if current == target {
            tracker.success();
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

    let step = &steps[current_step];
    let op = &step.op;
    let gathers = &step.op.fns;
    let cache = caches[current_step].clone();

    let proceed = |new_state: &ProgState<'d, '_>| {
        tracker.checking();
        if op.prune {
            if !viable(new_state, target,
                       step.matrix.as_ref().expect("Pruning matrix to be in place"),
                       step.copy_bounds.as_ref(),
                       new_state.domain.terms_in_level(op.term_level),
                       cache.as_ref(), &tracker,
                       current_step, print_pruned, prune_fuel) {
                return false;
            }
        }
        let ret = search(new_state, target, steps,
                         current_step + 1, stats, mode, caches,
                         print, print_pruned, prune_fuel);
        if ret {
            tracker.success();
            if print {
                println!("success_path [step {}]\n{}", current_step, new_state);
            }
        }
        ret
    };
    let ret =
        match mode {
            Mode::All => {
                gathers.iter().map(
                    |f| {
                        let res = current.gather_by(f, op);
                        proceed(&res)
                    })
                    // Yep, this is meant not to be short-circuiting
                    .fold(false, |acc, new| new || acc)
            }
            Mode::First => {
                gathers.iter().any(
                    |f| {
                        let res = current.gather_by(f, op);
                        proceed(&res)
                    })
            }
        };
    ret
}

pub fn synthesize<'d, 'l>(start: &ProgState<'d, 'l>, target: &ProgState<'d, 'static>,
                  steps: &[SearchStep],
                  mode: Mode, print: bool, print_pruned: bool, prune_fuel: usize,
                          spec_name: &str) -> bool {
    let n_steps = steps.len();
    let stats = (0..n_steps+1).map(|_| SearchStepStats::new()).collect::<Vec<_>>();
    let caches: Vec<SearchResultCache>
        = (0..n_steps).map(|_| Arc::new(RwLock::new(HashMap::new()))).collect();

    let start_time = Instant::now();
    let ret = search(start, target, steps, 0, &stats,
                     mode, &caches, print, print_pruned, prune_fuel);
    let dur = time_since(start_time);

    for (idx, stats) in (&stats).iter().enumerate() {
        if COLLECT_STATS {
            println!("stats:: n_syms={};",
                     steps.get(idx).map_or(0, |step| {
                         let range = target.domain.terms_in_level(step.op.term_level);
                         range.end - range.start
                     }));
        }
        println!("stats:{} name={}; pruning={}; {}", idx,
                 steps.get(idx).map_or(&"(last)".into(), |step| &step.op.op_name),
                 steps.get(idx).map_or(false, |l| l.op.prune),
                 stats);
    }
    println!("search:{} success={}; mode={:?}; prune_fuel={}; time={};",
             spec_name, ret, mode, prune_fuel, dur);
    ret
}
