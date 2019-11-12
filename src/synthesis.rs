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
use crate::transition_matrix::{TransitionMatrix,TransitionMatrixOps};
use crate::operators::SynthesisLevel;
use crate::operators::OpSetKind;

use crate::misc::{time_since};

use std::collections::HashMap;

use std::time::Instant;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::iter::FromIterator;

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
    failed: AtomicUsize,
    succeeded: AtomicUsize,
}

impl SearchLevelStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn success(&self) {
        self.succeeded.fetch_add(1, Ordering::Relaxed);
    }

    pub fn checking(&self) {
        self.tested.fetch_add(1, Ordering::Relaxed);
    }

    pub fn pruned(&self) {
        self.pruned.fetch_add(1, Ordering::Relaxed);
    }

    pub fn failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }
}

impl Clone for SearchLevelStats {
    fn clone(&self) -> Self {
        Self { tested: AtomicUsize::new(self.tested.load(Ordering::SeqCst)),
               pruned: AtomicUsize::new(self.pruned.load(Ordering::SeqCst)),
               succeeded: AtomicUsize::new(self.succeeded.load(Ordering::SeqCst)),
               failed: AtomicUsize::new(self.failed.load(Ordering::SeqCst)),
        }
    }
}

impl Display for SearchLevelStats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Let's force a memory fence here to be safe
        let tested = self.tested.load(Ordering::SeqCst);
        let found = self.succeeded.load(Ordering::Relaxed);
        let pruned = self.pruned.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);

        let continued = tested - found - pruned - failed;

        write!(f, "tested({}), found({}), failed({}), pruned({}), continued({})",
               tested, found, failed, pruned, continued)
    }
}

type ResultMap<'d> = RwLock<HashMap<ProgState<'d>, bool>>;
type SearchResultCache<'d> = Arc<ResultMap<'d>>;
type States<'d, 'l> = SmallVec<[Option<&'l ProgState<'d>>; 4]>;

// Invariant: any level with pruning enabled has a corresponding pruning matrix available
fn viable<'d>(current: &ProgState<'d>, target: &ProgState<'d>, matrix: &TransitionMatrix,
              expected_syms: &[DomRef],
              _cache: &ResultMap<'d>, tracker: &SearchLevelStats) -> bool {
    let mut did_lookup = false;
    for a in expected_syms.iter().copied() {
        for b in expected_syms.iter().copied() {
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
                    // println!("pruned candidate with ({}, {}) @ ({}, {})", a, b, t1, t2);
                    tracker.pruned();
                    return false;
                }
            }
        }
        // Why don't we check the cache right away?
        // Because, if the pruning rule fails before for (0, k) for some k,
        // we'll have done about fewer memory accesses than we would have for
        // the hash and equality testing needed to do the hash lookup
        // Therefore, we put this off a bit, to let the fast pruning pass go first
        //
        // This could probably be a tuneable parameter,
        // letting you control how aggressively you cache,
        // but I can't think of a good way to expose that.
        if !did_lookup {
            // TODO: disable cache because it's causing correctness issues
            // let probe = {cache.read().unwrap().get(current).copied()};
            // if let Some(v) = probe {
            //     tracker.cache_hit();
            //     return v;
            // }
            did_lookup = true;
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
                      caches: &'f [SearchResultCache<'d>]) -> bool {
    let tracker = &stats[current_level];

    if current_level == levels.len() {
        tracker.checking();
        let current = curr_states[0].unwrap();
        if current == target {
            tracker.success();
            println!("soln:{}", &current.name);
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
    // if current_level == 6 {
    //     println!("{}", curr_states[1].unwrap());
    // }
    // println!("[{} - {}] {}", current_level, level.ops.name, current);
    let cache = caches[current_level].clone();
    let proceed = |c: Option<&ProgState<'d>>| {
        tracker.checking();
        match c {
            Some(r) => {
                if level.prune {
                    if !viable(r, target,
                               level.matrix.as_ref().unwrap(),
                               &expected_syms[level.expected_syms],
                               cache.as_ref(), &tracker) {
                        return false;
                    }
                }
                let new_states = copy_replacing(&curr_states,
                                                lane, c);
                search(new_states, target, levels, expected_syms,
                       current_level + 1, stats, mode, caches)
            },
            None => {
                tracker.failed();
                false
            }
        }
    };

    let ops = &level.ops.ops; // Get at the actual vector of gathers
    let ret =
        match ops {
            OpSetKind::Gathers(gathers) => {
                if level.ops.fused_fold {
                    match mode {
                        Mode::All => {
                            gathers.iter().map(
                                |o| {
                                    let res = current.gather_fold_by(o);
                                    proceed(res.as_ref())
                                })
                                .fold(false, |acc, new| new || acc)
                        }
                        Mode::First => {
                            gathers.iter().any(
                                |o| {
                                    let res = current.gather_fold_by(o);
                                    proceed(res.as_ref())
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
                                    let res = Some(current.gather_by(o));
                                    proceed(res.as_ref())
                                })
                                .fold(false, |acc, new| new || acc)
                        }
                        Mode::First => {
                            gathers.iter().any(
                                |o| {
                                    let res = Some(current.gather_by(o));
                                    proceed(res.as_ref())
                                })
                        }
                    }
                }
            },
            OpSetKind::Merge(from, to) => {
                tracker.checking();
                let to = *to;
                let to_merge: SmallVec<[&ProgState; 6]> =
                    from.iter().copied().map(
                        |idx| curr_states[idx].unwrap()).collect();
                let next =
                    if level.ops.fused_fold {
                        ProgState::merge_folding(&to_merge)
                    } else {
                        Some(ProgState::merge(&to_merge))
                    };
                if next.is_some() {
                    let mut new_states = copy_replacing(&curr_states,
                                                        to, next.as_ref());
                    for i in from.iter().copied() {
                        if i != to {
                            new_states[i] = None
                        }
                    }

                    if level.prune {
                        if !viable(current, target,
                                   level.matrix.as_ref().unwrap(),
                                   &expected_syms[level.expected_syms],
                                   cache.as_ref(), &tracker) {
                            return false;
                        }
                    }

                    search(new_states, target, levels, expected_syms,
                           current_level + 1, stats, mode, caches)
                } else { tracker.failed(); false }
            }
            OpSetKind::Split(into, copies) => {
                tracker.checking();
                let mut new_states = curr_states.clone();
                let to_copy = curr_states[*into];
                for i in copies.iter().copied() {
                    new_states[i] = to_copy;
                }
                search(new_states, target, levels, expected_syms,
                       current_level + 1, stats, mode, caches)
            }
        };
    ret
}

pub fn synthesize(start: Vec<Option<ProgState>>, target: &ProgState,
                  levels: &[SynthesisLevel],
                  expected_syms: &[Vec<DomRef>],
                  mode: Mode) -> bool {

    let n_levels = levels.len();
    let stats = vec![SearchLevelStats::new(); n_levels + 1];
    let caches: Vec<SearchResultCache>
        = (0..n_levels).map(|_| Arc::new(RwLock::new(HashMap::new()))).collect();

    let states: States = SmallVec::from_iter(start.iter().map(|e| e.as_ref()));
    let start_time = Instant::now();
    let ret = search(states, target, levels, expected_syms, 0, &stats, mode, &caches);
    let dur = time_since(start_time);

    for (idx, stats) in (&stats).iter().enumerate() {
        println!("stats:{} ({}) {}", idx, levels.get(idx).map_or(&"[last]".into(), |x| &x.ops.name), stats);
    }
    println!("search:{} shape({:?}) {} mode({:?}) [{}]", target.name, target.state.shape(), ret, mode, dur);
    ret
}
