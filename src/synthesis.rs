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
use crate::state::ProgState;
use crate::transition_matrix::{TransitionMatrix,TransitionMatrixOps};
use crate::operators::SynthesisLevel;

use crate::misc::{time_since};

use ndarray::Dimension;

use std::collections::HashMap;

use std::time::Instant;

use std::fmt;
use std::fmt::{Display,Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use itertools::iproduct;


#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    All,
    First,
}

#[derive(Debug, Default)]
struct SearchLevelStats {
    tested: AtomicUsize,
    pruned: AtomicUsize,
    succeeded: AtomicUsize,
    cache_hit: AtomicUsize,
    cache_set: AtomicUsize,
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

    pub fn cache_hit(&self) {
        self.cache_hit.fetch_add(1, Ordering::Relaxed);
    }

    pub fn cache_set(&self) {
        self.cache_set.fetch_add(1, Ordering::Relaxed);
    }
}

impl Clone for SearchLevelStats {
    fn clone(&self) -> Self {
        Self { tested: AtomicUsize::new(self.tested.load(Ordering::SeqCst)),
               pruned: AtomicUsize::new(self.pruned.load(Ordering::SeqCst)),
               succeeded: AtomicUsize::new(self.succeeded.load(Ordering::SeqCst)),
               cache_hit: AtomicUsize::new(self.cache_hit.load(Ordering::SeqCst)),
               cache_set: AtomicUsize::new(self.cache_set.load(Ordering::SeqCst)),
        }
    }
}

impl Display for SearchLevelStats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Let's force a memory fence here to be safe
        let tested = self.tested.load(Ordering::SeqCst);
        let found = self.succeeded.load(Ordering::Relaxed);
        let pruned = self.pruned.load(Ordering::Relaxed);

        let continued = tested - found - pruned;

        let cache_hits = self.cache_hit.load(Ordering::SeqCst);
        let cache_puts = self.cache_set.load(Ordering::SeqCst);

        write!(f, "tested({}), found({}), pruned({}), continued({}) cache({}:{})",
               tested, found, pruned, continued, cache_hits, cache_puts)
    }
}

type ResultMap<'d> = RwLock<HashMap<ProgState<'d>, bool>>;
type SearchResultCache<'d> = Arc<ResultMap<'d>>;

// Invariant: any level with pruning enabled has a corresponding pruning matrix available
fn viable<'d>(current: &ProgState<'d>, target: &ProgState<'d>, matrix: &TransitionMatrix,
              cache: &ResultMap<'d>, tracker: &SearchLevelStats) -> bool {
    let level_min = target.domain.level_bounds[current.level];
    let level_max = target.domain.level_bounds[current.level + 1];

    let mut did_lookup = false;
    for a in level_min..level_max {
        for b in level_min..level_max {
            for (t1, t2) in iproduct!(target.inv_state[a].iter(), target.inv_state[b].iter()) {
                let result = iproduct!(current.inv_state[a].iter(), current.inv_state[b].iter())
                    .any(|(c1, c2)| {
                        let v = matrix.get(c1.slice(), c2.slice(), t1.slice(), t2.slice());
                        v
                    });
                if !result {
                    if did_lookup {
                        cache.write().unwrap().insert(current.clone(), false);
                        tracker.cache_set();
                    }
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
        if a == 0 {
            let probe = {cache.read().unwrap().get(current).copied()};
            if let Some(v) = probe {
                tracker.cache_hit();
                return v;
            }
            did_lookup = true;
        }
    }
    true
}

fn search<'d, 'f>(current: ProgState<'d>, target: &ProgState<'d>,
                  levels: &'f [SynthesisLevel], current_level: usize,
                  stats: &'f [SearchLevelStats], mode: Mode,
                  caches: &'f [SearchResultCache<'d>]) -> bool {
    let tracker = &stats[current_level];
    tracker.checking();

    if &current == target {
        tracker.success();
        println!("soln:{}", &current.name);
        return true;
    }
    if current_level == levels.len() {
        return false;
    }

    let level = &levels[current_level];
    let cache = caches[current_level].clone();
    if level.prune {
        if !viable(&current, target,
                   level.matrix.as_ref().unwrap(),
                   cache.as_ref(), &tracker) {
//        println!("{}", current);
            tracker.pruned();
            return false;
        }
    }
    let ops = &level.ops.ops; // Get at the actual vector of gathers
    let ret =
        if level.ops.fused_fold {
            match mode {
                Mode::All => {
                    ops.iter().map(|o|
                                   if let Some(res) = current.gather_fold_by(o) {
                                       search(res, target, levels, current_level + 1,
                                              stats, mode, caches)
                                   } else { false })
                        .fold(false, |acc, new| new || acc)
                }
                Mode::First => {
                    ops.iter().any(|o|
                                   if let Some(res) = current.gather_fold_by(o) {
                                       search(res, target, levels, current_level + 1,
                                              stats, mode, caches)
                                   } else { false })
                }
            }
        }
        else {
            match mode {
                Mode::All => {
                    // Yep, this is meant not to be short-circuiting
                    ops.iter().map(|o| search(current.gather_by(o), target,
                                              levels, current_level + 1, stats, mode, caches))
                        .fold(false, |acc, new| new || acc)
                }
                Mode::First => {
                    ops.iter().any(|o| search(current.gather_by(o), target,
                                              levels, current_level + 1, stats, mode, caches))
                }
            }
        };
    { cache.write().unwrap().insert(current.clone(), ret); }
    tracker.cache_set();
    ret
}

pub fn synthesize(start: ProgState, target: &ProgState,
                  levels: &[SynthesisLevel], mode: Mode) -> bool {

    let n_levels = levels.len();
    let stats = vec![SearchLevelStats::new(); n_levels + 1];
    let caches: Vec<SearchResultCache>
        = (0..n_levels).map(|_| Arc::new(RwLock::new(HashMap::new()))).collect();

    let start_time = Instant::now();
    let ret = search(start, target, levels, 0, &stats, mode, &caches);
    let dur = time_since(start_time);

    for (idx, stats) in (&stats).iter().enumerate() {
        println!("stats:{} {}", idx, stats);
    }
    println!("search:{} shape({:?}) {} mode({:?}) [{}]", target.name, target.state.shape(), ret, mode, dur);
    ret
}
