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

use std::time::Instant;

use itertools::iproduct;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    All,
    First,
}

// Invariant: any level with pruning enabled has a corresponding pruning matrix available
fn viable(current: &ProgState, target: &ProgState, matrix: &TransitionMatrix) -> bool {
    if target.domain_max != current.domain_max {
        println!("WARNING domain max differs {} -> {}", current.domain_max, target.domain_max);
        return false;
    }
    let dm = target.domain_max as usize;
    for a in 0..dm {
        for b in 0..dm {
            for (t1, t2) in iproduct!(target.inv_state[a].iter(), target.inv_state[b].iter()) {
                let result = iproduct!(current.inv_state[a].iter(), current.inv_state[b].iter())
                    .any(|(c1, c2)| {
                        let v = matrix.get(c1.slice(), c2.slice(), t1.slice(), t2.slice());
                        if !v {
                            println!("({:?}, {:?}) !-> ({:?}, {:?})", c1.slice(), c2.slice(), t1.slice(), t2.slice());
                        }
                        v
                    });
                println!("---");
                if !result {
                    return false;
                }
            }
        }
    }
    true
}

fn search(current: ProgState, target: &ProgState,
          levels: &[SynthesisLevel], current_level: usize,
          mode: Mode) -> bool {
    if &current == target {
        println!("soln:{}", &current.name);
        return true;
    }
    if current_level == levels.len() {
        return false;
    }

    let level = &levels[current_level];
    if level.prune && !viable(&current, target, level.matrix.as_ref().unwrap()) {
        println!("Pruned at {}", current_level);
        return false;
    }

    let ops = &level.ops.ops; // Get at the actual vector of gathers
    match mode {
        Mode::All => {
            ops.iter().map(|o| search(current.gather_by(o), target,
                                      levels, current_level + 1, mode))
                .any(|r| r)
        }
        Mode::First => {
            ops.iter().any(|o| search(current.gather_by(o), target,
                                      levels, current_level + 1, mode))
        }
    }
}

pub fn synthesize(start: ProgState, target: &ProgState,
                  levels: &[SynthesisLevel], mode: Mode) -> bool {
    let start_time = Instant::now();
    let ret = search(start, target, levels, 0, mode);
    let dur = time_since(start_time);
    println!("search:{} ({:?}) {} [{}]", target.name, target.state.shape(), ret, dur);
    ret
}
