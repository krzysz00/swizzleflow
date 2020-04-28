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
use std::collections::BTreeMap;

use crate::transition_matrix::{TransitionMatrixOps, TransitionMatrix};
use crate::misc::{COLLECT_STATS, loghist};

use itertools::Itertools;
//use itertools::iproduct;

// Note, because of how these matrices are used
// the lookup API takes (j, i) not (i, j)
// Also note that the matrix is really (m x m)x(k x k)
fn sparsify_mul_no_trans(a: &TransitionMatrix, b: &TransitionMatrix,
                         c: &mut TransitionMatrix) {
    let (k, m) = a.slots();
    let (n, k2) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut probes_success = BTreeMap::new();
    let mut probes_failure = BTreeMap::new();
    let mut k_idxs = Vec::with_capacity(m.pow(2) / 8);

    for i1 in 0..m {
        for kidx1 in 0..k {
            if a.get_idxs(kidx1, i1) {
                k_idxs.push(kidx1)
            }
        }
        for j1 in 0..n {
            let mut p = 0;
            for kidx1 in k_idxs.iter().copied() {
                if b.get_idxs(j1, kidx1) {
                    c.set_idxs(j1, i1, true);
                    if COLLECT_STATS {
                        *probes_success.entry(loghist(p + 1)).or_insert(0) += 1;
                    }
                    break;
                }
                p += 1;
            }
            if COLLECT_STATS && p == k_idxs.len() {
                *probes_failure.entry(loghist(p)).or_insert(0) += 1;
            }
        }
        k_idxs.clear();
    }

    if COLLECT_STATS {
        println!("mul_stats:: probes_success=[{:?}]; probes_failure=[{:?}];",
                 probes_success.iter().format(", "),
                 probes_failure.iter().format(", "));
    }
}

fn sparsify_mul_with_trans(a: &TransitionMatrix, b: &TransitionMatrix,
                           c: &mut TransitionMatrix) {
    let (k, m) = a.slots();
    let (n, k2) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut probes_success = BTreeMap::new();
    let mut probes_failure = BTreeMap::new();
    let mut k_idxs = Vec::with_capacity(m.pow(2) / 8);
    for j1 in 0..n {
        for kidx1 in 0..k {
            if b.get_idxs(j1, kidx1) {
                k_idxs.push(kidx1)
            }
        }

        for i1 in 0..m {
            let mut p = 0;
            for kidx1 in k_idxs.iter().copied() {
                if a.get_idxs(kidx1, i1) {
                    c.set_idxs(j1, i1, true);
                    if COLLECT_STATS {
                        *probes_success.entry(p + 1).or_insert(0) += 1;
                    }
                    break;
                }
                p += 1;
            }
            if COLLECT_STATS && p == k_idxs.len() {
                *probes_failure.entry(p).or_insert(0) += 1;
            }
        }
        k_idxs.clear();
    }

    if COLLECT_STATS {
        println!("mul_stats:: probes_success=[{:?}]; probes_failure=[{:?}];",
                 probes_success.iter().format(", "),
                 probes_failure.iter().format(", "));
    }
}

pub fn sparsifying_mul(a: &TransitionMatrix, b: &TransitionMatrix,
                       c: &mut TransitionMatrix) {
    if a.n_ones() <= b.n_ones() {
        sparsify_mul_no_trans(a, b, c);
    }
    else {
        sparsify_mul_with_trans(a, b, c);
    }
}
