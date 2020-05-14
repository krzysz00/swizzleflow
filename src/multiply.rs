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
use std::cmp::min;
use std::collections::BTreeMap;

use crate::transition_matrix::{TransitionMatrixOps, TransitionMatrix};
use crate::misc::{COLLECT_STATS, loghist};

use itertools::Itertools;
use itertools::iproduct;

use parking_lot::Mutex;
use rayon::prelude::*;


const TILE_SIZE: usize = 64;
// Note, because of how these matrices are used
// the lookup API takes (j, i) not (i, j)
// Also note that the matrix is really (m x m)x(k x k)
fn sparsify_mul_no_trans(a: &TransitionMatrix, b: &TransitionMatrix)
                         -> TransitionMatrix {
    let (k, m) = a.slots();
    let (n, k2) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }
    let m_squared = m.pow(2);

    let c = Mutex::new(TransitionMatrix::empty(a.get_target_shape(),
                                               b.get_current_shape()));
    let probes_success = Mutex::new(BTreeMap::new());
    let probes_failure = Mutex::new(BTreeMap::new());

    (0..m_squared/TILE_SIZE).into_par_iter()
    .for_each_init(
    || Vec::with_capacity(m_squared / 8),
    |k_idxs, i_tile| {
        for i_combined in i_tile*TILE_SIZE..min((i_tile + 1)*TILE_SIZE,m_squared) {
        let (i1, i2) = (i_combined / m, i_combined % m);
        for (kidx1, kidx2) in iproduct!(0..k, 0..k) {
            if a.get_idxs(kidx1, kidx2, i1, i2) {
                k_idxs.push((kidx1, kidx2))
            }
        }
        for (j1, j2) in iproduct!(0..n, 0..n) {
            let mut p = 0;
            for (kidx1, kidx2) in k_idxs.iter().copied() {
                if b.get_idxs(j1, j2, kidx1, kidx2) {
                    let mut c = c.lock();
                    c.set_idxs(j1, j2, i1, i2, true);
                    if COLLECT_STATS {
                        let mut probes_success = probes_success.lock();
                        *probes_success.entry(loghist(p + 1)).or_insert(0) += 1;
                    }
                    break;
                }
                p += 1;
            }
            if COLLECT_STATS && p == k_idxs.len() {
                let mut probes_failure = probes_failure.lock();
                *probes_failure.entry(loghist(p)).or_insert(0) += 1;
            }
        }
        k_idxs.clear();
    }
    });

    if COLLECT_STATS {
        println!("mul_stats:: probes_success=[{:?}]; probes_failure=[{:?}];",
                 probes_success.into_inner().iter().format(", "),
                 probes_failure.into_inner().iter().format(", "));
    }
    c.into_inner()
}

fn sparsify_mul_with_trans(a: &TransitionMatrix, b: &TransitionMatrix)
                           -> TransitionMatrix {
    let (k, m) = a.slots();
    let (n, k2) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let n_squared = n.pow(2);

    let c = Mutex::new(TransitionMatrix::empty(a.get_target_shape(),
                                               b.get_current_shape()));
    let probes_success = Mutex::new(BTreeMap::new());
    let probes_failure = Mutex::new(BTreeMap::new());

    (0..n_squared/TILE_SIZE).into_par_iter()
    .for_each_init(
    || Vec::with_capacity(n_squared / 8),
    |k_idxs, j_tile| {
    for j_combined in j_tile*TILE_SIZE..min((j_tile+1)*TILE_SIZE, n_squared) {
        let (j1, j2) = (j_combined / n, j_combined % n);
        for (kidx1, kidx2) in iproduct!(0..k, 0..k) {
            if b.get_idxs(j1, j2, kidx1, kidx2) {
                k_idxs.push((kidx1, kidx2))
            }
        }

        for (i1, i2) in iproduct!(0..m, 0..m) {
            let mut p = 0;
            for (kidx1, kidx2) in k_idxs.iter().copied() {
                if a.get_idxs(kidx1, kidx2, i1, i2) {
                    let mut c = c.lock();
                    c.set_idxs(j1, j2, i1, i2, true);
                    if COLLECT_STATS {
                        let mut probes_success = probes_success.lock();
                        *probes_success.entry(p + 1).or_insert(0) += 1;
                    }
                    break;
                }
                p += 1;
            }
            if COLLECT_STATS && p == k_idxs.len() {
                let mut probes_failure = probes_failure.lock();
                *probes_failure.entry(p).or_insert(0) += 1;
            }
        }
        k_idxs.clear();
    }
    });

    if COLLECT_STATS {
        println!("mul_stats:: probes_success=[{:?}]; probes_failure=[{:?}];",
                 probes_success.into_inner().iter().format(", "),
                 probes_failure.into_inner().iter().format(", "));
    }
    c.into_inner()
}

pub fn sparsifying_mul(a: &TransitionMatrix, b: &TransitionMatrix)
                       -> TransitionMatrix {
    if a.n_ones() <= b.n_ones() {
        sparsify_mul_no_trans(a, b)
    }
    else {
        sparsify_mul_with_trans(a, b)
    }
}
