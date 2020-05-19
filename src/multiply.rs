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
use crate::transition_matrix::{DenseTransitionMatrix,
                               TransitionMatrixOps, TransitionMatrix};
//use crate::misc::{COLLECT_STATS, loghist};

use itertools::iproduct;

// Note, because of how these matrices are used
// the lookup API takes (j, i) not (i, j)
// Also note that the matrix is really (m x m)x(k x k)
fn sparsify_mul_no_trans(a: &DenseTransitionMatrix, b: &DenseTransitionMatrix)
                         -> TransitionMatrix {
    let (k, m) = a.slots();
    let (n, k2) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseTransitionMatrix::empty(a.get_target_shape(),
                                             b.get_current_shape());

    let mut k_idxs = Vec::with_capacity(m.pow(2) / 8);

    for (i1, i2) in iproduct!(0..m, 0..m) {
        for (kidx1, kidx2) in iproduct!(0..k, 0..k) {
            if a.get_idxs(kidx1, kidx2, i1, i2) {
                k_idxs.push((kidx1, kidx2))
            }
        }
        for (j1, j2) in iproduct!(0..n, 0..n) {
            for (kidx1, kidx2) in k_idxs.iter().copied() {
                if b.get_idxs(j1, j2, kidx1, kidx2) {
                    c.set_idxs(j1, j2, i1, i2, true);
                    break;
                }
            }
        }
        k_idxs.clear();
    }

    println!("mul_stats:: transposed=false");
    c.into()
}

fn sparsify_mul_with_trans(a: &DenseTransitionMatrix, b: &DenseTransitionMatrix)
                           -> TransitionMatrix {
    let (k, m) = a.slots();
    let (n, k2) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseTransitionMatrix::empty(a.get_target_shape(),
                                             b.get_current_shape());

    let mut k_idxs = Vec::with_capacity(m.pow(2) / 8);
    for (j1, j2) in iproduct!(0..n, 0..n) {
        for (kidx1, kidx2) in iproduct!(0..k, 0..k) {
            if b.get_idxs(j1, j2, kidx1, kidx2) {
                k_idxs.push((kidx1, kidx2))
            }
        }

        for (i1, i2) in iproduct!(0..m, 0..m) {
            for (kidx1, kidx2) in k_idxs.iter().copied() {
                if a.get_idxs(kidx1, kidx2, i1, i2) {
                    c.set_idxs(j1, j2, i1, i2, true);
                    break;
                }
            }
        }
        k_idxs.clear();
    }

    println!("mul_stats:: transposed=true;");
    c.into()
}

pub fn sparsifying_mul(a: &TransitionMatrix, b: &TransitionMatrix)
                       -> TransitionMatrix {
    match (a, b) {
        (TransitionMatrix::Dense(a), TransitionMatrix::Dense(b)) => {
            if a.n_ones() <= b.n_ones() {
                sparsify_mul_no_trans(a, b)
            }
            else {
                sparsify_mul_with_trans(a, b)
            }
        }
    }
}
