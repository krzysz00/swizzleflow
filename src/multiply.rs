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
use crate::misc::{COLLECT_STATS};

use itertools::iproduct;

// Note, because of how these matrices are used
// the lookup API takes (j, i) not (i, j)
// Also note that the matrix is really (m x m)x(k x k)
fn sparsify_mul_no_trans(a: &DenseTransitionMatrix, b: &DenseTransitionMatrix)
                         -> DenseTransitionMatrix {
    let (m, k) = a.slots();
    let (k2, n) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseTransitionMatrix::empty(a.get_current_shape(),
                                             b.get_target_shape());

    let mut k_idxs = Vec::with_capacity(m.pow(2) / 8);

    for (i1, i2) in iproduct!(0..m, 0..m) {
        for (kidx1, kidx2) in iproduct!(0..k, 0..k) {
            if a.get_idxs(i1, i2, kidx1, kidx2) {
                k_idxs.push((kidx1, kidx2))
            }
        }
        for (kidx1, kidx2) in k_idxs.iter().copied() {
            c.update_row(i1, i2, b, kidx1, kidx2);
        }
        k_idxs.clear();
    }
    c
}

pub fn sparsifying_mul(a: &TransitionMatrix, b: &TransitionMatrix)
                       -> TransitionMatrix {
    match (a, b) {
        (TransitionMatrix::Dense(a), TransitionMatrix::Dense(b)) => {
            if COLLECT_STATS {
                let ones_a = a.n_ones();
                let ones_b = b.n_ones();
                let density_a = (ones_a as f64) / (a.n_elements() as f64);
                let density_b = (ones_b as f64) / (b.n_elements() as f64);
                println!("mul_stats:: ones_a={}; ones_b={}; density_a={}; density_b={};",
                         ones_a, ones_b, density_a, density_b);
            }
            let c = sparsify_mul_no_trans(a, b);
            c.into()
        }
    }
}
