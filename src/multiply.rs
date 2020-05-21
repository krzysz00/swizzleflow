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
                         -> DenseTransitionMatrix {
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
        for (kidx1, kidx2) in k_idxs.iter().copied() {
            c.update_row(i1, i2, b, kidx1, kidx2);
        }
        k_idxs.clear();
    }

    c
}

fn transpose(mat: &DenseTransitionMatrix) -> DenseTransitionMatrix {
    let mut ret = DenseTransitionMatrix::empty(mat.get_current_shape(),
                                               mat.get_target_shape());
    let (n, m) = mat.slots();
    for (i1, i2) in iproduct!(0..m, 0..m) {
        for (j1, j2) in iproduct!(0..n, 0..n) {
            ret.set_idxs(i1, i2, j1, j2, mat.get_idxs(j1, j2, i1, i2));
        }
    }
    ret
}

fn sparsify_mul_with_trans(a: &DenseTransitionMatrix, b: &DenseTransitionMatrix)
                           -> DenseTransitionMatrix {
    let b_tr = transpose(b);
    let a_tr = transpose(a);
    let c_tr = sparsify_mul_no_trans(&b_tr, &a_tr);
    let c = transpose(&c_tr);
    c
}

pub fn sparsifying_mul(a: &TransitionMatrix, b: &TransitionMatrix)
                       -> TransitionMatrix {
    match (a, b) {
        (TransitionMatrix::Dense(a), TransitionMatrix::Dense(b)) => {
            let ones_a = a.n_ones();
            let ones_b = b.n_ones();
            let density_a = (ones_a as f64) / (a.n_elements() as f64);
            let density_b = (ones_b as f64) / (b.n_elements() as f64);
            if ones_a <= ones_b {
                let c = sparsify_mul_no_trans(a, b);
                println!("mul_stats:: transposed=false; ones_a={}; ones_b={}; density_a={}; density_b={};",
                         ones_a, ones_b, density_a, density_b);
                c.into()
            }
            else {
                let c = sparsify_mul_with_trans(a, b);
                println!("mul_stats:: transposed=true; ones_a={}; ones_b={}; density_a={}; density_b={};",
                         ones_a, ones_b, density_a, density_b);
                c.into()
            }
        }
    }
}
