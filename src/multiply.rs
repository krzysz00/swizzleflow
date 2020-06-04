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
                               RowSparseTransitionMatrix,
                               TransitionMatrixOps, TransitionMatrix};
use crate::misc::{COLLECT_STATS};

use itertools::iproduct;

fn sparse_dense_mul(a: &RowSparseTransitionMatrix, b: &DenseTransitionMatrix)
                    -> DenseTransitionMatrix {
    let (m, k) = a.slots();
    let (k2, n) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseTransitionMatrix::empty(a.get_current_shape(),
                                             b.get_target_shape());

    for (i1, i2) in iproduct!(0..m, 0..m) {
        for (kidx1, kidx2) in a.view_row_set_elems(i1, i2).iter().copied() {
            c.update_row(i1, i2, b, kidx1, kidx2);
        }
    }
    c
}

fn sparse_sparse_mul(a: &RowSparseTransitionMatrix, b: &RowSparseTransitionMatrix)
                     -> DenseTransitionMatrix {
    let (m, k) = a.slots();
    let (k2, n) = b.slots();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseTransitionMatrix::empty(a.get_current_shape(),
                                             b.get_target_shape());

    for (i1, i2) in iproduct!(0..m, 0..m) {
        for (kidx1, kidx2) in a.view_row_set_elems(i1, i2).iter().copied() {
            for (j1, j2) in b.view_row_set_elems(kidx1, kidx2).iter().copied() {
                c.set_idxs(i1, i2, j1, j2, true);
            }
        }
    }
    c
}


pub fn transition_mul(a: &TransitionMatrix, b: &TransitionMatrix)
                      -> TransitionMatrix {
    if COLLECT_STATS {
        let ones_a = a.n_ones();
        let ones_b = b.n_ones();
        let density_a = (ones_a as f64) / (a.n_elements() as f64);
        let density_b = (ones_b as f64) / (b.n_elements() as f64);
        println!("mul_stats:: ones_a={}; ones_b={}; density_a={}; density_b={};",
                 ones_a, ones_b, density_a, density_b);
    }
    match (a, b) {
        (TransitionMatrix::RowSparse(a), TransitionMatrix::Dense(b)) => {
            println!("mul_stats:: b_sparse=false");
            let c = sparse_dense_mul(a, b);
            c.into()
        },
        (TransitionMatrix::RowSparse(a), TransitionMatrix::RowSparse(b)) => {
            println!("mul_stats:: b_sparse=true");
            let c = sparse_sparse_mul(a, b);
            c.into()
        },
        _ => panic!("Unsupported combination of types in multiply"),
    }
}
