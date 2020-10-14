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
use crate::matrix::{DenseMatrix,
                    RowSparseMatrix,
                    MatrixOps, Matrix};
use crate::transition_matrix::TransitionMatrix;
use crate::misc::{COLLECT_STATS};

fn sparse_dense_mul(a: &RowSparseMatrix, b: &DenseMatrix)
                    -> DenseMatrix {
    let (m, k) = a.dims();
    let (k2, n) = b.dims();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseMatrix::empty(m, n);

    for i in 0..m {
        for kidx in a.view_row_set_elems(i).iter().copied() {
            c.update_row(i, b, kidx);
        }
    }
    c
}

fn sparse_sparse_mul(a: &RowSparseMatrix, b: &RowSparseMatrix)
                     -> DenseMatrix {
    let (m, k) = a.dims();
    let (k2, n) = b.dims();
    if k != k2 {
        panic!("Invalid shapes for multiply ({}, {}), ({}, {})",
        m, k, k2, n);
    }

    let mut c = DenseMatrix::empty(m, n);

    for i in 0..m {
        for kidx in a.view_row_set_elems(i).iter().copied() {
            for j in b.view_row_set_elems(kidx).iter().copied() {
                c.set(i, j, true);
            }
        }
    }
    c
}


pub fn bool_mul(a: &Matrix, b: &Matrix)
                -> Matrix {
    if COLLECT_STATS {
        let ones_a = a.n_ones();
        let ones_b = b.n_ones();
        let density_a = (ones_a as f64) / (a.n_elements() as f64);
        let density_b = (ones_b as f64) / (b.n_elements() as f64);
        println!("mul_stats:: ones_a={}; ones_b={}; density_a={}; density_b={};",
                 ones_a, ones_b, density_a, density_b);
    }
    match (a, b) {
        (Matrix::RowSparse(a), Matrix::Dense(b)) => {
            println!("mul_stats:: b_sparse=false;");
            let c = sparse_dense_mul(a, b);
            c.into()
        },
        (Matrix::RowSparse(a), Matrix::RowSparse(b)) => {
            println!("mul_stats:: b_sparse=true;");
            let c = sparse_sparse_mul(a, b);
            c.into()
        },
        (Matrix::Dense(_), Matrix::Dense(_)) => panic!("Dense/dense multiply"),
        (Matrix::Dense(_), Matrix::RowSparse(_)) => panic!("Dense/row-sparse multiply"),
    }
}

pub fn transition_mul(a: &TransitionMatrix, b: &TransitionMatrix)
                      -> TransitionMatrix {
    let ret = bool_mul(&a.mat, &b.mat);
    TransitionMatrix::new_no_option(a.get_current_shapes(), b.get_target_shapes(), ret)
}

fn dense_dense_add(a: &mut DenseMatrix, b: &DenseMatrix) {
    let (m, n1) = a.dims();
    let (m2, n2) = b.dims();
    if m != m2 || n1 != n2 {
        panic!("Can't add {}x{} to {}x{} matrix", m, n1, m2, n2);
    }
    for i in 0..m {
        a.update_row(i, b, i);
    }
}

pub fn bool_add(a: &mut Matrix, b: &Matrix) {
    if COLLECT_STATS {
        let ones_a = a.n_ones();
        let ones_b = b.n_ones();
        let density_a = (ones_a as f64) / (a.n_elements() as f64);
        let density_b = (ones_b as f64) / (b.n_elements() as f64);
        println!("mul_stats:: ones_a={}; ones_b={}; density_a={}; density_b={};",
                 ones_a, ones_b, density_a, density_b);
    }
    match (a, b) {
        (Matrix::Dense(a), Matrix::Dense(b)) => {
            let c = dense_dense_add(a, b);
            c.into()
        },
        _ => panic!("Unsupported combination of types in add"),
    }
}


pub fn transition_add(a: &mut TransitionMatrix, b: &TransitionMatrix) {
    bool_add(&mut a.mat, &b.mat);
}
