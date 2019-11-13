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
extern crate intel_mkl_src;

mod misc;
mod state;
mod transition_matrix;
mod matrix_load;
mod operators;
mod expected_syms_util;
mod problem_desc;
mod synthesis;

use clap::clap_app;
use std::path::Path;
use std::io::BufReader;

use crate::problem_desc::ProblemDesc;
use crate::synthesis::{Mode, synthesize};
use crate::misc::open_file;

pub mod errors {
    use error_chain::error_chain;
    error_chain! {
        foreign_links {
            Io(std::io::Error);
            Json(serde_json::Error);
            Ndarray(ndarray::ShapeError);
        }

        errors {
            FileOpFailure(p: std::path::PathBuf) {
                description("couldn't open or create file")
                display("couldn't open or create file: {}", p.display())
            }
            ParseError(p: std::path::PathBuf) {
                description("couldn't parse spec")
                display("couldn't parse spec: {}", p.display())
            }
            NotAMatrix {
                description("not a matrix")
                display("not a matrix")
            }
            UnknownMatrixType(t: u8) {
                description("unknown matrix type")
                display("unknown matrix type: {}", t)
            }
            InvalidShapeDim(shape: Vec<usize>, dims: usize) {
                description("invalid shape dimensions")
                display("invalid shape {:?} - should be {} dimensional", shape, dims)
            }
            ShapeMismatch(shape1: Vec<usize>, shape2: Vec<usize>) {
                description("shapes don't match")
                display("{:?} should match up with {:?}", shape1, shape2)
            }
            AxisLengthMismatch(ax1: usize, len1: usize, ax2: usize, len2: usize) {
                description("axis lengths don't match")
                display("axis lengths don't match: d{} (length {}) != d{} (length {})", ax1, len1, ax2, len2)
            }
            WrongMergeArgs(expected: usize, got: usize) {
                description("incorrect number of merge arguments")
                display("incorrect number of merge arguments: expected {}, got {}", expected, got)
            }
            InvalidArrayData(shape: Vec<usize>) {
                description("invalid array data"),
                display("invalid array data for shape: {:?}", shape)
            }
            MissingShape(lane: usize) {
                description("no known input shape in lane"),
                display("no known input shape in lane: {}", lane)
            }
            UnknownBasisType(basis: String) {
                description("unknown basis type")
                display("unknown basis type: {}", basis)
            }
            UnknownProblem(problem: String) {
                description("unknown problem")
                display("unknown problem: {}", problem)
            }
            SymbolsNotInSpec {
                description("symbols not in spec")
                display("symbols not in spec")
            }
            MissingOption(key: String) {
                description("missing options")
                display("missing options: {}", key)
            }
            NoSplitFolds {
                description("can't fold after splits")
                display("can't fold after splits")
            }
            NoSplitPrune {
                description("can't prune after splits")
                display("can't prune after splits")
            }
            FinalLane(lane: usize) {
                description("lane of last step must be 0")
                display("lane of last step must be 0 but is {}", lane)
            }
            MatrixLoad(p: std::path::PathBuf) {
                description("couldn't read matrix file")
                display("couldn't read matrix file: {}", p.display())
            }
            LevelBuild(step: Box<crate::problem_desc::SynthesisLevelDesc>) {
                description("couldn't create level")
                display("couldn't create level: {}", serde_json::to_string(&step)
                        .unwrap_or_else(|_| "couldn't print".to_owned()))
            }
            BadSpec(spec: crate::problem_desc::ProblemDesc) {
                description("bad specification")
                display("bad specification: {:#?}", spec)
            }
        }
    }
}

use errors::*;

const DEFAULT_MATRIX_DIR: &str = "matrices/";

fn run() -> Result<()> {
    let args =
        clap_app!(swizzleflow =>
                  (version: "0.1")
                  (author: "Krzysztof Drewniak <krzysd@cs.washington.edu> et al.")
                  (about: "Tool for synthesizing high-performance kernels from dataflow graph sketches")
                  (@arg matrix_dir: -m --("matrix-dir") +takes_value value_name("MATRIX_DIR")
                   default_value_os(DEFAULT_MATRIX_DIR.as_ref())
                   "Directory to store matrices in")
                  (@arg all: -a --all "Find all solutions")
                  (@arg specs: ... value_name("SPEC") "Specification files (stdin if none specified)")
        ).setting(clap::AppSettings::TrailingVarArg).get_matches();

    let synthesis_mode = if args.is_present("all") { Mode::All } else { Mode::First };

    let matrix_dir = Path::new(args.value_of_os("matrix_dir").unwrap()); // We have a default
    let specs: Vec<(ProblemDesc, String)> = match args.values_of_os("specs") {
        Some(iter) => {
            iter.map(
                |path| {
                    let name = path.to_string_lossy().into_owned();
                    let file = open_file(path)?;
                    serde_json::from_reader(BufReader::new(file))
                        .chain_err(|| ErrorKind::ParseError(path.into()))
                        .map(|v| (v, name))
                }).collect::<Result<Vec<_>>>()?
        },
        None => vec![(serde_json::from_reader(BufReader::new(std::io::stdin().lock()))?,
                      "stdin".into())]
    };

    for (desc, name) in specs {
        println!("spec:{}", name);
        let spec = desc.get_spec().chain_err(|| ErrorKind::BadSpec(desc.clone()))?;
        let domain = desc.make_domain(spec.view());
        let mut levels = desc.get_levels()
            .chain_err(|| ErrorKind::BadSpec(desc.clone()))?;
        let (initial, target, expected_syms) =
            desc.build_problem(&domain, &levels, spec)
            .chain_err(|| ErrorKind::BadSpec(desc.clone()))?;
        let max_lanes = initial.len();
        matrix_load::add_matrices(matrix_dir, &mut levels, max_lanes)?;
        println!("Begin search");
        synthesize(initial, &target, &levels, &expected_syms, synthesis_mode);
        matrix_load::remove_matrices(&mut levels);
    }
    // The goal: (t, i) -> (t, i) merged with (t, i) -> (t, t - i)
    // Except that that second one isn't a swizzle inventor xform
    // We can fix it by adding k back though, at the cost of equivalence issues
    // Then, merge these, transpose
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        use std::io::Write;
        use error_chain::ChainedError; // trait which holds `display_chain`
        let stderr = &mut std::io::stderr();

        writeln!(stderr, "{}", e.display_chain()).expect("Failed to write error message");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use crate::problem_desc::{trove, poly_mult};
    use crate::operators::swizzle::{xform,rotate};
    use crate::operators::load::{load_rep,broadcast};
    use crate::operators::{identity_gather, transpose_gather};
    use crate::state::{ProgState, Domain, Value};

    fn fixed_solution_from_scalar_8x3<'d>(d: &'d Domain) -> ProgState<'d> {
        let initial = ProgState::linear(d, 1, &[24]);
        let shape = [8, 3];
        let s0 = initial.gather_by(&load_rep(&[24], &[8, 3]).unwrap()
                                   .gathers().unwrap()[0]);
        let s1 = s0.gather_by(&xform(&shape, 1, 0, 1, 2, 1, 8, None, false));
        let s2 = s1.gather_by(&rotate(&shape, 1, 0, 1, 0, None));
        let s3 = s2.gather_by(&xform(&shape, 0, 1, 0, 3, 1, 3, None, false));
        let s4 = s3.gather_by(&rotate(&shape, 0, 1, 0, 0, None));
        s4
    }

    #[test]
    fn trove_solution_works() {
        let symbols: ndarray::Array1<Value> = (0u16..24u16).map(Value::Symbol).collect();
        let symbols = symbols.into_dyn();
        let domain = Domain::new(symbols.view());
        let spec = ProgState::new_from_spec(&domain, trove(8, 3), "trove").unwrap();
        let solution = fixed_solution_from_scalar_8x3(&domain);
        println!("spec {}\n solution {}", spec, solution);
        assert_eq!(spec, solution);
    }

    #[test]
    fn poly_mult_works() {
        use ndarray::ArrayD;
        use itertools::iproduct;
        use crate::operators::select::{cond_keep_gather,Op};
        let f1 = xform(&[4, 4], 1, 0, 1, 1, 0, 4, None, false);
        let r1 = rotate(&[4, 4], 1, 0, 1, 0, None);
        let f2 = xform(&[4, 4], 0, 1, 1, 1, -1, 4, None, false);
        let r2 = rotate(&[4, 4], 1, 0, 1, 0, None);
        let broadcast = broadcast(&[4, 4], &[4, 2, 4], 1).unwrap().gathers().unwrap()[0].clone();
        let spec = poly_mult(4);
        let domain = Domain::new(spec.view());
        let arr1 = iproduct!(0..4, 0..4).map(|(_i, j)| crate::state::Value::Symbol(j)).collect();
        let arr1 = ArrayD::from_shape_vec(vec![4, 4], arr1).unwrap();
        let state = ProgState::new_from_spec(&domain, arr1, "init").unwrap();
        let s1 = state.gather_by(&f1);
        let s2 = s1.gather_by(&r1);

        let arr2 = iproduct!(0..4, 0..4).map(|(_i, j)| crate::state::Value::Symbol(4 + j)).collect();
        let arr2 = ArrayD::from_shape_vec(vec![4, 4], arr2).unwrap();
        let state2 = ProgState::new_from_spec(&domain, arr2, "init").unwrap();
        let s3 = state2.gather_by(&f2);
        let s4 = s3.gather_by(&r2);

        let m = ProgState::merge_folding(&[&s2, &s4]).unwrap();
        let b = m.gather_by(&broadcast);

        let mut retain = std::collections::BTreeMap::new();
        retain.insert(1, 0);
        let c1 = cond_keep_gather(&[4, 2, 4], 2, 0, 0,
                                  Op::Leq, &retain);
        let d1 = b.gather_by(&c1);

        retain.insert(1, 1);
        let c2 = cond_keep_gather(&[4, 2, 4], 2, 0, 0,
                                  Op::Gt, &retain);
        let d2 = d1.gather_fold_by(&c2).unwrap();

        let tr = transpose_gather(&[2, 4], &[4, 2]);
        let transposed = d2.gather_by(&tr);
        let reshape = identity_gather(&[8]);
        let result = transposed.gather_by(&reshape);
        let spec_state = ProgState::new_from_spec(&domain, spec, "spec").unwrap();
        assert_eq!(result, spec_state);
    }
}
