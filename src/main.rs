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
mod operators;
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
        }

        errors {
            FileOpFailure(p: std::path::PathBuf) {
                description("couldn't open or create file")
                display("couldn't open or create file: {}", p.display())
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
            UnknownBasisType(basis: String) {
                description("unknown basis type")
                display("unknown basis type: {}", basis)
            }
            UnknownProblem(problem: String) {
                description("unknown problem")
                display("unknown problem: {}", problem)
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
    let specs: Vec<ProblemDesc> = match args.values_of_os("specs") {
        Some(iter) => {
            let files: Result<Vec<_>> = iter.map(open_file).collect();
            let result: Result<Vec<_>> =
                files?.iter().map(|f|
                                 serde_json::from_reader(BufReader::new(f))
                                 .map_err(|e| e.into())).collect();
            result?
        },
        None => vec![serde_json::from_reader(BufReader::new(std::io::stdin().lock()))?]
    };

    for desc in specs {
        let spec = desc.get_spec()?;
        let domain = desc.make_domain(spec.view());
        let (initial, target, mut levels) = desc.to_problem(&domain, spec)?;
        operators::add_matrices(matrix_dir, &mut levels)?;
        synthesize(initial, &target, &levels, synthesis_mode);
        operators::remove_matrices(&mut levels);
    }
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
    use crate::problem_desc::{trove};
    use crate::operators::swizzle::{fan,rotate,OpAxis};
    use crate::operators::load::load_rep;
    use crate::state::{ProgState, Domain, Value};

    fn fixed_solution_from_scalar_8x3<'d>(d: &'d Domain) -> ProgState<'d> {
        let initial = ProgState::linear(d, &[24]);
        let s0 = initial.gather_by(&load_rep(&[24], &[8, 3]).unwrap().ops[0]);
        let s1 = s0.gather_by(&fan(8, 3, OpAxis::Rows, 0, 2));
        let s2 = s1.gather_by(&rotate(8, 3, OpAxis::Rows, -7, 8, 0));
        let s3 = s2.gather_by(&fan(8, 3, OpAxis::Columns, 0, 3));
        let s4 = s3.gather_by(&rotate(8, 3, OpAxis::Columns, -5, 3, 0));
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
}
