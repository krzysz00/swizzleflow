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
mod misc;
mod state;
mod transition_matrix;
mod operators;
mod problem_desc;

pub mod errors {
    use error_chain::error_chain;
    error_chain! {
        foreign_links {
            Io(std::io::Error);
            Json(serde_json::Error);
        }

        errors {
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
        }
    }
}

use errors::*;

fn main() -> Result<()> {
    use problem_desc::*;
    let desc = ProblemDesc {
            start_name: "linear".to_owned(),
            end_name: "trove".to_owned(),
            steps: vec![
                SynthesisLevelDesc { basis: "sRr".to_owned(),
                                     in_sizes: vec![3, 4], out_sizes: vec![3, 4],
                                     prune: false},
            ]
        };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &desc)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::problem_desc::trove;
    use crate::operators::swizzle::{fan,rotate,OpAxis};
    use crate::state::ProgState;

    fn fixed_solution_from_scala_3x8() -> ProgState {
        let initial = ProgState::linear(&[3, 8]);
        let s1 = initial.gather_by(&fan(3, 8, OpAxis::Columns, 0, 2));
        let s2 = s1.gather_by(&rotate(3, 8, OpAxis::Columns, -7, 8, 0));
        let s3 = s2.gather_by(&fan(3, 8, OpAxis::Rows, 0, 3));
        let s4 = s3.gather_by(&rotate(3, 8, OpAxis::Rows, -5, 3, 0));
        s4
    }

    #[test]
    fn trove_solution_works() {
        let spec = trove(3, 8);
        let solution = fixed_solution_from_scala_3x8();
        println!("spec {}\n solution {}", spec, solution);
        assert_eq!(spec, solution);
    }
}
