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
pub mod misc;
mod state;
pub mod matrix;
mod transition_matrix;
pub mod multiply;
pub mod matrix_load;
mod operators;
mod expected_syms_util;
pub mod problem_desc;
pub mod synthesis;

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
            WrongStackArgs(expected: usize, got: usize) {
                description("incorrect number of stack arguments")
                display("incorrect number of stack arguments: expected {}, got {}", expected, got)
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
            BadOptionLength(key: String, expected: usize) {
                description("bad option length")
                display("bad option length for {}, expected {}", key, expected)
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
            ZeroFoldDimension {
                description("cannot fold on length 0 dimensions")
                display("cannot fold on length 0 dimensions")
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
            InvalidCmdArg(name: &'static str, reqs: &'static str) {
                description("argument is not valid")
                display("value for {} is not valid (must be {})", name, reqs)
            }
        }
    }
}
