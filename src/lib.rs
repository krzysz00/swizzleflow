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
#![recursion_limit="256"]
pub mod misc;
pub mod lexer;
mod builtins;
pub mod parser;
pub mod program_transforms;
pub mod state;
pub mod matrix;
mod transition_matrix;
pub mod multiply;
pub mod abstractions;
mod operators;
pub mod synthesis;

pub mod errors {
    use error_chain::error_chain;
    error_chain! {
        foreign_links {
            Io(std::io::Error);
            Ndarray(ndarray::ShapeError);
        }

        errors {
            UnclonedString(line: usize, col: usize) {
                description("Unclosed string")
                display("Unclosed string starting at line {}, column {}", line, col)
            }
            FileOpFailure(p: std::path::PathBuf) {
                description("couldn't open or create file")
                display("couldn't open or create file: {}", p.display())
            }
            BadArrayLen(len: usize, expected: usize, token: crate::lexer::Token) {
                description("wrong number of elements in array")
                display("Starting at line {}, column {}, expected {} elements in array, got {}",
                        token.line, token.col, len, expected)
            }
            InvalidRange(lower: i64, upper: i64, token: crate::lexer::Token) {
                description("invalid range literal"),
                display("At line {}, column {}: invalid range literal range({}, {})",
                        token.line, token.col, lower, upper)
            }
            ParseError(got: crate::lexer::Token, expected: &'static str) {
                description("unexpected token")
                display("expected {} but got {:?} at line {}, column {}",
                        expected, got.t, got.line, got.col)
            }
            FileParseError(p: String) {
                description("couldn't parse spec")
                display("couldn't parse spec: {}", p)
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
            MissingOption(key: String) {
                description("missing options")
                display("missing options: {}", key)
            }
            BadOptionLength(key: String, expected: usize) {
                description("bad option length")
                display("bad option length for {}, expected {}", key, expected)
            }
            WrongArity(got: usize, expected: usize) {
                description("wrong number of arguments for bulitin function"),
                display("wrong number of arguments for bulitin function: got {}, expceted {}", got, expected)
            }
            ZeroFoldDimension {
                description("cannot fold on length 0 dimensions")
                display("cannot fold on length 0 dimensions")
            }
            BadGoalShape(program: Vec<Option<crate::misc::ShapeVec>>,
                         goal: Vec<Option<crate::misc::ShapeVec>>) {
                description("goal doesn't have appropriate shape"),
                display("goal doesn't have appropriate shape: program computes {:?}, goal has {:?}",
                        program, goal)
            }
            MatrixLoad(p: std::path::PathBuf) {
                description("couldn't read matrix file")
                display("couldn't read matrix file: {}", p.display())
            }
            LevelBuild(step: ()) {
                description("couldn't create level")
                display("couldn't create level")
            }
            BadSpec(spec: ()) {
                description("bad specification")
                display("bad specification")
            }
            InvalidCmdArg(name: &'static str, reqs: &'static str) {
                description("argument is not valid")
                display("value for {} is not valid (must be {})", name, reqs)
            }
        }
    }
}
