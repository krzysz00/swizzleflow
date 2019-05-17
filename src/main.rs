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
mod operators;
mod swizzle_ops;
mod transition_matrix;

use ndarray::{Array,Ix};
use state::{ProgState,Symbolic};

fn trove(m: Ix, n: Ix) -> ProgState {
    let array = Array::from_shape_fn((m, n),
                                     move |(i, j)| (i + j * m) as Symbolic)
        .into_dyn();
    ProgState::new((m * n) as Symbolic, array, "trove")
}

fn main() {
    let spec = trove(3, 8);
    println!("{}", spec);

    println!("Basis sets tests");

}

#[cfg(test)]
mod tests {
    use super::trove;
    use crate::swizzle_ops::{fan,rotate,OpAxis};
    use crate::state::ProgState;

    fn fixed_solution_from_scala_3x8() -> ProgState {
        let initial = ProgState::linear(24, &[3, 8]);
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
