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
mod state;
mod swizzle_ops;

use ndarray::{Array,Ix};
use state::{ProgState,Symbolic};
use swizzle_ops::{fan,rotate,OpAxis};

fn trove(m: Ix, n: Ix) -> ProgState {
    let array = Array::from_shape_fn((m, n),
                                     move |(i, j)| (i + j * m) as Symbolic)
        .into_dyn();
    ProgState::new((m * n) as Symbolic, array, "trove")
}

fn fixed_solution_from_scala() -> ProgState {
    let initial = ProgState::linear(24, &[3, 8]);
    let s1 = initial.gather_by(&fan(3, 8, OpAxis::Columns, 0, 2));
    let s2 = s1.gather_by(&rotate(3, 8, OpAxis::Columns, -7, 8, 0));
    let s3 = s2.gather_by(&fan(3, 8, OpAxis::Rows, 0, 3));
    let s4 = s3.gather_by(&rotate(3, 8, OpAxis::Rows, -5, 3, 0));
    s4
}

fn main() {
    println!("{}", trove(3, 8));
    println!("{}", fixed_solution_from_scala());
    println!("{:?}", trove(3, 8));
}
