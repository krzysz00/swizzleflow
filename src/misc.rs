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
pub(crate) const EPSILON: f32 = 1e-5;
pub type ShapeVec = smallvec::SmallVec<[usize; 3]>;

pub fn time_since(start: std::time::Instant) -> f64 {
    let dur = start.elapsed();
    dur.as_secs() as f64 + (dur.subsec_nanos() as f64 / 1.0e9)
}

use ndarray::Array2;
pub fn regularize_float_mat(arr: &mut Array2<f32>) {
    arr.mapv_inplace(|v| if v.abs() < EPSILON { 0.0 } else { 1.0 })
}