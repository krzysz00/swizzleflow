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
use crate::errors::*;
use crate::misc::ShapeVec;

use crate::state::Gather;
use crate::operators::OpSet;

use ndarray::Ix;

use std::cmp::min;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Mode {
    Rep,
    Trunc,
}

fn load(in_shape: &[Ix], out_shape: &[Ix], mode: Mode) -> Result<OpSet> {
    let name = match mode {
        Mode::Rep => "load_rep",
        Mode::Trunc => "load_trunc",
    };

    let in_bound = in_shape[0];
    let out_split = out_shape.len() - (in_shape.len() - 1);

    if &in_shape[1..] != &out_shape[out_split..] {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }
    let gather =
        Gather::new(in_shape.len(), out_shape,
                    move |out_idxs: &[Ix], in_idx_output: &mut Vec<Ix>| {
                        let mut linear_idx = 0;
                        for (idx, stride) in out_idxs[0..out_split].iter()
                            .zip(out_shape[0..out_split].iter()).rev() {
                                linear_idx = linear_idx * stride + idx;
                            }
                        linear_idx = match mode {
                            Mode::Rep => linear_idx % in_bound,
                            Mode::Trunc => min(linear_idx, in_bound),
                        };
                        in_idx_output.push(linear_idx);

                        in_idx_output.extend(&out_idxs[out_split..]);
                    }, name);
    Ok(OpSet::new(name, vec![gather],
                  ShapeVec::from_slice(in_shape), ShapeVec::from_slice(out_shape)))
}

pub fn load_rep(in_shape: &[Ix], out_shape: &[Ix]) -> Result<OpSet> {
    load(in_shape, out_shape, Mode::Rep)
}

pub fn load_trunc(in_shape: &[Ix], out_shape: &[Ix]) -> Result<OpSet> {
    load(in_shape, out_shape, Mode::Trunc)
}
