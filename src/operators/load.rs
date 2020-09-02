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

use crate::state::{Gather,to_opt_ix};

use std::cmp::min;

use ndarray::Ix;

use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Mode {
    Rep,
    Trunc,
}

fn load(in_shape: &[Ix], out_shape: &[Ix], mode: Mode) -> Result<Vec<Gather>> {
    let name = match mode {
        Mode::Rep => "load_rep",
        Mode::Trunc => "load_trunc",
    };

    let in_bound = in_shape[0];
    let out_split = out_shape.len() - (in_shape.len() - 1);

    if in_shape[1..] != out_shape[out_split..] {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }
    let gather =
        Gather::new(out_shape,
                    move |out_idxs: &[Ix]| {
                        let mut linear_idx = 0;
                        for (idx, stride) in out_idxs[0..out_split].iter()
                            .zip(out_shape[0..out_split].iter()).rev() {
                                linear_idx = linear_idx * stride + idx;
                            }
                        linear_idx = match mode {
                            Mode::Rep => linear_idx % in_bound,
                            Mode::Trunc => min(linear_idx, in_bound),
                        };
                        let mut storage = SmallVec::<[usize; 4]>::new();
                        storage.push(linear_idx);
                        storage.extend((&out_idxs[out_split..]).iter().copied());
                        to_opt_ix(&storage, &in_shape)
                    }, name);
    Ok(vec![gather])
}

pub fn load_rep(in_shape: &[Ix], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    load(in_shape, out_shape, Mode::Rep)
}

pub fn load_trunc(in_shape: &[Ix], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    load(in_shape, out_shape, Mode::Trunc)
}

pub fn broadcast(in_shape: &[Ix], out_shape: &[Ix], group: Ix) -> Result<Vec<Gather>> {
    let name = "broadcast";
    let out_split = out_shape.len() - in_shape.len() + group;

    if &in_shape[group..] != &out_shape[out_split..] {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }
    let gather =
        Gather::new(out_shape,
                    move |out_idxs: &[Ix]| {
                        let mut idxs = Vec::with_capacity(8);
                        idxs.extend_from_slice(&out_idxs[..group]);
                        idxs.extend_from_slice(&out_idxs[out_split..]);
                        to_opt_ix(&idxs, in_shape)
                    }, name);
    Ok(vec![gather])
}

pub fn load_grid_2d(in_shape: &[Ix], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    if in_shape.len() != 2 {
        return Err(ErrorKind::InvalidShapeDim(in_shape.to_vec(), 2).into());
    }
    if out_shape.len() != 4 {
        return Err(ErrorKind::InvalidShapeDim(out_shape.to_vec(), 4).into());
    }
    let ni = out_shape[0];
    let nj = out_shape[1];
    let gather =
        Gather::new(out_shape,
                    move |idxs: &[Ix]| {
                        let j = idxs[1] + nj * idxs[3];
                        let i = idxs[0] + ni * idxs[2];
                        to_opt_ix(&[i, j], in_shape)
                    },
        "load_grid_2d");
    Ok(vec![gather])
}
