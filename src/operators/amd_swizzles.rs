// Copyright (C) 2021 Krzysztof Drewniak et al.

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
use crate::misc::ShapeVec;

use crate::operators::permutations::permutations_in_group;

use std::collections::HashSet;
use std::iter::FromIterator;

use ndarray::Ix;
use num_integer::Integer;

fn swizzle_b32_far(in_shape: &[Ix], out_shape: &[Ix],
                       axis: Ix, restrict: Option<(Ix, Ix)>,
                       and: Ix, or: Ix, xor: Ix) -> Gather {
    let name = format!("swizzle_b32[and={},or={},xor={}]", and, or, xor);
    Gather::new(out_shape, move |out_idx: &[Ix]| {
        if let Some((ax, val)) = restrict {
            if out_idx[ax] != val {
                return (0, to_opt_ix(out_idx, in_shape))
            }
        }
        let mut out = ShapeVec::from_slice(out_idx);
        let out_coord: usize = out[axis];
        let out_group = out_coord.div_floor(&32) * 32;
        let out_in_group = out_coord.mod_floor(&32);
        out[axis] = out_group + ((out_in_group & and) | or) ^ xor;
        (0, to_opt_ix(&out, in_shape))
    }, name)
}

pub fn amd_swizzles_perm(in_shape: &[Ix], out_shape: &[Ix],
                              axis: Ix, group: Ix, restrict: Option<(Ix, Ix)>) -> Result<Vec<Gather>> {
    // Error checking happens here
    let small_swizzles = permutations_in_group(in_shape, out_shape, axis, group, restrict)?;
    let mut ret: HashSet<_, std::collections::hash_map::RandomState> =
        HashSet::from_iter(small_swizzles.into_iter());
    const PERM_AND_MASK: Ix = 0x1f;
    const PERM_OR_MASK: Ix = 0;
    ret.extend((0..32).map(|xor| {
        swizzle_b32_far(in_shape, out_shape,
            axis, restrict,
            PERM_AND_MASK, PERM_OR_MASK, xor)
    }));
    Ok(ret.into_iter().collect::<Vec<_>>())
}