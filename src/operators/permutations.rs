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
use crate::operators::identity_gather;

use std::collections::HashSet;
use std::iter::FromIterator;

use ndarray::Ix;
use num_integer::Integer;
use itertools::Itertools;

use smallvec::smallvec;

fn permute_in_group(in_shape: &[Ix], out_shape: &[Ix],
        axis: Ix, group: Ix, restrict: Option<(Ix, Ix)>,
        permutation: &[Ix]) -> Gather {
    let name = format!("permute_pull[axis={},group={},perm={}]", axis, group,
        permutation.iter().copied().join(":"));
    Gather::new(out_shape,
                move |out_idx: &[Ix]| {
                    if let Some((ax, val)) = restrict {
                        if out_idx[ax] != val {
                            return (0, to_opt_ix(out_idx, in_shape))
                        }
                    }
                    let mut out = ShapeVec::from_slice(out_idx);
                    let out_coord: usize = out[axis];
                    let coord_group = out_coord.div_floor(&group) * group;
                    let lane = out_coord.mod_floor(&group);
                    out[axis] = coord_group + permutation[lane];
                    (0, to_opt_ix(&out, in_shape))
                }, name)
}

pub fn permutations_in_group(in_shape: &[Ix], out_shape: &[Ix],
        axis: Ix, group: Ix, restrict: Option<(Ix, Ix)>) -> Result<Vec<Gather>> {
    if in_shape.len() <= axis || out_shape.len() <= axis {
        return Err(ErrorKind::InvalidShapeDim(
                    if in_shape.len() <= out_shape.len() { in_shape.to_owned()} else { out_shape.to_owned() },
                    axis).into());
    }
    if let Some((ix, val)) = restrict {
        if ix >= out_shape.len() {
            return Err(ErrorKind::InvalidShapeDim(out_shape.to_owned(), ix).into());
        }
        if val >= out_shape[ix] {
            return Err(ErrorKind::AxisLengthMismatch(ix, val, ix, out_shape[ix]).into());
        }
    }

    if in_shape.iter().copied().product::<Ix>() != out_shape.iter().copied().product() {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_owned(), out_shape.to_owned()).into());
    }
    let mut ret = HashSet::new();
    ret.insert(identity_gather(out_shape));

    // Non-recursively generate all permutations
    // See https://en.wikipedia.org/wiki/Heap%27s_algorithm
    let mut perm = ShapeVec::from_iter(0..group);
    let mut perm_stack: ShapeVec = smallvec![0; group];
    let mut i = 1;

    while i < group {
        if perm_stack[i] < i {
            if i % 2 == 0 {
                perm.swap(0, i)
            }
            else {
                perm.swap(perm_stack[i], i)
            }

            ret.insert(permute_in_group(in_shape, out_shape, axis, group, restrict, &perm));
            perm_stack[i] += 1;
            i = 1;
        }
        else {
            perm_stack[i] = 0;
            i += 1;
        }
    }
    Ok(ret.into_iter().collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_perms() -> Result<()> {
        use itertools::Itertools;

        let ret = permutations_in_group(&[4], &[4], 0, 4, None)?;
        assert_eq!(ret.len(), 24);
        let our_perms = ret.iter().map(|g| g.data.iter().map(|d| d.1 as usize).collect::<Vec<_>>()).collect::<HashSet<Vec<Ix>>>();
        let canonical_perms = (0..4).permutations(4).collect::<HashSet<Vec<Ix>>>();
        assert_eq!(our_perms, canonical_perms);
        Ok(())
    }

    #[test]
    pub fn test_grouping() -> Result<()> {
        let ret = permutations_in_group(&[4], &[4], 0, 2, None)?;
        assert_eq!(ret.len(), 2);
        let expected: Vec<Vec<Ix>> = vec![vec![0, 1, 2, 3], vec![1, 0, 3, 2]];
        let expected = expected.into_iter().collect::<HashSet<_>>();
        let actual = ret.iter().map(|g| g.data.iter().map(|d| d.1 as usize).collect()).collect::<HashSet<Vec<Ix>>>();
        assert_eq!(expected, actual);
        Ok(())
    }
}