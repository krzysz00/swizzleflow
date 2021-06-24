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
use crate::misc::ShapeVec;
use crate::state::{Gather, to_opt_ix};
use super::identity_gather;
use crate::errors::*;

use ndarray::{Ix,Ixs};
use num_integer::Integer;

use std::collections::HashSet;

use itertools::iproduct;

pub fn xform(in_shape: &[Ix], out_shape: &[Ix],
             main_axis: Ix, second_axis: Ix, swizzle_axis: Ix,
             cf: Ixs, cr: Ixs, dr: Ix, group: Option<Ix>, wrap: bool) -> Gather {
    let len_of_stable = in_shape[main_axis];
    let group = group.unwrap_or(len_of_stable);
    let wrap_name = if wrap { "wrap," } else { "" };
    let name = if group != len_of_stable {
        format!("xform[group={},{}cf={},cr={},dr={}]", group, wrap_name, cf, cr, dr)
    }
    else {
        format!("xform[{}cf={},cr={},dr={}]", wrap_name, cf, cr, dr)
    };
    let gcd = group.gcd(&(cf.abs() as usize));
    let df = group / gcd;
    Gather::new(&out_shape,
                move |idxs: &[Ix]| {
                    let mut out = ShapeVec::from_slice(idxs);
                    let i = idxs[main_axis];
                    let ig = i.mod_floor(&group);
                    let is = ig as isize;
                    let k = idxs[second_axis];
                    let ks = k as isize;
                    let fan_loc = cf * is + (ig / df) as isize;
                    let raw_rot_loc = cr * ks + k.div_floor(&dr) as isize;
                    let rot_loc =
                        if wrap { raw_rot_loc.mod_floor(&(gcd as isize)) }
                        else { raw_rot_loc };
                    let get_from = (fan_loc + rot_loc).mod_floor(&(group as isize))
                        as usize + ((i / group) * group);
                    out[swizzle_axis] = get_from;
                    (0, to_opt_ix(out.as_slice(), &in_shape))
                }, name)
}

pub fn rotate(in_shape: &[Ix], out_shape: &[Ix],
              main_axis: Ix, _second_axis: Ix, swizzle_axis: Ix,
              shift: Ixs, group: Option<Ix>) -> Gather {
    let len_of_stable = in_shape[main_axis];
    let group = group.unwrap_or(len_of_stable);
    let name = if group != len_of_stable {
        format!("rot[group={},shift={}]", group, shift)
    } else {
        format!("rot[shift={}]", shift)
    };
    Gather::new(out_shape,
                move |idxs: &[Ix]| {
                    let mut out = ShapeVec::from_slice(idxs);
                    let i = idxs[main_axis];
                    let is = i as isize;
                    let loc = shift + is;
                    let get_from = loc.mod_floor(&(group as isize)) as usize
                        + ((i / group) * group);
                    out[swizzle_axis] = get_from;
                    (0, to_opt_ix(out.as_slice(), in_shape))
                }, name)
}

pub fn simple_xforms(in_shape: &[Ix], out_shape: &[Ix],
                     main_axis: Ix, second_axis: Ix,
                     swizzle_axis: Ix) -> Result<Vec<Gather>> {
    let mut ret = HashSet::new();

    if in_shape == out_shape {
        ret.insert(identity_gather(out_shape));
    }

    let cf_bound = in_shape[main_axis] as isize;
    let cr_bound = cf_bound;
    let dr_bound = in_shape[second_axis];
    ret.extend(iproduct!((2..=dr_bound).filter(|i| dr_bound % i == 0).rev(),
                         (0..cr_bound).chain(-1..0),
                         (0..cf_bound).chain(-1..0))
               .map(|(dr, cr, cf)| xform(in_shape, out_shape,
                                         main_axis, second_axis, swizzle_axis,
                                         cf, cr, dr, None, false)));
    Ok(ret.into_iter().collect::<Vec<_>>())
}

pub fn simple_rotations(in_shape: &[Ix], out_shape: &[Ix],
                        main_axis: Ix, second_axis: Ix,
                        swizzle_axis: Ix) -> Result<Vec<Gather>> {
    let mut ret = HashSet::new();
    if in_shape == out_shape {
        ret.insert(identity_gather(out_shape));
    }
    let shift_bound = in_shape[main_axis] as isize;
    ret.extend((0..=shift_bound).chain(-shift_bound+1..0)
               .map(|c| rotate(in_shape, out_shape,
                               main_axis, second_axis, swizzle_axis,
                               c, None)));
    Ok(ret.into_iter().collect::<Vec<_>>())
}

pub fn all_xforms(in_shape: &[Ix], out_shape: &[Ix],
                  main_axis: Ix, second_axis: Ix,
                  swizzle_axis: Ix) -> Result<Vec<Gather>> {
    let mut ret = HashSet::new();

    if in_shape == out_shape {
        ret.insert(identity_gather(out_shape));
    }
    let stable_len = in_shape[main_axis];
    let dr_bound = in_shape[second_axis];

    for g in (2..=stable_len).rev().filter(|i| stable_len % i == 0) {
        let gs = g as isize;
        let cf_bound = gs;
        let cr_bound = gs;
        ret.extend(iproduct!([false, true].iter(),
                             (2..=dr_bound).filter(|i| dr_bound % i == 0).rev(),
                             (0..cr_bound).chain(-1..0),
                             (0..cf_bound).chain(-1..0))
                   .map(|(wrap, dr, cr, cf)|
                        xform(in_shape, out_shape,
                              main_axis, second_axis, swizzle_axis,
                              cf, cr, dr, Some(g), *wrap)));
    }
    Ok(ret.into_iter().collect::<Vec<_>>())
}

pub fn all_rotations(in_shape: &[Ix], out_shape: &[Ix],
                     main_axis: Ix, second_axis: Ix,
                     swizzle_axis: Ix) -> Result<Vec<Gather>> {
    let mut ret = HashSet::new();
    let stable_len = in_shape[main_axis];

    if in_shape == out_shape {
        ret.insert(identity_gather(out_shape));
    }
    for g in (2..=stable_len).rev().filter(|i| stable_len % i == 0) {
        let gs = g as isize;
        ret.extend((0..gs).chain(-gs+1..0)
                   .map(|c| rotate(in_shape, out_shape,
                                   main_axis, second_axis, swizzle_axis,
                                   c, Some(g))));
    }
    Ok(ret.into_iter().collect::<Vec<_>>())
}
