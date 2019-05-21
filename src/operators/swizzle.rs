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
use crate::state::Gather;
use super::OpSet;
use crate::errors::*;

use ndarray::{Ix,Ixs};
use num_integer::Integer;

use std::collections::HashSet;
use std::cmp;

use itertools::iproduct;
use smallvec::smallvec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpAxis { Rows, Columns }
use OpAxis::*;

pub fn fan(m: Ix, n: Ix, perm_within: OpAxis, stable_scale: Ix, cf: Ix) -> Gather {
    let len_of_stable = match perm_within { Rows => n, Columns => m };
    let name = format!("fan({},{})", stable_scale, cf);
    match perm_within {
        Rows => Gather::new(2, &[m, n],
                            move |idxs: &[Ix], out: &mut Vec<Ix>| {
                                let i = idxs[0];
                                let j = idxs[1];
                                let cf = stable_scale * i + cf;
                                let df = len_of_stable / len_of_stable.gcd(&cf);
                                let get_from = (cf * j + j / df).mod_floor(&len_of_stable);
                                out.extend(&[i, get_from]);
                            }, name),
        Columns => Gather::new(2, &[m, n],
                               move |idxs: &[Ix], out: &mut Vec<Ix>| {
                                   let i = idxs[0];
                                   let j = idxs[1];
                                   let cf = stable_scale * j + cf;
                                   let df = len_of_stable / len_of_stable.gcd(&cf);
                                   let get_from = (cf * i + i / df).mod_floor(&len_of_stable);
                                   out.extend(&[get_from, j]);
                               }, name),
    }
}

pub fn rotate(m: Ix, n: Ix, perm_within: OpAxis, stable_scale: Ixs, div: Ix, shift: Ixs) -> Gather {
    let len_of_stable = match perm_within { Rows => n, Columns => m };
    let name = format!("rot({},{},{})", stable_scale, div, shift);
    match perm_within {
        Rows => Gather::new(2, &[m, n],
                            move |idxs: &[Ix], out: &mut Vec<Ix>| {
                                let i = idxs[0];
                                let is = i as isize;
                                let j = idxs[1];
                                let js = j as isize;
                                let loc = stable_scale * is + i.div_floor(&div) as isize + shift + js;
                                let get_from = loc.mod_floor(&(len_of_stable as isize)) as usize;
                                out.extend(&[i, get_from]);
                            }, name),
        Columns => Gather::new(2, &[m, n],
                               move |idxs: &[Ix], out: &mut Vec<Ix>| {
                                   let i = idxs[0];
                                   let is = i as isize;
                                   let j = idxs[1];
                                   let js = j as isize;
                                   let loc = stable_scale * js + j.div_floor(&div) as isize + shift + is;
                                   let get_from = loc.mod_floor(&(len_of_stable as isize)) as usize;
                                   out.extend(&[get_from, j]);
                               }, name),
    }
}

pub fn identity(shape: &[Ix]) -> Gather {
    Gather::new(shape.len(), shape, |idxs, ops| ops.extend(idxs), "id")
}

pub fn simple_fans(shape: &[Ix], perm_within: OpAxis) -> Result<OpSet> {
    let mut ret = HashSet::new();

    if shape.len() != 2 {
        return Err(ErrorKind::InvalidShapeDim(shape.to_owned(), 2).into());
    }
    let m = shape[0];
    let n = shape[1];

    ret.insert(identity(shape));
    let k_bound = cmp::max(m, n);
    let c_bound = match perm_within { Rows => n, Columns => m };
    ret.extend(iproduct!((0..k_bound), (0..c_bound)).map(|(k, c)| fan(m, n, perm_within, k, c)));
    let name = match perm_within { Rows => "sFr", Columns => "sFc"};
    Ok(OpSet::new(name, ret.into_iter().collect(), smallvec![m, n], smallvec![m, n]))
}

pub fn simple_rotations(shape: &[Ix], perm_within: OpAxis) -> Result<OpSet> {
    let mut ret = HashSet::new();

    if shape.len() != 2 {
        return Err(ErrorKind::InvalidShapeDim(shape.to_owned(), 2).into());
    }
    let m = shape[0];
    let n = shape[1];

    ret.insert(identity(shape));
    let k_bound = cmp::max(m, n) as isize;
    let c_bound = match perm_within { Rows => n, Columns => m } as isize;
    let d_bound = match perm_within { Rows => m, Columns => n};
    ret.extend(iproduct!((-k_bound+1..k_bound),
                         (2..=d_bound).filter(|i| d_bound % i == 0),
                         (-c_bound+1..c_bound))
               .map(|(k, d, c)| rotate(m, n, perm_within, k, d, c)));
    let name = match perm_within { Rows => "sRr", Columns => "sRc"};
    Ok(OpSet::new(name, ret.into_iter().collect(), smallvec![m, n], smallvec![m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scala_3x16_basis_sizes() {
        assert_eq!(simple_fans(&[3, 16], OpAxis::Columns).unwrap().ops.len(), 5);
        assert_eq!(simple_rotations(&[3, 16], OpAxis::Columns).unwrap().ops.len(), 36);
        assert_eq!(simple_fans(&[3, 16], OpAxis::Rows).unwrap().ops.len(), 255);
        assert_eq!(simple_rotations(&[3, 16], OpAxis::Rows).unwrap().ops.len(), 256);
    }

    #[test]
    #[should_panic]
    fn test_shape_interface() {
        simple_fans(&[3], OpAxis::Columns).unwrap();
    }
}
