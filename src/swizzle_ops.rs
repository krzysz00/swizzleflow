use crate::state::Gather;

use ndarray::{Ix,Ixs};
use num_integer::Integer;

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
