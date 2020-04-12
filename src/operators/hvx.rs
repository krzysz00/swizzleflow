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
use crate::operators::{OpSetKind, identity_gather};

use ndarray::Ix;

use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HvxRegs {
    pub u: usize,
    pub v: Option<usize>,
    pub d: usize,
    pub dd: Option<usize>
}

impl fmt::Display for HvxRegs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.d)?;
        if let Some(dd) = self.dd {
            write!(f, ",{}", dd)?;
        }
        write!(f, "<-{}", self.u)?;
        if let Some(v) = self.v {
            write!(f, ",{}", v)?;
        }
        Ok(())
    }
}

fn opt_eq<T: Eq>(a: Option<T>, b: T) -> bool {
    a.map(|e| e == b).unwrap_or(false)
}

fn generalize_instr<T>(map: T,
                       &HvxRegs {u, v, d, dd}: &HvxRegs,
                       in_shape: &[Ix], out_shape: &[Ix], name: impl Into<String>)
                       -> Gather
where T: Fn(Ix, Ix, Ix) -> (Ix, Ix) {
    let n = in_shape[1];
    Gather::new(
        out_shape,
        |coords| {
            let r = coords[0];
            let i = coords[1];
            let is_d = r == d;
            let is_d2 = opt_eq(dd, r);
            if !is_d && !is_d2 {
                to_opt_ix(coords, in_shape)
            }
            else {
                let (r_src, i_src) = map(if is_d { 0 } else { 1 }, i, n);
                if r_src == 0 {
                    to_opt_ix(&[u, i_src], in_shape)
                }
                else if r_src == 1 {
                    to_opt_ix(&[v.expect("Two source registers should have v"),
                                i_src], in_shape)
                }
                else {
                    panic!("HVX instruction template accessed too many inputs");
                }
            }
    }, name)
}

fn swap_regs(regs: &HvxRegs, in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    generalize_instr(|r, i, _| (if r == 0 { 1 } else { 0 }, i),
                     regs, in_shape, out_shape, format!("swap_regs({})", regs))
}

fn vshuffo(regs: &HvxRegs,
           in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    generalize_instr(
        |_r, i, _| (if i % 2 == 0 { 1 } else { 0 }, (i & !1) + 1),
        regs, in_shape, out_shape, format!("vsuffo({})", regs))
}

fn vshuffe(regs: &HvxRegs, in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    generalize_instr(
        |_r, i, _| (if i % 2 == 0 { 1 } else { 0 }, i & !1 ),
        regs, in_shape, out_shape, format!("vsuffe({})", regs))
}

fn vshuffoe(regs: &HvxRegs, in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    generalize_instr(
        |r, i, _| (if i % 2 == 0 { 1 } else { 0 }, (i & !1) + (if r == 0 { 0 } else { 1 })),
        regs, in_shape, out_shape, format!("vshuffoe({})", regs))
}

fn vswap(mask: usize, regs: &HvxRegs,
         in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    // If the i-th bit of the mask is 1, register 0 in the destination gets
    // from the 1st source register (v), and register 1 reads from register 0 (u)
    // So we need an xnor
    generalize_instr(|r, i, _| (!(r ^ (mask >> i)) & 1, i),
                     regs, in_shape, out_shape,
                     format!("vswap({}, {})", regs, mask))
}

fn vmux(mask: usize, regs: &HvxRegs,
        in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    generalize_instr(|_r, i, _| (!(mask >> i) & 1, i),
                     regs, in_shape, out_shape,
                     format!("vmux({}, {})", regs, mask))
}

fn hvx_errs(in_shape: &[Ix], out_shape: &[Ix]) -> Result<()> {
    if out_shape.len() != 2 {
        return Err(ErrorKind::InvalidShapeDim(out_shape.to_owned(), 2).into())
    }
    if in_shape.len() != 2 {
        return Err(ErrorKind::InvalidShapeDim(in_shape.to_owned(), 2).into());
    }
    if out_shape[1] != in_shape[1] {
        return Err(ErrorKind::AxisLengthMismatch(1, out_shape[1], 1, in_shape[1]).into());
    }
    Ok(())
}

pub fn hvx_2x2(out_shape: &[Ix], in_shape: &[Ix],
               regs: &[HvxRegs], swaps: bool) -> Result<OpSetKind> {
    hvx_errs(in_shape, out_shape)?;
    let n = in_shape[1];
    let mut ret = HashSet::new();
    ret.insert(identity_gather(in_shape));
    for r in regs {
        ret.insert(vshuffoe(r, in_shape, out_shape));
        ret.insert(swap_regs(r, in_shape, out_shape));

        if swaps {
            ret.extend((0..(1 << n)).map(|i| vswap(i, r,
                                                   in_shape, out_shape)));
        }
    }
    return Ok(ret.into_iter().collect::<Vec<_>>().into())
}

pub fn hvx_2x1(out_shape: &[Ix], in_shape: &[Ix],
               regs: &[HvxRegs], swaps: bool) -> Result<OpSetKind> {
    hvx_errs(in_shape, out_shape)?;

    let n = in_shape[1];
    let mut ret = HashSet::new();
    ret.insert(identity_gather(in_shape));
    for r in regs {
        ret.insert(vshuffo(r, in_shape, out_shape));
        ret.insert(vshuffe(r, in_shape, out_shape));

        if swaps {
            ret.extend((0..(1 << n)).map(|i| vmux(i, r,
                                                  in_shape, out_shape)));
        }
    }
    return Ok(ret.into_iter().collect::<Vec<_>>().into())
}

// TODO, get the 1x1 permutation network thing working
