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

use crate::misc::{ShapeVec, equal_except};
use crate::state::{Gather, Value, Symbolic};

use crate::operators::{identity, transpose, stack, rot_idx_r, rot_idx_l};
use crate::operators::swizzle::{simple_xforms, simple_rotations,
                                all_xforms, all_rotations};
use crate::operators::select::{reg_select, cond_keep, general_select};
use crate::operators::load::{load_rep, load_trunc, load_grid_2d, broadcast};
use crate::operators::hvx::{hvx_2x2, hvx_2x1, HvxRegs};

use ndarray::{Array, Ix, ArrayD};
use itertools::iproduct;

use std::collections::{HashSet, BTreeMap};
use std::convert::TryInto;
use std::iter::FromIterator;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Opt {
    Flag,
    Int(isize),
    Arr(Vec<isize>),
}

pub type OptMap = BTreeMap<String, Opt>;

use Opt::*;

pub fn int_option(options: Option<&OptMap>, key: &str) -> Option<isize> {
    if let Some(Int(n)) = options.and_then(|m| m.get(key)) {
        Some(*n)
    }
    else {
        None
    }
}

fn required_size_option(options: Option<&OptMap>, key: &str) -> Result<usize> {
    int_option(options, key).and_then(|n| n.try_into().ok())
        .ok_or_else(|| Error::from(ErrorKind::MissingOption(key.to_string())))
}

fn bool_option(options: Option<&OptMap>, key: &str) -> bool {
   options.and_then(|m| m.get(key)) == Some(&Flag)
}

fn array_option<'o, 's>(options: Option<&'o OptMap>, key: &'s str) -> Option<&'o [isize]> {
    if let Some(Arr(a)) = options.and_then(|m| m.get(key)) {
        Some(a)
    }
    else {
        None
    }
}

fn parse_hvx_opts(options: Option<&OptMap>, inplace: bool, fresh: bool,
                  in_shape: &[Ix], out_shape: &[Ix],
                  two_in: bool, two_out: bool) -> Result<(Vec<HvxRegs>, bool)> {
    let permutations = bool_option(options, "swaps");
    let all = !inplace && !fresh;

    let fixed_ins = array_option(options, "in");
    let in_lim = int_option(options, "in_lim").map(|v| v as usize)
        .unwrap_or(in_shape[0]);
    let fixed_outs = array_option(options, "out");

    let n_in = in_shape[0];
    let n_out = out_shape[0];

    let ins: Vec<(usize, Option<usize>)> =
        if two_in {
            if let Some(us) = fixed_ins {
                if us.len() % 2 != 0 {
                    return Err(ErrorKind::BadOptionLength("in".into(), us.len() + 1).into());
                }
                us.chunks(2).map(|s| (s[0] as usize, Some(s[1] as usize))).collect()
            }
            else {
                iproduct![0..in_lim, 0..in_lim].map(|(u, v)| (u, Some(v))).collect()
            }
        }
        else {
            if let Some(us) = fixed_ins {
                us.iter().copied().map(|i| (i as usize, None)).collect()
            }
            else {
                (0..in_lim).map(|i| (i, None)).collect()
            }
        };

    let outs: Vec<(usize, Option<usize>)> =
        if fresh {
            if n_out <= n_in {
                let mut wanted_shape = out_shape.to_vec();
                wanted_shape[0] = n_in + (if two_out { 2 + n_in % 2 } else { 1 });
                return Err(ErrorKind::ShapeMismatch(out_shape.to_vec(),
                                                    wanted_shape).into());
            }
            let delta = n_out - n_in;
            if two_out {
                if delta % 2 != 0 {
                    let mut wanted_shape = in_shape.to_vec();
                    wanted_shape[0] += 1;
                    return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(),
                                                        wanted_shape).into());
                }
                (0..delta/2).map(|i| (n_in + 2 * i,
                                      Some(n_in + 2 * i + 1)))
                    .collect()
            }
            else {
                (0..delta).map(|i| (n_in + i, None)).collect()
            }
        }
        else if let Some(ds) = fixed_outs {
            if two_out {
                if ds.len() % 2 != 0 {
                    return Err(ErrorKind::BadOptionLength("out".into(),
                                                          ds.len() + 1).into());
                }
                ds.chunks(2).map(|s| (s[0] as usize, Some(s[1] as usize))).collect()
            }
            else {
                ds.iter().copied().map(|i| (i as usize, None)).collect()
            }
        }
        else if all {
            if two_out {
                (0..n_out/2).map(|i| (2 * i, Some(2 * i + 1))).collect()
            }
            else {
                (0..n_out).map(|i| (i, None)).collect()
            }
        }
        else { vec![] };

    // The +1 is for inplace, and is added always just to be safe
    let mut ret = Vec::with_capacity(ins.len() * (outs.len() + 1));
    for (u, v) in ins {
        if inplace {
            if !two_out || u % 2 == 0 {
                let d = u;
                let dd = if two_out { Some(u + 1) } else { None };
                ret.push(HvxRegs { u, v, d, dd });
            }
            if let Some(d) = v {
                if !two_out || d % 2 == 0 {
                    let dd = if two_out { Some(d + 1) } else { None };
                    ret.push(HvxRegs { u, v, d, dd })
                }
            }
        }
        for &(d, dd) in &outs {
            ret.push(HvxRegs {u, v, d, dd });
        }
    }
    Ok((ret, permutations))
}

fn parse_swizzle_options(options: Option<&OptMap>) -> Result<(usize, usize, usize)> {
    let main_idx = required_size_option(options, "main")?;
    let second_idx = required_size_option(options, "second")?;
    let out_idx = required_size_option(options, "out")?;
    Ok((main_idx, second_idx, out_idx))
}


pub fn gather(name: &str, in_shapes: &[ShapeVec], out_shape: &[usize],
              options: Option<&OptMap>) -> Result<Vec<Gather>> {
    match name {
        "id" | "identity" | "reshape" => {
            identity(in_shapes, out_shape)
        },
        "transpose" => {
            transpose(in_shapes, out_shape)
        },
        "stack" => {
            stack(in_shapes, out_shape)
        }
        "rot_idx" => {
            if let Some(r) = int_option(options, "r") {
                if int_option(options, "l").is_some() {
                    return Err(ErrorKind::BadOptionLength("l".into(), 0).into());
                }
                let r = r as Ix;
                rot_idx_r(in_shapes, out_shape, r)
            }
            else if let Some(l) = int_option(options, "l") {
                let l = l as Ix;
                rot_idx_l(in_shapes, out_shape, l)
            }
            else {
                Err(ErrorKind::MissingOption("l or r".into()).into())
            }
        },
        "load_rep" => {
            load_rep(in_shapes, out_shape)
        },
        "load_trunc" => {
            load_trunc(in_shapes, out_shape)
        },
        "load_grid_2d" => {
            load_grid_2d(in_shapes, out_shape)
        },
        "broadcast" => {
            let group = int_option(options, "group").unwrap_or(0);
            broadcast(in_shapes, out_shape, group as usize)
        },
        "xforms_no_group" | "xforms" |
        "row_xforms_no_group" | "row_xforms" |
        "col_xforms_no_group" | "col_xforms" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }
            let in_shape: &[usize] = in_shapes[0].as_slice();

            let params =
                if options.is_some() {
                    parse_swizzle_options(options)
                } else if name.starts_with("row_") {
                    Ok((1, 0, 1))
                } else if name.starts_with("col_") {
                    Ok((0, 1, 0))
                } else {
                    Err(ErrorKind::MissingOption("main".to_string()).into())
                };
            let (main_idx, second_idx, out_idx) = params?;
            if !equal_except(in_shape, out_shape, out_idx) {
                return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
            }

            if name.ends_with("_no_group") {
                simple_xforms(in_shape, out_shape, main_idx, second_idx, out_idx)
            }
            else {
                all_xforms(in_shape, out_shape, main_idx, second_idx, out_idx)
            }
        },
        "rots_no_group" | "rots" |
        "row_rots_no_group" | "row_rots" |
        "col_rots_no_group" | "col_rots" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }
            let in_shape: &[usize] = in_shapes[0].as_slice();

            let params =
                if options.map_or(false, |m| m.contains_key("main")) {
                    parse_swizzle_options(options)
                } else if name.starts_with("row_") {
                    Ok((1, 0, 1))
                } else if name.starts_with("col_") {
                    Ok((0, 1, 0))
                } else {
                    Err(ErrorKind::MissingOption("main".to_string()).into())
                };
            let (main_idx, second_idx, out_idx) = params?;
            if !equal_except(in_shape, out_shape, out_idx) {
                return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
            }

            if name.ends_with("_no_group") {
                simple_rotations(in_shape, out_shape, main_idx, second_idx, out_idx)
            }
            else {
                all_rotations(in_shape, out_shape, main_idx, second_idx, out_idx)
            }
        },
        "reg_select_no_consts" | "reg_select" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }
            let in_shape: &[usize] = in_shapes[0].as_slice();

            if out_shape[0..out_shape.len()-1] != in_shape[0..in_shape.len()-1] {
                return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
            }
            if in_shape[in_shape.len()-1] != 2 {
                let mut correct_shape = in_shape.to_vec();
                correct_shape[in_shape.len()-1] = 2;
                return Err(ErrorKind::ShapeMismatch(correct_shape, in_shape.to_vec()).into());
            }
            let n = out_shape[0] as isize;
            // As in Swizzle Inventor
            let default_consts = [0, 1, -1, n, -n];
            let consts = if name == "reg_select_no_consts" {
                &[0]
            } else {
                array_option(options, "consts").unwrap_or(&default_consts)
            };
            reg_select(out_shape, consts)
        },
        "cond_keep_no_consts" | "cond_keep" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }

            if out_shape != in_shapes[0].as_slice() {
                return Err(ErrorKind::ShapeMismatch(in_shapes[0].to_vec(), out_shape.to_vec()).into())
            }
            let n = out_shape[0] as isize;
            // As is Swizzle Inventor
            let default_consts = [0, 1, -1, n, -n];
            let consts = if name == "cond_keep_no_consts" {
                &[0]
            } else {
                array_option(options, "consts").unwrap_or(&default_consts)
            };
            let mut restrict = std::collections::BTreeMap::new();
            if let Some(v) = array_option(options, "restrict") {
                for s in v.chunks(2) {
                    restrict.insert(s[0] as usize, s[1] as usize);
                }
            }
            cond_keep(out_shape, consts, &restrict)
        },
        "general_select_no_consts" | "general_select" => {
            let axis = required_size_option(options, "axis")?;
            let n = out_shape[axis] as isize;
            // As in Swizzle Inventor
            let default_consts = [0, 1, -1, n, -n];
            let consts = if name == "general_select_no_consts" {
                &[0]
            } else {
                array_option(options, "consts").unwrap_or(&default_consts)
            };
            let dims: Vec<usize> =
                array_option(options, "dims")
                .map(|v| v.iter().copied()
                     .map(|i| i as usize).collect())
                .unwrap_or_else(|| (0..out_shape.len()).collect());
            general_select(in_shapes, out_shape, axis,
                           consts, &dims)
        },
        "hvx_2x2" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }
            let in_shape: &[usize] = in_shapes[0].as_slice();

            let (regs, swaps) =
                parse_hvx_opts(options,
                               bool_option(options, "inplace"),
                               bool_option(options, "fresh"),
                               in_shape, out_shape,
                               true, true)?;
            hvx_2x2(in_shape, out_shape, &regs, swaps)
        },
        "hvx_2x1" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }
            let in_shape: &[usize] = in_shapes[0].as_slice();

            let (regs, swaps) =
                parse_hvx_opts(options,
                               bool_option(options, "inplace"),
                               bool_option(options, "fresh"),
                               in_shape, out_shape,
                               true, false)?;
            hvx_2x1(in_shape, out_shape, &regs, swaps)
        },
        "hvx_inplace" => {
            if in_shapes.len() != 1 {
                return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
            }
            let in_shape: &[usize] = in_shapes[0].as_slice();

            let (regs_2_2, swaps) =
                parse_hvx_opts(options,
                               true, false,
                               in_shape, out_shape,
                               true, true)?;
            let (regs_2_1, _) =
                parse_hvx_opts(options,
                               true, false,
                               in_shape, out_shape,
                               true, false)?;
            let v2_2 = hvx_2x2(in_shape, out_shape, &regs_2_2, swaps)?;
            let v2_1 = hvx_2x1(in_shape, out_shape, &regs_2_1, swaps)?;
            let mut ret = HashSet::<Gather>::from_iter(v2_2.into_iter());
            ret.extend(v2_1.into_iter());
            Ok(ret.into_iter().collect::<Vec<_>>().into())
        },
        other => {
            return Err(ErrorKind::UnknownBasisType(other.to_owned()).into())
        }
    }
}

// Targets:
pub fn trove(m: Ix, n: Ix) -> ArrayD<Value> {
    Array::from_shape_fn((m, n),
                         move |(i, j)|
                         Value::Symbol((j + i * n) as Symbolic))
        .into_dyn()
}

pub fn trove_sum(m: Ix, n: Ix) -> ArrayD<Value> {
    let arr: ndarray::Array1<Value> =
        (0..m).map(|i|
                   (0..n).map(|j| Value::Symbol((j + i * n) as Symbolic)).collect())
        .map(Value::fold).collect();
    arr.into_dyn()
}

fn convolve_dealg(width: Ix, k: Ix) -> ArrayD<Value> {
    Array::from_shape_fn((width, k),
                         move |(i, j)| Value::Symbol((i + j)  as Symbolic))
        .into_dyn()
}

fn convolve(width: Ix, k: Ix) -> ArrayD<Value> {
    // Hopefully u16 is enough for everyone
    let width = width as Symbolic;
    let k = k as Symbolic;
    let arr: ndarray::Array1<Value> =
        (0..width).map(|w|
                       (0..k).map(|i| Value::Symbol(w + i)).collect())
        .map(Value::fold).collect();
    arr.into_dyn()
}

fn weighted_convolve(width: Ix, k: Ix) -> ArrayD<Value> {
    // Hopefully u16 is enough for everyone
    let width = width as Symbolic;
    let k = k as Symbolic;
    let weight_min = width + k - 1;
    let arr: ndarray::Array1<Value> =
        (0..width).map(|w|
                       (0..k).map(|i| Value::fold(vec![Value::Symbol(w + i),
                                                       Value::Symbol(weight_min + i)]))
                       .collect())
        .map(Value::fold).collect();
    arr.into_dyn()
}

fn stencil_2d(width: Ix, k: Ix) -> ArrayD<Value> {
    let n = width + k - 1;
    let source: Vec<Value> =
        (0..(n * n)).map(|i| Value::Symbol(i as Symbolic)).collect();
    let arr =
        ndarray::Array2::from_shape_fn(
            (width, width),
            |(i, j)| Value::fold(
                iproduct!(0..k, 0..k).map(|(ii, jj)| source[j + jj + n * (i + ii)].clone())
                    .collect()));
    arr.into_dyn()
}

pub fn poly_mult(n: Ix) -> ArrayD<Value> {
    let ns = n as Symbolic;
    let arr: ndarray::Array1<Value> =
        (0..ns).map(|i| Value::fold(
            (0..=i).map(
                |j| Value::fold(vec![Value::Symbol(j), Value::Symbol(ns + i - j)]))
                .collect()))
        .chain((0..ns).map(|i| Value::fold(
            ((i + 1)..ns).map(
                |j| Value::fold(vec![Value::Symbol(j), Value::Symbol(ns + ns + i - j)]))
                .collect())))
        .collect();
    arr.into_dyn()
}

pub fn goal(name: &str, shape: &[Ix],
            options: Option<&OptMap>) -> Result<ArrayD<Value>> {
    match name {
        "trove" => {
            match shape {
                &[m, n] => Ok(trove(m, n)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
            }
        }
        "trove_sum" => {
            match shape {
                // TODO, transitional
                &[m] => {
                    let n = required_size_option(options, "n")?;
                    Ok(trove_sum(m, n))
                },
                &[m, n] => Ok(trove_sum(m, n)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 1).into())
            }
        }
        "conv_dealg" => {
            match shape {
                &[width, k] => Ok(convolve_dealg(width, k)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
            }
        }
        "conv" => {
            match shape {
                &[width] => {
                    let k = required_size_option(options, "k")?;
                    Ok(convolve(width, k))
                },
                // TODO: transitional
                &[width, k] => Ok(convolve(width, k)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 1).into())
            }
        }
        "weight_conv" => {
            match shape {
                &[width] => {
                    let k = required_size_option(options, "k")?;
                    Ok(weighted_convolve(width, k))
                },
                // TODO: transitional
                &[width, k] => Ok(weighted_convolve(width, k)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 1).into())
            }
        }
        "stencil_2d" => {
            match shape {
                &[width, width2] => {
                    if width != width2 {
                        return Err(ErrorKind::InvalidShapeDim(shape.to_owned(), 1).into());
                    }
                    let k = required_size_option(options, "k")?;
                    Ok(stencil_2d(width, k))
                },
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 1).into())
            }
        }
        "poly_mult" => {
            match shape {
                &[_n] => {
                    let w = required_size_option(options, "w")?;
                    Ok(poly_mult(w))
                },
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 1).into())
            }
        }
        other => Err(ErrorKind::UnknownProblem(other.to_owned()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    pub fn trove_works() {
        let trove_spec: [Symbolic; 12] = [0, 1, 2,
                                          3, 4, 5,
                                          6, 7, 8,
                                          9, 10, 11];
        let trove_spec: Vec<Value> = (&trove_spec).iter().copied().map(Value::Symbol).collect();
        let trove_spec_arr = Array::from_shape_vec((3, 4), trove_spec).unwrap().into_dyn();
        assert_eq!(trove_spec_arr, trove(3, 4));
    }

    #[test]
    pub fn conv_1d_end_works() {
        let conv_final: [Symbolic; 4 * 3] = [0, 1, 2,
                                             1, 2, 3,
                                             2, 3, 4,
                                             3, 4, 5];
        let conv_final = (&conv_final).iter().copied().map(Value::Symbol).collect();
        let conv_final_arr = Array::from_shape_vec((4, 3), conv_final).unwrap().into_dyn();
        assert_eq!(conv_final_arr, convolve_dealg(4, 3));
    }

}
