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

use crate::operators::{SynthesisLevel,OpSet, OpSetKind,
                       identity, transpose};
use crate::state::{ProgState, Domain, Value, Symbolic, DomRef,
                   Gather, to_opt_ix};
use crate::operators::swizzle::{simple_xforms, simple_rotations,
                                all_xforms, all_rotations};
use crate::operators::select::{reg_select, cond_keep, general_select};
use crate::operators::load::{load_rep, load_trunc, load_grid_2d, broadcast};
use crate::expected_syms_util::fold_expected;
use crate::misc::{ShapeVec, extending_set};

use std::collections::{BTreeSet, BTreeMap, HashMap};

use serde::{Serialize, Deserialize};

use ndarray::{Array, ArrayD, Ix};

use smallvec::SmallVec;

type OptionMap = BTreeMap<String, Vec<isize>>;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum GathersDesc {
    Builtin(String),
    Custom(Vec<(String, Vec<u64>)>)
}

impl GathersDesc {
    pub fn to_opset_kind(&self, in_shape: &[usize], out_shape: &[usize],
                         options: Option<&OptionMap>) -> Result<OpSetKind> {
        use GathersDesc::*;
        match self {
            Builtin(s) => {
                match s.as_ref() {
                    "identity" | "reshape" => {
                        let out_prod: usize = out_shape.iter().product();
                        let in_prod: usize = in_shape.iter().product();
                        if out_prod != in_prod {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
                        }
                        identity(out_shape)
                    },
                    "transpose" => {
                        if out_shape.iter().rev().zip(in_shape.iter()).any(|(a, b)| a != b) {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
                        }
                        transpose(out_shape, in_shape)
                    },
                    "load_rep" => {
                        load_rep(in_shape, out_shape)
                    },
                    "load_trunc" => {
                        load_trunc(in_shape, out_shape)
                    },
                    "load_grid_2d" => {
                        load_grid_2d(in_shape, out_shape)
                    }
                    "broadcast" => {
                        let group = options
                            .and_then(|m| m.get("group"))
                            .and_then(|o| o.get(0).copied()).unwrap_or(0);
                        broadcast(in_shape, out_shape, group as usize)
                    },
                    "rots_no_group" | "rots" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        if let Some(m) = options {
                            let main_idx = m.get("main").and_then(|v| v.get(0).copied())
                                .ok_or_else(|| ErrorKind::MissingOption("main".to_string()))?
                                as usize;
                            let second_idx = m.get("second").and_then(|v| v.get(0).copied())
                                .ok_or_else(|| ErrorKind::MissingOption("second".to_string()))?
                                as usize;
                            let out_idx = m.get("out").and_then(|v| v.get(0).copied())
                                .ok_or_else(|| ErrorKind::MissingOption("out".to_string()))?
                                as usize;
                            if s == "rots_no_group" {
                                simple_rotations(out_shape, main_idx, second_idx, out_idx)
                            }
                            else {
                                all_rotations(out_shape, main_idx, second_idx, out_idx)
                            }
                        }
                        else {
                            Err(ErrorKind::MissingOption("main".to_string()).into())
                        }
                    },
                   "xforms_no_group" | "xforms" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        if let Some(m) = options {
                            let main_idx = m.get("main").and_then(|v| v.get(0).copied())
                                .ok_or_else(|| ErrorKind::MissingOption("main".to_string()))?
                                as usize;
                            let second_idx = m.get("second").and_then(|v| v.get(0).copied())
                                .ok_or_else(|| ErrorKind::MissingOption("second".to_string()))?
                                as usize;
                            let out_idx = m.get("out").and_then(|v| v.get(0).copied())
                                .ok_or_else(|| ErrorKind::MissingOption("out".to_string()))?
                                as usize;
                            if s == "rots_no_group" {
                                simple_xforms(out_shape, main_idx, second_idx, out_idx)
                            }
                            else {
                                all_xforms(out_shape, main_idx, second_idx, out_idx)
                            }
                        }
                        else {
                            Err(ErrorKind::MissingOption("main".to_string()).into())
                        }
                    },
                    "row_rots_no_group" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        // This is a compatibility alias
                        simple_rotations(out_shape, 1, 0, 1)
                    },
                    "col_rots_no_group" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        simple_rotations(out_shape, 0, 1, 0)
                    },
                    "row_xforms_no_group" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        simple_xforms(out_shape, 1, 0, 1)
                    },
                    "col_xforms_no_group" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        simple_xforms(out_shape, 0, 1, 0)
                    },
                    "row_rots" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        all_rotations(out_shape, 1, 0, 1)
                    },
                    "col_rots" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        all_rotations(out_shape, 0, 1, 0)
                    },
                    "row_xforms" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        all_xforms(out_shape, 1, 0, 1)
                    },
                    "col_xforms" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        all_xforms(out_shape, 0, 1, 0)
                    },
                    "reg_select_no_consts" | "reg_select" => {
                        if out_shape[0..out_shape.len()-1] != in_shape[0..in_shape.len()-1] {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
                        }
                        if in_shape[in_shape.len()-1] != 2 {
                            let mut correct_shape = in_shape.to_vec();
                            correct_shape[in_shape.len()-1] = 2;
                            return Err(ErrorKind::ShapeMismatch(correct_shape, in_shape.to_vec()).into());
                        }
                        let n = out_shape[0] as isize;
                        let default_consts = [0, 1, -1, n, -n];
                        let consts = if s == "reg_select_no_consts" {
                            &[0]
                        } else {
                            options.and_then(|m| m.get("consts").map(|v| v.as_slice()))
                                .unwrap_or(&default_consts)
                        };
                        reg_select(out_shape, consts)
                    },
                    "cond_keep_no_consts" | "cond_keep" => {
                        if out_shape != in_shape {
                            return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                        }
                        let n = out_shape[0] as isize;
                        let default_consts = [0, 1, -1, n, -n];
                        let consts = if s == "cond_keep_no_consts" {
                            &[0]
                        } else {
                            options.and_then(|m| m.get("consts").map(|v| v.as_slice()))
                                .unwrap_or(&default_consts)
                        };
                        let mut restrict = std::collections::BTreeMap::new();
                        if let Some(v) = options.and_then(|m| m.get("restrict")) {
                            for s in v.chunks(2) {
                                restrict.insert(s[0] as usize, s[1] as usize);
                            }
                        }
                        cond_keep(out_shape, consts, &restrict)
                    },
                    "general_select_no_consts" | "general_select" => {
                        let axis = options.and_then(|m| m.get("axis"))
                            .and_then(|v| v.get(0).copied())
                            .ok_or_else(|| ErrorKind::MissingOption("axis".to_string()))?
                            as usize;
                        let n = out_shape[axis] as isize;
                        let default_consts = [0, 1, -1, n, -n];
                        let consts = if s == "general_select_no_consts" {
                            &[0]
                        } else {
                            options.and_then(|m| m.get("consts").map(|v| v.as_slice()))
                                .unwrap_or(&default_consts)
                        };
                        let dims: Vec<usize> =
                            options.and_then(|m| m.get("dims")
                                             .map(|v| v.iter().copied()
                                                  .map(|i| i as usize).collect()))
                            .unwrap_or_else(|| (0..out_shape.len()).collect());
                        general_select(out_shape, in_shape, axis,
                                       consts, &dims)
                    },
                    other => {
                        return Err(ErrorKind::UnknownBasisType(other.to_owned()).into())
                    }
                }
            },
            Custom(gathers) => {
                gathers.iter().map(
                    |(name, gather)| {
                        let gather: Vec<Ix> = gather.iter().copied().map(|d| d as Ix).collect();
                        let array = gather.chunks(in_shape.len())
                            .map(|s| to_opt_ix(s, in_shape))
                            .collect();
                        let array = ArrayD::from_shape_vec(out_shape.as_ref(), array)
                            .chain_err(|| ErrorKind::InvalidArrayData(out_shape.to_vec()))?;
                        Ok(Gather::new_raw(array, name.clone()))
                    }).collect::<Result<Vec<_>>>().map(|r| r.into())
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum InitialDesc {
    From(Symbolic),
    Data(Vec<Symbolic>),
}

impl InitialDesc {
    pub fn to_progstate<'d>(&self, domain: &'d Domain, shape: &[Ix],
                            name: &Option<String>) -> Result<ProgState<'d>> {
        let array =
            match self {
                InitialDesc::From(offset) => {
                    let shape_len: usize = shape.iter().copied().product();
                    let shape_len = shape_len as u16;
                    let array: Vec<Value> = (*offset..offset+shape_len)
                        .map(Value::Symbol).collect();
                    ArrayD::from_shape_vec(shape, array)
                        .chain_err(|| ErrorKind::ShapeMismatch(shape.to_owned(),
                                                               vec![*offset as usize]))
                },
                InitialDesc::Data(data) => {
                    let array: Vec<Value> = data.iter().copied().map(Value::Symbol).collect();
                    ArrayD::from_shape_vec(shape, array)
                        .chain_err(|| ErrorKind::InvalidArrayData(shape.to_owned()))
                }
            };
        let array = array?;
        let name =
            match name {
                Some(n) => n.to_owned(),
                None => match self {
                    InitialDesc::From(_) => "init".to_owned(),
                    InitialDesc::Data(_) => "custom_start".to_owned(),
                }
            };
        ProgState::new_from_spec(domain, array, name)
            .ok_or_else(|| ErrorKind::SymbolsNotInSpec.into())
    }
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum SynthesisLevelKind {
    Gather(GathersDesc),
    Merge(Vec<u64>),
    Split(Vec<u64>),
    Initial(InitialDesc),
}

impl SynthesisLevelKind {
    pub fn is_initial(&self) -> bool {
        use SynthesisLevelKind::*;
        match self {
            Gather(_) => false,
            Merge(_) => false,
            Split(_) => false,
            Initial(_) => true,
        }
    }

    pub fn to_opset_kind(&self, in_shape: &[usize], out_shape: &[usize],
                         options: Option<&OptionMap>,
                         lane: usize) -> Result<OpSetKind> {
        use SynthesisLevelKind::*;
        match self {
            Gather(desc) => desc.to_opset_kind(in_shape, out_shape, options),
            Merge(from) => Ok(OpSetKind::Merge(from.iter().copied()
                                               .map(|x| x as usize).collect(),
                                               lane)),
            Split(to) => Ok(OpSetKind::Split(lane,
                                             to.iter().copied()
                                             .map(|x| x as usize).collect())),
            Initial(_) => panic!("Shouldn't be called with initial inputs")
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SynthesisLevelDesc {
    pub name: Option<String>,
    pub lane: u64,
    pub step: SynthesisLevelKind,
    pub out_shape: Vec<u64>,
    pub options: Option<HashMap<String, Vec<i64>>>,
    pub prune: bool,
    pub then_fold: bool,
}

impl SynthesisLevelDesc {
    pub fn is_initial(&self) -> bool {
        self.step.is_initial()
    }

    pub fn initial_desc(&self) -> Option<&InitialDesc> {
        match self.step {
            SynthesisLevelKind::Initial(ref d) => Some(d),
            _ => None
        }
    }

    fn to_opset(&self, in_shapes: &[Option<Vec<usize>>])
                -> Result<OpSet> {
        let lane = self.lane as usize;
        let mut out_shape: ShapeVec = self.out_shape.iter().copied().map(|x| x as usize).collect();
        let options: Option<OptionMap> = self.options.as_ref()
            .map(|o| o.iter().map(
                |(k, v)| (k.clone(), v.iter().copied().map(
                    |x| x as isize).collect())).collect());
        let in_shape: ShapeVec = if self.step.is_initial() {
            panic!("You shouldn't've called this with an initial state")
        } else if let Some(Some(s)) = in_shapes.get(lane) {
            ShapeVec::from_slice(s.as_ref())
        } else {
            return Err(ErrorKind::MissingShape(lane).into());
        };

        let opset = self.step.to_opset_kind(in_shape.as_slice(),
                                            out_shape.as_slice(),
                                            options.as_ref(),
                                            lane)?;

        // Post-processing
        match &opset {
            OpSetKind::Gathers(_) => {
                if self.then_fold {
                    out_shape.pop();
                }
            },
            OpSetKind::Merge(from, _to) => {
                for idx in from.iter().copied() {
                    if let Some(Some(s)) = in_shapes.get(idx) {
                        if self.then_fold && s.as_slice() != out_shape.as_ref() {
                            return Err(ErrorKind::ShapeMismatch(s.to_owned(),
                                                                out_shape.to_vec()).into());
                        }
                        else if !self.then_fold && s.as_slice() != &out_shape.as_slice()[0..out_shape.len()-1] {
                            return Err(ErrorKind::ShapeMismatch(s.to_owned(),
                                                                out_shape.to_vec()).into());
                        }
                    }
                    else {
                        return Err(ErrorKind::MissingShape(idx).into())
                    }
                }
                if !self.then_fold && out_shape[out_shape.len()-1] != from.len() {
                    return Err(ErrorKind::WrongMergeArgs(out_shape[out_shape.len()-1],
                                                         from.len()).into());
                }
            }
            OpSetKind::Split(_, _) => {
                if self.then_fold {
                    return Err(ErrorKind::NoSplitFolds.into());
                }
                if self.prune {
                    return Err(ErrorKind::NoSplitPrune.into());
                }
            }
        }
        let mut name: std::borrow::Cow<'static, str> =
            match &self.name {
                Some(n) => n.to_owned().into(),
                None =>
                    match &self.step {
                        SynthesisLevelKind::Gather(GathersDesc::Builtin(s)) =>
                            s.to_owned().into(),
                        SynthesisLevelKind::Gather(_) => "custom_gathers".into(),
                        SynthesisLevelKind::Merge(_) =>
                            if self.then_fold {
                                "merge_folding".into()
                            } else {
                                "merge".into()
                            },
                        SynthesisLevelKind::Split(_) => "split".into(),
                        SynthesisLevelKind::Initial(_) => "inital??".into(),
                    }
            };
        if let Some(m) = options {
            // These are conveniently in sorted order, so we can fix our name length issues
            let mut temp = name.into_owned();
            temp.push('{');
            for (k, v) in &m {
                temp.push_str(k);
                temp.push('[');
                for i in v {
                    temp.push_str(&i.to_string());
                    temp.push('|');
                }
                temp.pop();
                temp.push_str("]");
            }
            temp.push('}');
            name = temp.into();
        }
        Ok(OpSet::new(name, opset, in_shape.clone(), out_shape, self.then_fold))
    }

    pub fn to_synthesis_level(&self, shapes: &mut Vec<Option<Vec<usize>>>,
                              expected_syms_idxs: &mut Vec<Option<usize>>,
                              count: &mut usize) -> Result<SynthesisLevel> {
        let opset = self.to_opset(&shapes)?;
        let lane = self.lane as usize;
        update_expected_syms_idxs(&opset, lane, expected_syms_idxs, count);
        let expected_syms = expected_syms_idxs[lane].unwrap();
        let ret = SynthesisLevel::new(opset, lane, expected_syms, self.prune);
        update_shape_info(&ret, shapes);
        Ok(ret)
    }
}

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
    use itertools::iproduct;
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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum TargetDesc {
    Builtin(String),
    Custom { data: Vec<Symbolic>, n_folds: u64 }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProblemDesc {
    pub target: TargetDesc,
    pub target_info: Vec<u64>,
    pub steps: Vec<SynthesisLevelDesc>,
}

fn custom_target(data: &[Symbolic], shape: &[usize], n_folds: usize) -> Result<ArrayD<Value>> {
    let data: Vec<Value> = data.iter().copied().map(Value::Symbol).collect();
    let mut array = ArrayD::from_shape_vec(shape, data)
        .chain_err(|| ErrorKind::InvalidArrayData(shape.to_owned()))?;
    for _ in 0..n_folds {
        array = array.map_axis(ndarray::Axis(array.ndim() - 1),
                               |data| Value::fold(data.to_vec()));
    }
    Ok(array)
}


fn update_shape_info(level: &SynthesisLevel, shapes: &mut Vec<Option<Vec<usize>>>) {
    let lane = level.lane;

    match &level.ops.ops {
        OpSetKind::Gathers(_) => {
            extending_set(shapes, lane, Some(level.ops.out_shape.to_vec()));
        },
        OpSetKind::Merge(from, to) => {
            for idx in from {
                shapes[*idx] = None;
            }
            extending_set(shapes, *to, Some(level.ops.out_shape.to_vec()));
        }
        OpSetKind::Split(from, to) => {
            shapes[*from] = None;
            for idx in to {
                extending_set(shapes, *idx, Some(level.ops.out_shape.to_vec()));
            }
        }
    }
}

// Note: levels take expected syms indices of their immediate successor
// because pruning happens at the end of a level
fn update_expected_syms_idxs(ops: &OpSet, lane: usize,
                             expected_syms_idxs: &mut Vec<Option<usize>>,
                             count: &mut usize) {
    use OpSetKind::*;
    match ops.ops {
        Gathers(_) => {
            if ops.fused_fold {
                expected_syms_idxs[lane] = Some(*count);
                *count += 1;
            }
        }
        Merge(ref from, to) => {
            for idx in from.iter().copied() {
                expected_syms_idxs[idx] = None;
            }
            expected_syms_idxs[to] = Some(*count);
            *count += 1;
        }
        Split(from, ref to) => {
            let idx = expected_syms_idxs[from].take();
            for i in to.iter().copied() {
                extending_set(expected_syms_idxs, i, idx);
            }
        }
    }
}

fn update_expected_syms(level: &SynthesisLevel, domain: &Domain,
                        expected_syms_idxs: &mut Vec<Option<usize>>,
                        expected_syms_sets: &mut Vec<BTreeSet<DomRef>>) {
    use OpSetKind::*;
    let lane = level.lane;
    match level.ops.ops {
        Gathers(_) => {
            if level.ops.fused_fold {
                let new_set = fold_expected(domain,
                                            &expected_syms_sets[
                                                expected_syms_idxs[lane].unwrap()]);
                expected_syms_idxs[lane] = Some(expected_syms_sets.len());
                expected_syms_sets.push(new_set);
            }
        }
        Merge(ref from, to) => {
            let mut merge_set = BTreeSet::<DomRef>::new();
            for idx in from.iter().copied() {
                match expected_syms_idxs[idx].take() {
                    Some(v) => merge_set.extend(expected_syms_sets[v].iter()),
                    None => ()
                }
            }
            if level.ops.fused_fold {
                merge_set = fold_expected(domain, &merge_set);
            }
            expected_syms_idxs[to] = Some(expected_syms_sets.len());
            expected_syms_sets.push(merge_set);
        }
        Split(from, ref to) => {
            let idx = expected_syms_idxs[from].take();
            for i in to.iter().copied() {
                extending_set(expected_syms_idxs, i, idx);
            }
        }
    }
}

impl ProblemDesc {
    pub fn get_spec(&self) -> Result<ArrayD<Value>> {
        let target_info = self.target_info.iter().copied().map(|x| x as usize)
            .collect::<SmallVec<[usize; 4]>>();
        match &self.target {
            TargetDesc::Builtin(name) =>
                match name.as_str() {
                    "trove" => {
                        match target_info.as_slice() {
                            &[m, n] => Ok(trove(m, n)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                        }
                    }
                    "trove_sum" => {
                        match target_info.as_slice() {
                            &[m, n] => Ok(trove_sum(m, n)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                        }
                    }
                    "conv_dealg" => {
                        match target_info.as_slice() {
                            &[width, k] => Ok(convolve_dealg(width, k)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                        }
                    }
                    "conv" => {
                        match target_info.as_slice() {
                            &[width, k] => Ok(convolve(width, k)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                        }

                    }
                    "weight_conv" => {
                        match target_info.as_slice() {
                            &[width, k] => Ok(weighted_convolve(width, k)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                        }

                    }
                    "stencil_2d" => {
                        match target_info.as_slice() {
                            &[width, k] => Ok(stencil_2d(width, k)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                        }

                    }
                    "poly_mult" => {
                        match target_info.as_slice() {
                            &[n] => Ok(poly_mult(n)),
                            other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 1).into())
                        }
                    }
                    other => Err(ErrorKind::UnknownProblem(other.to_owned()).into())
                },
                TargetDesc::Custom { data, n_folds } => {
                    let n_folds = *n_folds as usize;
                    custom_target(data, &target_info, n_folds)
                }
            }
    }

    pub fn make_domain(&self, spec: ndarray::ArrayViewD<Value>) -> Domain {
        Domain::new(spec)
    }

    pub fn get_levels(&self) -> Result<Vec<SynthesisLevel>> {
        let mut ret = Vec::<SynthesisLevel>::with_capacity(self.steps.len());
        let mut shapes: Vec<Option<Vec<usize>>> = Vec::new();
        let mut expected_syms_idxs = Vec::<Option<usize>>::new();
        let mut expected_syms_count = 0;

        for step in &self.steps {
            let lane = step.lane as usize;
            if step.is_initial() {
                let shape: Vec<usize> = step.out_shape.iter().copied()
                    .map(|x| x as usize).collect();

                extending_set(&mut shapes, lane, Some(shape));
                extending_set(&mut expected_syms_idxs, lane, Some(expected_syms_count));
                expected_syms_count += 1;
            }
            else {
                let level = step.to_synthesis_level(&mut shapes,
                                                    &mut expected_syms_idxs,
                                                    &mut expected_syms_count)
                    .chain_err(|| ErrorKind::LevelBuild(Box::new(step.clone())))?;
                ret.push(level);
            }
        }
        // Error checking
        {
            let last_idx = ret.len() - 1;
            let last = &mut ret[last_idx];
            if last.prune {
                println!("WARNING: It doesn't make sense to prune after the last instructions are applied. Suppressing.");
                last.prune = false;
            }
            if last.lane != 0 {
                return Err(ErrorKind::FinalLane(last.lane).into());
            }
        }
        Ok(ret)
    }

    pub fn build_problem<'d>(&self,
                             domain: &'d Domain,
                             levels: &[SynthesisLevel],
                             spec: ArrayD<Value>)
                             -> Result<(Vec<Option<ProgState<'d>>>,
                                        ProgState<'d>,
                                        Vec<Vec<DomRef>>)>
    {
        let mut initials = Vec::<Option<ProgState<'d>>>::new();
        let mut expected_syms_idxs = Vec::<Option<usize>>::new();
        let mut expected_syms_sets = Vec::<BTreeSet<DomRef>>::new();

        for (idx, step) in self.steps.iter().enumerate() {
            let lane = step.lane as usize;
            if step.is_initial() {
                let shape: Vec<usize> = step.out_shape.iter().copied()
                    .map(|x| x as usize).collect();
                let state = step.initial_desc().unwrap()
                     .to_progstate(domain, &shape, &step.name)
                    .chain_err(|| ErrorKind::LevelBuild(Box::new(step.clone())))?;
                let symbols = crate::expected_syms_util::expcted_of_inital_state(&state);

                extending_set(&mut initials, lane, Some(state));
                extending_set(&mut expected_syms_idxs, lane, Some(expected_syms_sets.len()));
                expected_syms_sets.push(symbols);
            }
            else {
                let level = &levels[idx - initials.len()]; // Initial steps aren't synthesis levels
                update_expected_syms(&level, domain, &mut expected_syms_idxs,
                                     &mut expected_syms_sets);
            }
        }

        let target_name: std::borrow::Cow<'static, str> =
            match &self.target {
                TargetDesc::Builtin(s) => s.clone().into(),
                TargetDesc::Custom { data: _d, n_folds: _n } => "custom_target".into(),
            };
        let spec = ProgState::new_from_spec(domain, spec,
                                            target_name).unwrap();
        let expected_syms = expected_syms_sets.into_iter()
            .map(|s| s.into_iter().collect()).collect();
        Ok((initials, spec, expected_syms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Value};

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

    #[test]
    pub fn can_construct() {
        use crate::operators::swizzle;
        use smallvec::smallvec;
        use std::collections::HashSet;

        let desc = ProblemDesc {
            target: TargetDesc::Builtin("trove".to_owned()),
            target_info: vec![3, 4],
            steps: vec![
                SynthesisLevelDesc { step: SynthesisLevelKind::Initial(InitialDesc::From(0)),
                                     out_shape: vec![3, 4], lane: 0, name: None, options: None,
                                     prune: false, then_fold: false},
                SynthesisLevelDesc { step: SynthesisLevelKind::Gather(GathersDesc::Builtin("row_rots_no_group".to_owned())),
                                     out_shape: vec![3, 4], lane: 0, name: None, options: None,
                                     prune: false, then_fold: false},
            ]
        };
        let spec = desc.get_spec().unwrap();
        let domain = desc.make_domain(spec.view());
        let trove_state = ProgState::new_from_spec(&domain, trove(3, 4), "trove").unwrap();
        let levels = desc.get_levels().unwrap();
        let (start, end, expected_syms)
            = desc.build_problem(&domain, &levels, spec).unwrap();
        assert_eq!(start, vec![Some(
            crate::state::ProgState::linear(&domain, 1, &[3, 4]))]);
        assert_eq!(end, trove_state);
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].prune, false);
        assert!(levels[0].matrix.is_none());
        assert_eq!(expected_syms.len(), 1);
        assert_eq!(expected_syms[0].len(), 12);
        let ops = &levels[0].ops;
        let trove_shape: ShapeVec = smallvec![3, 4];
        assert_eq!(ops.in_shape, trove_shape);
        assert_eq!(ops.out_shape, trove_shape);
        assert_eq!(ops.ops.gathers().unwrap().iter().collect::<HashSet<_>>(),
                   swizzle::simple_rotations(&[3, 4], 1, 0, 1).unwrap()
                   .gathers().unwrap()
                   .iter().collect::<HashSet<_>>());
    }
}
