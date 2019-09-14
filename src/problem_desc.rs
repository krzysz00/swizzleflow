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
use crate::operators::SynthesisLevel;
use crate::state::{ProgState,Domain,Value,Symbolic};
use crate::operators::swizzle::{simple_fans, simple_rotations, OpAxis};
use crate::operators::reg_select::{reg_select_no_const};
use crate::operators::load::{load_rep,load_trunc};
use crate::errors::*;
use crate::misc::ShapeVec;

use serde::{Serialize, Deserialize};

use ndarray::{Array, ArrayD, Ix};

use smallvec::SmallVec;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SynthesisLevelDesc {
    pub basis: String,
    pub in_sizes: Vec<u64>,
    pub out_sizes: Vec<u64>,
    pub prune: bool,
    pub then_fold: bool,
}

impl SynthesisLevelDesc {
    pub fn to_synthesis_level(&self) -> Result<SynthesisLevel> {
        let in_shape: ShapeVec = self.in_sizes.iter().map(|x| *x as usize).collect();
        let out_shape: ShapeVec = self.out_sizes.iter().map(|x| *x as usize).collect();

        let mut ops =
            match self.basis.as_ref() {
                "load_rep" => {
                    load_rep(&in_shape, &out_shape)?
                },
                "load_trunc" => {
                    load_trunc(&in_shape, &out_shape)?
                },
                "sRr" => {
                    if out_shape != in_shape {
                        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                    }
                    simple_rotations(&out_shape, OpAxis::Rows)?
                }
                "sRc" => {
                    if out_shape != in_shape {
                        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                    }
                    simple_rotations(&out_shape, OpAxis::Columns)?
                }
                "sFr" => {
                    if out_shape != in_shape {
                        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                    }
                    simple_fans(&out_shape, OpAxis::Rows)?
                }
                "sFc" => {
                    if out_shape != in_shape {
                        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into())
                    }
                    simple_fans(&out_shape, OpAxis::Columns)?
                }
                "regSelNC" => {
                    if out_shape[0..out_shape.len()-1] != in_shape[0..in_shape.len()-1] {
                        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
                    }
                    if in_shape[in_shape.len()-1] != 2 {
                        let mut correct_shape = in_shape.to_vec();
                        correct_shape[in_shape.len()-1] = 2;
                        return Err(ErrorKind::ShapeMismatch(correct_shape, in_shape.to_vec()).into());
                    }
                    reg_select_no_const(&out_shape)?
                }
                other => {
                    return Err(ErrorKind::UnknownBasisType(other.to_owned()).into())
                }
            };
        // Handle operations that don't natively fold
        if self.then_fold {
            ops.add_fused_fold();
        }
        Ok(SynthesisLevel::new(ops, self.prune))
    }
}

pub fn trove(m: Ix, n: Ix) -> ArrayD<Value> {
    Array::from_shape_fn((m, n),
                         move |(i, j)|
                         Value::Symbol((j + i * n) as Symbolic))
        .into_dyn()
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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProblemDesc {
    pub end_name: String,
    pub end_info: Vec<u64>,
    pub steps: Vec<SynthesisLevelDesc>,
}

impl ProblemDesc {
    pub fn get_spec(&self) -> Result<ArrayD<Value>> {
        let end_info = self.end_info.iter().copied().map(|x| x as usize)
            .collect::<SmallVec<[usize; 4]>>();
        match self.end_name.as_str() {
            "trove" => {
                match end_info.as_slice() {
                    &[m, n] => Ok(trove( m, n)),
                    other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                }
            }
            "conv_dealg" => {
                match end_info.as_slice() {
                    &[width, k] => Ok(convolve_dealg(width, k)),
                    other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                }
            }
            "conv" => {
                match end_info.as_slice() {
                    &[width, k] => Ok(convolve(width, k)),
                    other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
                }

            }
            other => Err(ErrorKind::UnknownProblem(other.to_owned()).into())
        }
    }

    pub fn make_domain(&self, spec: ndarray::ArrayViewD<Value>) -> Domain {
        Domain::new(spec)
    }

    pub fn to_problem<'d>(&self,
                          domain: &'d Domain,
                          spec: ArrayD<Value>) -> Result<(ProgState<'d>,
                                                          ProgState<'d>,
                                                         Vec<SynthesisLevel>)> {
        let levels: Result<Vec<SynthesisLevel>> =
            self.steps.iter()
            .map(|x| x.to_synthesis_level().chain_err(|| ErrorKind::LevelBuild(Box::new(x.clone()))))
            .collect();
        let levels = levels?;

        let start_shape = levels[0].ops.in_shape.as_slice();
        let start_symbols: usize = start_shape.iter().product();
        if start_symbols != domain.num_symbols() {
            return Err(ErrorKind::ShapeMismatch(
                levels[0].ops.in_shape.to_vec(),
                vec![domain.num_symbols()]).into());
        }

        let initial = ProgState::linear(domain, start_shape);
        // All the values in the spec had better be in the domain
        let spec = ProgState::new_from_spec(domain, spec, &self.end_name).unwrap();
        Ok((initial, spec, levels))
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
            end_name: "trove".to_owned(),
            end_info: vec![3, 4],
            steps: vec![
                SynthesisLevelDesc { basis: "sRr".to_owned(),
                                     in_sizes: vec![3, 4], out_sizes: vec![3, 4],
                                     prune: false, then_fold: false},
            ]
        };
        let spec = desc.get_spec().unwrap();
        let domain = desc.make_domain(spec.view());
        let trove_state = ProgState::new_from_spec(&domain, trove(3, 4), "trove").unwrap();
        let (start, end, levels) = desc.to_problem(&domain, spec).unwrap();
        assert_eq!(start, crate::state::ProgState::linear(&domain, &[3, 4]));
        assert_eq!(end, trove_state);
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].prune, false);
        assert!(levels[0].matrix.is_none());
        let ops = &levels[0].ops;
        let trove_shape: ShapeVec = smallvec![3, 4];
        assert_eq!(ops.in_shape, trove_shape);
        assert_eq!(ops.out_shape, trove_shape);
        assert_eq!(ops.ops.iter().collect::<HashSet<_>>(),
                   swizzle::simple_rotations(&[3, 4], swizzle::OpAxis::Rows).unwrap().ops
                   .iter().collect::<HashSet<_>>());
    }
}
