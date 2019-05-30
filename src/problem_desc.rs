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
use crate::state::{ProgState,Symbolic};
use crate::operators::swizzle::{simple_fans, simple_rotations, OpAxis};
use crate::operators::reg_select::{reg_select_no_const};
use crate::operators::load::{load_rep,load_trunc};
use crate::errors::*;
use crate::misc::ShapeVec;

use serde::{Serialize, Deserialize};

use ndarray::{Array, Ix};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SynthesisLevelDesc {
    pub basis: String,
    pub in_sizes: Vec<u64>,
    pub out_sizes: Vec<u64>,
    pub prune: bool,
}

impl SynthesisLevelDesc {
    pub fn to_synthesis_level(&self) -> Result<SynthesisLevel> {
        let in_shape: ShapeVec = self.in_sizes.iter().map(|x| *x as usize).collect();
        let out_shape: ShapeVec = self.out_sizes.iter().map(|x| *x as usize).collect();

        let ops =
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
                    if &out_shape[0..out_shape.len()-1] != &in_shape[0..in_shape.len()-1] {
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
        Ok(SynthesisLevel::new(ops, self.prune))
    }
}

pub fn trove(m: Ix, n: Ix) -> ProgState {
    let array = Array::from_shape_fn((m, n),
                                     move |(i, j)| (j + i * n) as Symbolic)
        .into_dyn();
    ProgState::new((m * n) as Symbolic, array, "trove")
}

fn convolve_dealg(width: Ix, k: Ix) -> ProgState {
    let array = Array::from_shape_fn((width, k),
                                     move |(i, j)| (i + j)  as Symbolic)
        .into_dyn();
    ProgState::new((width + k - 1) as Symbolic, array, "conv_dealg")
}

pub fn lookup_problem(problem: &str, shape: &[Ix]) -> Result<ProgState> {
    match problem {
        "trove" => {
            match shape {
                &[m, n] => Ok(trove(m, n)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
            }
        }
        "conv_dealg" => {
            match shape {
                &[width, k] => Ok(convolve_dealg(width, k)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
            }
        }
        other => Err(ErrorKind::UnknownProblem(other.to_owned()).into())
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProblemDesc {
    pub end_name: String,
    pub steps: Vec<SynthesisLevelDesc>,
}

impl ProblemDesc {
    pub fn to_problem(&self) -> Result<(ProgState, ProgState, Vec<SynthesisLevel>)> {
        let levels: Result<Vec<SynthesisLevel>> = self.steps.iter().map(|x| x.to_synthesis_level()).collect();
        let levels = levels?;

        let start_shape = levels[0].ops.in_shape.as_slice();
        let end_shape = levels[levels.len() - 1].ops.out_shape.as_slice();

        let initial = ProgState::linear(start_shape);
        let spec = lookup_problem(&self.end_name, end_shape)?;
        Ok((initial, spec, levels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ProgValue;

    #[test]
    pub fn trove_works() {
        let trove_spec: [ProgValue; 12] = [0, 3, 6, 9,
                                          1, 4, 7, 10,
                                          2, 5, 8, 11];
        let trove_spec_arr = Array::from_shape_vec((3, 4), (&trove_spec).to_vec()).unwrap().into_dyn();
        let trove_spec_prog = ProgState::new(12, trove_spec_arr, "trove");
        assert_eq!(trove_spec_prog, trove(3, 4));
    }

    #[test]
    pub fn conv_1d_end_works() {
        let conv_final: [ProgValue; 4 * 3] = [0, 1, 2,
                                              1, 2, 3,
                                              2, 3, 4,
                                              3, 4, 5];
        let conv_final_arr = Array::from_shape_vec((4, 3), (&conv_final).to_vec()).unwrap().into_dyn();
        let conv_final_prog = ProgState::new(6, conv_final_arr, "conv_regs_loaded");
        assert_eq!(conv_final_prog, convolve_dealg(4, 3));
    }

    #[test]
    pub fn can_construct() {
        use crate::operators::swizzle;
        use smallvec::smallvec;
        use std::collections::HashSet;

        let desc = ProblemDesc {
            end_name: "trove".to_owned(),
            steps: vec![
                SynthesisLevelDesc { basis: "sRr".to_owned(),
                                     in_sizes: vec![3, 4], out_sizes: vec![3, 4],
                                     prune: false},
            ]
        };
        let (start, end, levels) = desc.to_problem().unwrap();
        assert_eq!(start, crate::state::ProgState::linear(&[3, 4]));
        assert_eq!(end, trove(3, 4));
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
