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
                other => {
                    return Err(ErrorKind::UnknownBasisType(other.to_owned()).into())
                }
            };
        Ok(SynthesisLevel::new(ops, self.prune))
    }
}

pub fn trove(m: Ix, n: Ix) -> ProgState {
    let array = Array::from_shape_fn((m, n),
                                     move |(i, j)| (i + j * m) as Symbolic)
        .into_dyn();
    ProgState::new((m * n) as Symbolic, array, "trove")
}

pub fn lookup_problem(problem: &str, shape: &[Ix]) -> Result<ProgState> {
    match problem {
        "trove" => {
            match shape {
                &[m, n] => Ok(trove(m, n)),
                other => Err(ErrorKind::InvalidShapeDim(other.to_owned(), 2).into())
            }
        }
        "id" | "linear" => Ok(ProgState::linear(shape)),
        other => Err(ErrorKind::UnknownProblem(other.to_owned()).into())
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ProblemDesc {
    pub start_name: String,
    pub end_name: String,
    pub steps: Vec<SynthesisLevelDesc>,
}

impl ProblemDesc {
    pub fn to_problem(&self) -> Result<(ProgState, ProgState, Vec<SynthesisLevel>)> {
        let levels: Result<Vec<SynthesisLevel>> = self.steps.iter().map(|x| x.to_synthesis_level()).collect();
        let levels = levels?;
        let start_shape = levels[0].ops.in_shape.as_slice();
        let end_shape = levels[levels.len() - 1].ops.out_shape.as_slice();
        let initial = lookup_problem(&self.start_name, start_shape)?;
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
    pub fn can_construct() {
        use crate::operators::swizzle;
        use smallvec::smallvec;
        use std::collections::HashSet;

        let desc = ProblemDesc {
            start_name: "linear".to_owned(),
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
