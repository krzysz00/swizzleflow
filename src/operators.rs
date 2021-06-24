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
pub mod swizzle;
pub mod select;
pub mod load;
pub mod permutations;
pub mod hvx;

use crate::errors::*;

use crate::state::{Gather, to_opt_ix};
use crate::misc::{ShapeVec};

use ndarray::Ix;


    // pub fn to_name(&self) -> String {
    //     let in_strings: SmallVec<[String; 4]> = self.in_shape.iter().map(|v| v.to_string()).collect();
    //     let out_strings: SmallVec<[String; 4]> = self.out_shape.iter().map(|v| v.to_string()).collect();
    //     format!("{}-{}-{}", out_strings.join(","), self.name, in_strings.join(","))
    // }

pub fn identity_gather(shape: &[Ix]) -> Gather {
    Gather::new(shape, |idxs| (0, to_opt_ix(idxs, shape)), "id")
}

pub fn transpose_gather(in_shape: &[Ix], out_shape: &[Ix]) -> Gather {
    Gather::new(out_shape, |idxs| {
        let mut ret = 0;
        for (idx, scale) in idxs.iter().rev().zip(in_shape.iter()) {
            ret = ret * scale + idx;
        }
        (0, ret as isize)
    }, "tr")
}

pub fn stack_gather(in_shapes: &[ShapeVec], out_shape: &[Ix]) -> Gather {
    let last = out_shape.len()-1;
    Gather::new(out_shape, |idxs| {
        (idxs[last], to_opt_ix(&idxs[0..last], in_shapes[idxs[last]].as_slice()))
    },
                "stack")
}

pub fn identity(in_shapes: &[ShapeVec], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }

    let in_shape: &[usize] = in_shapes[0].as_slice();
    let out_prod: usize = out_shape.iter().product();
    let in_prod: usize = in_shape.iter().product();
    if out_prod != in_prod {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }

    Ok(vec![identity_gather(out_shape)])
}

pub fn transpose(in_shapes: &[ShapeVec], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }
    let in_shape = in_shapes[0].as_slice();
    if out_shape.iter().rev().zip(in_shape.iter()).any(|(a, b)| a != b) {
        return Err(ErrorKind::ShapeMismatch(in_shape.to_vec(), out_shape.to_vec()).into());
    }

    Ok(vec![transpose_gather(in_shape, out_shape)])
}

pub fn stack(in_shapes: &[ShapeVec], out_shape: &[Ix]) -> Result<Vec<Gather>> {
    for s in in_shapes {
        if s.as_slice() != &out_shape[0..out_shape.len()-1] {
            return Err(ErrorKind::ShapeMismatch(out_shape.to_vec(), s.to_vec()).into());
        }
    }
    if in_shapes.len() != out_shape[out_shape.len()-1] {
        // TODO, better error message
        return Err(ErrorKind::ShapeMismatch(vec![out_shape[out_shape.len()-1]],
                                            vec![in_shapes.len()]).into());
    }
    Ok(vec![stack_gather(in_shapes, out_shape)])
}

pub fn rot_idx_r(in_shapes: &[ShapeVec],
                 out_shape: &[Ix], rot: Ix) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }
    let in_shape = in_shapes[0].as_slice();
    let split = in_shape.len() - rot;
    if in_shape[split..] != out_shape[..rot] || in_shape[..split] != out_shape[rot..] {
        let mut expected_out = in_shape.to_vec();
        expected_out.rotate_right(rot);
        return Err(ErrorKind::ShapeMismatch(out_shape.to_vec(),
                                            expected_out)
                   .into());
    }

    let gather =
        Gather::new(out_shape, |idxs| {
            let mut idxs = ShapeVec::from_slice(idxs);
            // The shape is rotated right, here we need the inverse
            // operation to get the original indices
            idxs.rotate_left(rot);
            (0, to_opt_ix(&idxs, in_shape))
        }, format!("rot_idx{{r={}}}", rot));
    Ok(vec![gather])
}

pub fn rot_idx_l(in_shapes: &[ShapeVec],
                 out_shape: &[Ix], rot: Ix) -> Result<Vec<Gather>> {
    if in_shapes.len() != 1 {
        return Err(ErrorKind::WrongArity(in_shapes.len(), 1).into());
    }
    let in_shape = in_shapes[0].as_slice();
    let split = in_shape.len() - rot;
    if in_shape[..rot] != out_shape[split..] || in_shape[rot..] != out_shape[..split] {
        let mut expected_out = in_shape.to_vec();
        expected_out.rotate_left(rot);
        return Err(ErrorKind::ShapeMismatch(out_shape.to_vec(),
                                            expected_out)
                   .into());
    }

    let gather =
        Gather::new(out_shape, |idxs| {
            let mut idxs = ShapeVec::from_slice(idxs);
            idxs.rotate_right(rot);
            (0, to_opt_ix(&idxs, in_shape))
        }, format!("rot_idx{{l={}}}", rot));
    Ok(vec![gather])
}

#[cfg(test)]
mod test {
    use crate::state::summarize;
    use smallvec::smallvec;

    #[test]
    fn summary_one() {
        let shape = vec![smallvec![3, 3]];
        let gather = super::identity(&shape, &[3, 3]).unwrap();
        let summary = summarize(&gather, 1).unwrap();
        assert_eq!(&gather[0], &summary);
    }

    #[test]
    fn summary_fails() {
        let gathers =
            vec![super::identity_gather(&[2, 3]),
                 super::transpose_gather(&[3, 2], &[2, 3])];
        assert_eq!(summarize(&gathers, 1), None)
    }

    #[test]
    fn summary_passes() {
        use crate::state::Gather;
        let map = std::collections::BTreeMap::new();
        let cond_keep: Vec<Gather> =
            super::select::cond_keep(&[4, 3], &[0, 1, -1],
                                     &map).unwrap();
        let summary = summarize(&cond_keep, 1);
        let identity = super::identity_gather(&[4, 3]);
        assert_eq!(summary, Some(identity));
    }
}
