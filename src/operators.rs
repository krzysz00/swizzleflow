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
use crate::misc::ShapeVec;

use smallvec::SmallVec;

use std::borrow::Cow;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operators {
    pub name: Cow<'static, str>,
    pub ops: Vec<Gather>,
    pub in_shape: ShapeVec,
    pub out_shape: ShapeVec,
}

impl Operators {
    pub fn new<T>(name: T, ops: Vec<Gather>, in_shape: ShapeVec, out_shape: ShapeVec) -> Self
    where T: Into<Cow<'static, str>> {
        Self { name: name.into(), ops, in_shape, out_shape }
    }

    pub fn to_name(&self) -> String {
        let in_strings: SmallVec<[String; 4]> = self.in_shape.iter().map(|v| v.to_string()).collect();
        let out_strings: SmallVec<[String; 4]> = self.out_shape.iter().map(|v| v.to_string()).collect();
        format!("{}-{}-{}", in_strings.join(","), self.name, out_strings.join(","))
    }
}
