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

pub(crate) const EPSILON: f32 = 1e-5;
pub type ShapeVec = smallvec::SmallVec<[usize; 3]>;

pub fn time_since(start: std::time::Instant) -> f64 {
    let dur = start.elapsed();
    dur.as_secs() as f64 + (f64::from(dur.subsec_nanos()) / 1.0e9)
}

use std::fs::File;
use std::path::Path;
pub fn open_file<P: AsRef<Path>>(path: P) -> Result<File> {
    File::open(path.as_ref()).map_err(Error::from)
        .chain_err(|| ErrorKind::FileOpFailure(path.as_ref().to_owned()))
}

pub fn create_file<P: AsRef<Path>>(path: P) -> Result<File> {
    File::create(path.as_ref()).map_err(Error::from)
        .chain_err(|| ErrorKind::FileOpFailure(path.as_ref().to_owned()))
}
