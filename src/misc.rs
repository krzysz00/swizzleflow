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

#[allow(dead_code)]
pub(crate) const EPSILON: f32 = 1e-5;

pub type ShapeVec = smallvec::SmallVec<[usize; 3]>;

#[cfg(feature = "stats")]
pub const COLLECT_STATS: bool = true;
#[cfg(not(feature = "stats"))]
pub const COLLECT_STATS: bool = false;

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

pub fn extending_set<T: Default>(vec: &mut Vec<T>, idx: usize, item: T) {
    for _ in vec.len()..=idx {
        vec.push(T::default());
    }
    vec[idx] = item
}

use ndarray::Ix;
pub fn in_bounds(index: &[Ix], bounds: &[Ix]) -> bool {
    index.iter().zip(bounds.iter()).all(move |(i, v)| i < v)
}

// Used in previous experiments for reporting, might be needed again
#[allow(dead_code)]
pub fn loghist(n: usize) -> usize {
    if n < 10 {
        n
    }
    else if n < 100 {
        (n / 10) * 10
    }
    else {
        (n / 100) * 100
    }
}

pub fn parse_opt_arg<T: std::str::FromStr>(arg: Option<&str>,
                                           name: &'static str,
                                           reqs: &'static str) -> Result<Option<T>> {
    if let Some(s) = arg {
        s.parse::<T>()
            .map_err(|_| ErrorKind::InvalidCmdArg(name, reqs).into())
            .map(|v| Some(v))
    }
    else {
        Ok(None)
    }
}

use std::io::{Read,Write};
use byteorder::{LittleEndian,WriteBytesExt,ReadBytesExt};
pub fn write_length_tagged_idxs<T: Write>(io: &mut T, data: &[Ix]) -> std::io::Result<()> {
    io.write_u64::<LittleEndian>(data.len() as u64)?;
    for i in data {
        io.write_u64::<LittleEndian>(*i as u64)?;
    }
    Ok(())
}

pub fn read_length_tagged_idxs<T: Read>(io: &mut T) -> std::io::Result<ShapeVec> {
    let length = io.read_u64::<LittleEndian>()? as usize;
    let mut buffer = ShapeVec::with_capacity(length);
    for _ in 0..length {
        buffer.push(io.read_u64::<LittleEndian>()? as usize);
    }
    Ok(buffer)
}
