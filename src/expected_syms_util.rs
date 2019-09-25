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
use crate::state::{DomRef,Domain,ProgState};

use std::collections::BTreeSet;

pub fn expcted_of_inital_state(prog: &ProgState) -> BTreeSet<DomRef> {
    prog.state.iter().copied().collect()
}

pub fn fold_expected(domain: &Domain, current: &BTreeSet<DomRef>) -> BTreeSet<DomRef> {
    let superterms: BTreeSet<DomRef> = current.iter().copied()
        .flat_map(|c| domain.imm_superterms(c).iter().copied())
        .collect();
    superterms.into_iter()
        .filter(|x| domain.subterms_all_within(*x, current))
        .collect()
}
