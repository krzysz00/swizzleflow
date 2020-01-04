#!/usr/bin/env python3
# Copyright (C) 2019 Krzysztof Drewniak et al.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import parsing
import pandas as pd

def process(results):
    search_stats = parsing.search_stats(results)
    for spec, df in search_stats.items():
        df["spec"] = spec
    df = pd.concat(search_stats.values())
    return df

if __name__ == '__main__':
    results = parsing.get_results()
    output = process(results)
    printable = output[["spec", "name", "tested", "found", "pruned", "failed", "continued"]]
    print(printable.to_csv(index=False), end='')
