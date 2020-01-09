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
import extraction
import pandas as pd

def process(results):
    matrix_info = extraction.matrix_stats(results)
    search_info = extraction.select_categories(results, ["search"])
    output = {}
    for spec, m in matrix_info.items():
        matrix_time = m["time"].sum() if len(m) > 0 else 0.0
        search_time = search_info[spec]["time"].sum()
        output[spec] = {"matrix time": matrix_time, "search time": search_time, "total": matrix_time + search_time}
    df = pd.DataFrame.from_dict(output, orient='index')
    df.index.name = "spec"
    return df

if __name__ == '__main__':
    results = parsing.get_results()
    output = process(results)
    print(output.to_csv(), end='')
