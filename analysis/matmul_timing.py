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

import sys
import os.path

import pandas as pd

import parsing
import extraction

def join_stats(dfs, names=None):
    dfs = [(df[df['category'] == 'mul'].drop('category', axis=1)\
            if 'category' in df else df)\
           for df in dfs]
    if names is None:
        names = [f"time[{i}]" for i in range(len(dfs))]
    for df in dfs:
        df['key'] = df['key'].map(lambda s:
                                  s[len('matrices/'):] if s.startswith('matrices/')
                                  else s)
    dfs = [df[['key', 'spec', 'time']]
           .rename({'key': 'matrix', 'time': name}, axis=1)
           .set_index('matrix') for df, name in zip(dfs, names)]
    ret = dfs[0].join([df.drop('spec', axis=1) for df in dfs[1:]])
    return ret

def join_results(raw_datas, names=None):
    dfs = [extraction.pull_spec_in(extraction.matrix_stats(\
                extraction.humanize_names(data)))
           for data in raw_datas]
    return join_stats(dfs, names)

def load_results(files, names=None):
    if names is None:
        names = [f"{os.path.basename(name)} time" for name in files]
    return join_results([parsing.parse_file(f) for f in files], names)

if __name__ == '__main__':
    df = load_results(sys.argv[1:])
    print(df.to_csv())
