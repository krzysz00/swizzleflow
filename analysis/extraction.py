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

import pandas as pd
import os
import re

def select_categories(results, categories=None):
    if isinstance(results, dict):
        return {k: select_categories(v, categories) for k, v in results.items()}
    elif isinstance(results, list):
        if categories is not None:
            results = [x for x in results if x["category"] in categories]
        frame = pd.DataFrame(results)
        return frame
    else:
        raise ValueError("Unexpected results format")

def matrix_stats(results):
    return select_categories(results, ["build", "load", "mul", "union"])

def search_stats(results):
    ret = select_categories(results, ["stats"])
    for v in ret.values():
        v.drop(columns=["category", "key"], inplace=True)
    return ret

def search_info(results):
    select = select_categories(results, ["search"])
    for k, v in select.items():
        v.drop(columns=["category"], inplace=True)
        v["key"] = k
    ret = pd.concat(select.values())
    ret.set_index(ret["key"], inplace=True)
    ret.drop(columns=["key"], inplace=True)
    ret.index.name = "spec"
    return ret

def pull_spec_in(tables):
    ret = []
    for spec, df in tables.items():
        new_df = df.copy()
        new_df["spec"] = spec
        ret.append(new_df)
    return pd.concat(ret)

_SWINV_SPEC_RE = re.compile("^specs/swinv_like[^/]*/l(\\d)/.*\\.json$")
def _humanize_name(name):
    ret = os.path.basename(name)
    if ret[-5:] == ".json":
        ret = ret[:-5]
    match = _SWINV_SPEC_RE.match(name)
    if match:
        ret = f"l{match[1]}/{ret}"
    return ret

def humanize_names(specs):
    if isinstance(specs, dict):
        return {_humanize_name(k): v for k, v, in specs.items()}

    if isinstance(specs, pd.Series) or isinstance(specs, pd.DataFrame):
        ret = specs.copy()
        if len(ret.index) > 0 and isinstance(ret.index[0], str):
            ret.index = ret.index.map(_humanize_name)
        if "spec" in ret:
            ret["spec"] = ret["spec"].map(_humanize_name)
        return ret

def expand_target_checks(df, to_copy=["spec", "lane"]):
    if "target_checks" not in df:
        raise ValueError("No target check stats to expand")
    col = df["target_checks"]
    values = [dict(vs) for vs in col]
    keys = set().union(*[v.keys() for v in values])
    for d in values:
        for k in keys:
            if k not in d:
                d[k] = 0

    final_df = pd.DataFrame(values, index=col.index)
    final_df = final_df.reindex(sorted(final_df.columns), axis=1)
    for name in to_copy:
        if name in df:
            final_df.insert(0, name, df[name])
    return final_df

def compute_basis_size(df):
    if isinstance(df, dict):
        for v in df.values():
            compute_basis_size(v)
    else:
        moved_cont = pd.Series([1]).append(df['continued'][:-1], ignore_index=True)
        df['basis_size'] = df['tested'] // moved_cont
