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

import ast
import fileinput
import pandas as pd
import re

def parse_value(string):
    if string == "true":
        return True
    elif string == "false":
        return False
    else:
        try:
            return ast.literal_eval(string)
        except (ValueError, SyntaxError):
            return string

STATS_REGEX = re.compile("(\\w+):(\\S+)\\s+(.*)")
DATUM_REGEX = re.compile("(\w+)=([^;]+);")

def parse_results(stream):
    ret = {}
    accum = None
    for line in stream:
        line = line.strip()
        if line.startswith("spec:"):
            _, name = line.split(":")
            ret[name] = list()
            # Reference to same object, will update
            accum = ret[name]
            continue
        if line.startswith("solution:"):
            value = line[len("solution:"):]
            accum.append({"category": "solution", "key": value})
            continue
        match = STATS_REGEX.match(line)
        if match is not None:
            category = match[1]
            key = match[2]
            data = match[3]
            parsed = {"category": category, "key": parse_value(key) if key != ":" else len(accum)}
            for key, value in DATUM_REGEX.findall(data):
                parsed[key] = parse_value(value)
            accum.append(parsed)
    return ret

def parse_file(path):
    with open(path, mode='r') as f:
        return parse_results(f)

def get_results():
    return parse_results(fileinput.input())

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
