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
import pandas
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
            parsed = {"category": category, "key": parse_value(key) if key != ":" else ':'}
            for key, value in DATUM_REGEX.findall(data):
                parsed[key] = parse_value(value)
            # Merge in statistics about multiplications
            if accum and accum[-1]['key'] == ':' and accum[-1]['category'].endswith('_stats')\
               and accum[-1]['category'].startswith(category):
                stats = accum.pop()
                stats.update(parsed)
                parsed = stats
            accum.append(parsed)
    return ret

def parse_file(path):
    with open(path, mode='r') as f:
        return parse_results(f)

def get_results():
    return parse_results(fileinput.input())

def parse_swizzle_inventor(stream):
    data = []
    for line in stream:
        if line.startswith("run:"):
            spec, raw_time = line[4:].split(' ')
            data.append([spec, int(raw_time)])
    return pandas.DataFrame(data, columns=["spec", "time"])

def parse_swizzle_inventor_file(path):
    with open(path, mode='r') as f:
        return parse_swizzle_inventor(f)
