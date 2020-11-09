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

def fetch(dataset):
    return extraction.humanize_names(parsing.parse_file(f"../results/{dataset}"))

def fetch_swizzle_inventor(dataset):
    return extraction.extract_swizzle_inventor_times(
        parsing.parse_swizzle_inventor_file(f"../results/{dataset}"))

PROBLEM_NAMES = {
    '1d-conv': '1D convolution (k=3)',
    '1d-conv-5': '1D convolution (k=5)',
    '1d-conv-7': '1D convolution (k=7)',
    '1d-conv-9': '1D convolution (k=9)',
    '1d-conv-9': '1D convolution (k=9)',
    '1d-conv-11': '1D convolution (k=11)',
    '1d-conv-13': '1D convolution (k=13)',
    '1d-stencil': '1D stencil (k=3)',
    'trove-crc-1': 'Trove (CRC, s=1)',
    'trove-crc-2': 'Trove (CRC, s=2)',
    'trove-crc-3': 'Trove (CRC, s=3)',
    'trove-crc-4': 'Trove (CRC, s=4)',
    'trove-crc-5': 'Trove (CRC, s=5)',
    'trove-crc-7': 'Trove (CRC, s=7)',
    'trove-crc-9': 'Trove (CRC, s=9)',
    'trove-cr_sum-1': 'Trove (Sum, s=1)',
    'trove-cr_sum-2': 'Trove (Sum, s=2)',
    'trove-cr_sum-3': 'Trove (Sum, s=3)',
    'trove-cr_sum-4': 'Trove (Sum, s=4)',
    'trove-cr_sum-5': 'Trove (Sum, s=5)',
    'trove-cr_sum-7': 'Trove (Sum, s=7)',
    'trove-cr_sum-9': 'Trove (Sum, s=9)',
    'trove-cr_sum-11': 'Trove (Sum, s=11)',
    'trove-cr_sum-13': 'Trove (Sum, s=13)',
    'trove-rcr-1': 'Trove (RCR, s=1)',
    'trove-rcr-2': 'Trove (RCR, s=2)',
    'trove-rcr-3': 'Trove (RCR, s=3)',
    'trove-rcr-4': 'Trove (RCR, s=4)',
    'trove-rcr-5': 'Trove (RCR, s=5)',
    'trove-rcr-7': 'Trove (RCR, s=7)',
    'trove-rcr-9': 'Trove (RCR, s=9)',
    '2d-stencil-3': '2D stencil (k=3)',
    '2d-stencil-5': '2D stencil (k=5)',
    '2d-stencil-7': '2D stencil (k=7)',
    '2d-stencil-9': '2D stencil (k=9)',
    'mult-32-with-4': 'FFM (w=4, registers)',
    'mult-64-with-16': 'FFM (w=8, registers)',
    'mult-64-with-16-shared': 'FFM (w=8, shared mem)'
}

def split_spec_pretty(df):
    df = df.copy()
    df = extraction.split_spec(df)
    df.insert(0, 'Level', df['level'].map({1: 'R', 2: 'FC', 3: 'F'}))
    df.insert(0, 'Problem', df['problem'].map(PROBLEM_NAMES))
    df.drop(columns=['spec', 'level', 'problem'], inplace=True)
    return df
