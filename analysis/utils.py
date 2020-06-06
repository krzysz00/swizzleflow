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
    '1d-conv': 'Convolution (weights, k=3)',
    '1d-stencil': 'Convolution (no weights, k=3)',
    'trove-crc-1': 'Trove (CRC, s=1)',
    'trove-crc-2': 'Trove (CRC, s=2)',
    'trove-crc-3': 'Trove (CRC, s=3)',
    'trove-crc-4': 'Trove (CRC, s=4)',
    'trove-crc-5': 'Trove (CRC, s=5)',
    'trove-crc-7': 'Trove (CRC, s=7)',
    'trove-cr_sum-1': 'Trove (Sum, s=1)',
    'trove-cr_sum-2': 'Trove (Sum, s=2)',
    'trove-cr_sum-3': 'Trove (Sum, s=3)',
    'trove-cr_sum-4': 'Trove (Sum, s=4)',
    'trove-cr_sum-5': 'Trove (Sum, s=5)',
    'trove-cr_sum-7': 'Trove (Sum, s=7)',
    'trove-rcr-1': 'Trove (RCR, s=1)',
    'trove-rcr-2': 'Trove (RCR, s=2)',
    'trove-rcr-3': 'Trove (RCR, s=3)',
    'trove-rcr-4': 'Trove (RCR, s=4)',
    'trove-rcr-5': 'Trove (RCR, s=5)',
    'trove-rcr-7': 'Trove (RCR, s=7)',
    '2d-stencil-3': '2D stencil (k=3)',
    '2d-stencil-5': '2D stencil (k=5)',
    '2d-stencil-7': '2D stencil (k=7)',
    'mult-32-with-4': 'FFM (w=4)',
}
