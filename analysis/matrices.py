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
import numpy as np
import matplotlib.pyplot as plt

import collections
from typing import Sequence, List

def read_u32(f):
    return int.from_bytes(f.read(4), byteorder='little', signed=False)

def read_u64(f):
    return int.from_bytes(f.read(8), byteorder='little', signed=False)

DENSE_MATRIX_TAG = 3;
ROW_SPARSE_MATRIX_TAG = 4
FILE_MARKER =  b"SWIZFLOW"
BIT_WIDTH = 32

class Matrix:
    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n
        self.data = np.zeros((m, n), bool)

    def get(self, i: int, j: int) -> bool:
        return self.data[i, j]

    def set(self, i: int, j: int, v: bool) -> None:
        self.data[i, j] = v

    @staticmethod
    def load(file) -> 'Matrix':
        header = file.read(8)
        if header != FILE_MARKER:
            raise ValueError("Not a swizzleflow matrix")
        tag = file.read(1)[0]
        m = read_u64(file)
        n = read_u64(file)
        ret = Matrix(m, n)

        if tag == DENSE_MATRIX_TAG:
            for i in range(m):
                io_len = read_u64(file)
                for j in range(io_len):
                    data = read_u32(file)
                    for jj in range(BIT_WIDTH):
                        idx = jj + BIT_WIDTH * j
                        if idx < n:
                            ret.set(i, idx,
                                    (data & (1 << jj)) != 0)
        elif tag == ROW_SPARSE_MATRIX_TAG:
            for i in range(m):
                row_length = read_u64(file)
                for _ in range(row_length):
                    j = read_u64(file)
                    ret.set(i, j, True)
        else:
            raise ValueError("Unknown matrix type")
        return ret

def row_major(index: Sequence[int], dims: Sequence[int]) -> int:
    ret = 0
    for i, bound in zip(index, dims):
        ret = ret * bound + i
    return ret

def read_sequence(file) -> List[int]:
    length = read_u64(file)
    return [read_u64(file) for _ in range(length)]

class TransitionMatrix:
    def __init__(self, current_dims: Sequence[int], target_dims: Sequence[int],
                 matrix: Matrix) -> None:
        self.current_dims = tuple(current_dims)
        self.target_dims = tuple(target_dims)
        self.current_len = np.prod(self.current_dims)
        self.target_len = np.prod(self.target_dims)
        self.matrix = matrix
        assert(matrix.m == self.current_len * self.current_len)
        assert(matrix.n == self.target_len * self.target_len)

    def get(self, i1: Sequence[int], i2: Sequence[int],
            j1: Sequence[int], j2: Sequence[int]) -> bool:
        i1_idx = row_major(i1, self.current_dims)
        i2_idx = row_major(i2, self,current_dims)
        j1_idx = row_major(j1, self.target_dims)
        j2_idx = row_major(i2, self.target-dims)
        return self.get_idxs(i1_id2, i2_idx, j1_idx, j2_idx)
    def get_idxs(self, i1: int, i2: int, j1: int, j2: int) -> bool:
        self.matrix.get(i2 + self.current_len * i1,
                        j2 + self.target_len * j1)

    @staticmethod
    def load(file) -> 'TransitionMatrix':
        current_dims = read_sequence(file)
        target_dims = read_sequence(file)
        matrix = Matrix.load(file)
        return TransitionMatrix(current_dims, target_dims, matrix)

    @staticmethod
    def from_file(filename) -> 'TransitionMatrix':
        with open(filename, 'rb') as f:
            return TransitionMatrix.load(f)

def de_row_major(idx: int, dims: Sequence[int]) -> List[int]:
    ret = []
    for d in reversed(dims):
        ret.append(idx % d)
        idx = idx // d
    ret.reverse()
    return ret

def label_row(idx: int, dims: Sequence[int]) -> str:
    return f"({','.join(map(str, de_row_major(idx, dims)))})"

def to_tick_label(i: int, dims: Sequence[int]) -> str:
    return label_row(i, dims)

def visualize_slice(mat: TransitionMatrix, d1: int, d2: int, fixed_dim='target') -> plt.Figure:
    # The other fixed_dim is 'current
    if fixed_dim not in ['current', 'target']:
        raise ArgumentError("Invalid fixed_dim, must be 'current' or 'target")
    slice_target = (fixed_dim == 'target')
    off_dims, off_len = (mat.target_dims, mat.target_len) if slice_target\
        else (mat.current_dims, mat.current_len)
    if isinstance(d1, collections.Sequence):
        d1 = row_major(d1, off_dims)
    if isinstance(d2, collections.Sequence):
        d2 = row_major(d2, off_dims)

    slice = d2 + (off_len * d1)
    extent = mat.current_len if slice_target else mat.target_len
    data = (mat.matrix.data[:, slice] if slice_target
            else mat.matrix.data[slice, :]).reshape((extent, extent))
    dims = mat.current_dims if slice_target else mat.target_dims
    last_dim = dims[-1]

    fig = plt.Figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(data, cmap=plt.cm.gray.reversed(), interpolation='nearest')

    ticks = np.arange(0, extent, last_dim)
    ax.set_xticks(ticks - 0.5)
    ax.set_xticklabels(map(lambda i: to_tick_label(i, dims), ticks))
    ax.set_yticks(ticks - 0.5)
    ax.set_yticklabels(map(lambda i: to_tick_label(i, dims), ticks))

    ax.grid(which='major', color='0.35')

    ax.minorticks_on()
    minor_ticks = np.arange(0, extent) - 0.5
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', color='0.8')

    ax.set_title(f"First/second location {'reaches' if slice_target else 'can come from'} {label_row(d1, off_dims)} and {label_row(d2, off_dims)}, respectively")
    ax.set_ylabel("First location")
    ax.set_xlabel("Second location in pair")

    return fig
