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
                        if jj + BIT_WIDTH * j < n:
                            ret.set(i, jj + BIT_WIDTH * j,
                                    (data & (1 << j)) != 0)
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
