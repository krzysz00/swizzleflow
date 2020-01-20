# SwizzleFlow - synthesizing high-performance kernels from dataflow graph sketches

We don't actually have any README content yet, but what this code will
do is located in [the designdocument](https://www.overleaf.com/read/twvwgqfbxmyx).

## Build
`cargo build --release`
To include more statistics about the operation of the algorithm at the cost of performance, use `cargo build --features stats --release`

To run, execute `./target/release/swizzleflow` or `cargo run`
The tool's arguments are
```
Krzysztof Drewniak <krzysd@cs.washington.edu> et al.
Tool for synthesizing high-performance kernels from dataflow graph sketches

USAGE:
    swizzleflow [FLAGS] [OPTIONS] [SPEC]...

FLAGS:
    -a, --all        Find all solutions
    -h, --help       Prints help information
    -p, --print      Print trace of valid salutions
    -V, --version    Prints version information

OPTIONS:
    -m, --matrix-dir <MATRIX_DIR>    Directory to store matrices in [default: matrices/]

ARGS:
    <SPEC>...    Specification files (stdin if none specified)
```
## Licensing
Copyright (C) 2019 Krzysztof Drewniak et al.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
<
