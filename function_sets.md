# Function sets

This is a list of all the built-in sets of function/gathers in Swizzleflow
(though often a "set" is one element), what they do, and what their options are.

## `identity` or `reshape`
At its simplest, the identity function.

However, if you use a different out_shape than the input,
you have something that'll act like numpy's `reshape` - reading the input in row-major order.

## `transpose`
It's a transpose. Unlike with `identity`, you can't pull any fancy different-shape tricks here

## `rot_idx`

Rotates your array indices.

You'll want this if, for example, you need to fold on an axis that isn't the last one - at least until general folds become supported.

For example, if you have a 2x3x4 array, you can create a 4x2x3 array by rotating right by 1.
Thit'll map, say, the element at (1, 2, 3) to (3, 1, 2).

**Options:**
You must give either `r` or `l`, a one-element array containing the constant to rotate right or left by.

If you give both `r` and `l`, that's an error

## `load_rep`

A column-major load, repeating as needed

## `load_trunc`

Also a column-major load, but this time it loads 0s after the input runs out

## `load_grid_2d`

Loads onto a grid of threads, going from (i, j) to (i1, j1, i2, j2),
in column-major order.

## `broadcast`

Makes copies of the elements to fill the new axes.

Ex, takes a 3x3 array to a 4x3x3 by copying it.

**Options:** `group` sets where the new axis is. Ex. We can make a 3x3 array into a 3x4x3 array (broadcasting the rows) with `group: [1]`.

***TODO*** describe the actual bases here, as opposed to the finicky shuffling constructs.
