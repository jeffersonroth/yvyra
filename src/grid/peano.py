"""Peano curves."""

import numpy as np
from typing import Tuple


def get_peano_indices(n: int) -> np.ndarray:
    """
    Generates the Peano curve path indices for an n x n grid.
    n must be a power of 3.
    """
    if n == 1:
        return np.array([[0, 0]])

    # Recursive call for the smaller scale
    prev_indices = get_peano_indices(n // 3)
    m = n // 3

    new_indices = []

    # Define the 3x3 traversal order (row, col)
    # Following: BL, ML, TL, TM, MM, BM, BR, MR, TR
    # Note: Using (row, col) where row 0 is bottom, row 2 is top
    order = [
        (0, 0, False, False),
        (1, 0, False, False),
        (2, 0, False, False),  # Left column (up)
        (2, 1, True, False),
        (1, 1, True, False),
        (0, 1, True, False),  # Middle column (down + horiz flip)
        (0, 2, False, False),
        (1, 2, False, False),
        (2, 2, False, False),  # Right column (up)
    ]

    for row_offset, col_offset, flip_row, flip_col in order:
        sub_grid = prev_indices.copy()

        if flip_row:
            # Mirror the sub-grid vertically
            sub_grid[:, 0] = (m - 1) - sub_grid[:, 0]
        if flip_col:
            # Mirror the sub-grid horizontally
            sub_grid[:, 1] = (m - 1) - sub_grid[:, 1]

        # Offset the sub-grid to its position in the larger NxN grid
        sub_grid[:, 0] += row_offset * m
        sub_grid[:, 1] += col_offset * m
        new_indices.append(sub_grid)

    return np.vstack(new_indices)


def peano_to_row_major(arr: np.ndarray) -> np.ndarray:
    """
    Takes an NxN array and returns a 1xN^2 array ordered by the Peano curve.
    """
    N = arr.shape[0]
    # Ensure N is a power of 3
    if not (N > 0 and 3 ** int(np.round(np.log(N) / np.log(3))) == N):
        raise ValueError("Array dimension must be a power of 3 (3, 9, 27...).")

    indices = get_peano_indices(N)

    # Extract values based on the Peano path
    # We use N-1-row because standard row-major (0,0) is Top-Left,
    # but Peano logic usually starts Bottom-Left.
    path_values = [arr[N - 1 - r, c] for r, c in indices]

    return np.array(path_values)


if __name__ == "__main__":
    from grid.utils import row_major_to_grid, grid_to_row_major

    row_major_input = "467100805912835607085647192296351470708920351531408926073064510624519783159783064"
    peano_expected = "160725943572903186094618527186430057394025108057816493075186430934752610168009275"

    grid = row_major_to_grid(row_major_input, (9, 9))
    assert grid is not None, "Failed to convert row major to grid"
    assert grid.shape == (9, 9), "Grid shape is not 9x9"
    print(grid)

    peano_grid = peano_to_row_major(grid)
    assert peano_grid is not None, "Failed to convert solver grid to peano grid"
    peano_row_major = grid_to_row_major(peano_grid)
    assert peano_row_major is not None, "Failed to convert peano grid to row major"
    print(peano_row_major)
    assert peano_row_major == peano_expected, "Peano grid values are not as expected"
