"""Grid."""

import numpy as np

from grid.symbols import Symbols, get_unknown_symbol


def validate_grid(
    grid: np.ndarray,
    symbols: Symbols,
) -> bool:
    """Validate if the Sudoku grid is valid*.

    Only valid for square grids.
    Only validates that:
    - Only valid symbols are present
    - Symbols count is valid
    - Symbols do not repeat in rows (except for unknown symbols)
    - Symbols do not repeat in columns (except for unknown symbols)
    - Symbols do not repeat in boxes (except for unknown symbols)
    """
    # check if grid shape is valid
    if grid.shape[0] != grid.shape[1]:
        return False
    unknown_symbol = get_unknown_symbol(symbols)
    # check if grid is only composed of valid symbols
    if not all(cell in symbols for row in grid for cell in row):
        return False
    # check if symbols do not repeat in rows (except for unknown symbols)
    for row in grid:
        if len(set(row) - {unknown_symbol}) != len(row) - (row == unknown_symbol).sum():
            return False
    # check if symbols do not repeat in columns (except for unknown symbols)
    for col in grid.T:
        if len(set(col) - {unknown_symbol}) != len(col) - (col == unknown_symbol).sum():
            return False
    # check if symbols do not repeat in sqrt(n) x sqrt(n) boxes (except for unknown symbols)
    n = int(np.sqrt(grid.shape[0]))
    for i in range(0, grid.shape[0], n):
        for j in range(0, grid.shape[1], n):
            box = grid[i : i + n, j : j + n]
            if len(set(box.flatten()) - {unknown_symbol}) != len(
                box.flatten()
            ) - (box.flatten() == unknown_symbol).sum():
                return False
    return True


def validate_solution(grid: np.ndarray, symbols: Symbols) -> bool:
    """Validate if the Sudoku grid is solved.

    Only valid for square grids.
    Only validates that:
    - Unknown symbol is not present
    - Only valid symbols are present
    - Symbols count is valid
    - Symbols do not repeat in rows
    - Symbols do not repeat in columns
    - Symbols do not repeat in boxes
    """
    # check if unknown symbol in grid
    unknown_symbol = get_unknown_symbol(symbols)
    if unknown_symbol in grid:
        return False
    return validate_grid(grid, symbols)


class Grid:
    """Grid class."""

    def __init__(self, grid: np.ndarray, symbols: Symbols):
        self.grid = grid
        self.symbols = symbols

    def __repr__(self):
        return f"Grid({self.grid}, {self.symbols})"

    def is_valid(self) -> bool:
        """Check if the grid is valid."""
        return validate_grid(self.grid, self.symbols)

    def is_solved(self) -> bool:
        """Check if the grid is solved."""
        return validate_solution(self.grid, self.symbols)


if __name__ == "__main__":
    from grid.symbols import DEFAULT_4X4_SYMBOLS, DEFAULT_9X9_SYMBOLS
    from grid.utils import row_major_to_grid

    # 9x9
    UNSOLVED_ROW_MAJOR_9X9 = "467100805912835607085647192296351470708920351531408926073064510624519783159783064"
    SOLVED_ROW_MAJOR_9X9 = "467192835912835647385647192296351478748926351531478926873264519624519783159783264"

    _unsolved_grid_9x9 = row_major_to_grid(UNSOLVED_ROW_MAJOR_9X9, grid_type=(9, 9))
    assert _unsolved_grid_9x9 is not None, "Failed to convert row major to 9x9 grid"
    unsolved_grid_9x9 = Grid(_unsolved_grid_9x9, DEFAULT_9X9_SYMBOLS)
    print(unsolved_grid_9x9)
    assert unsolved_grid_9x9.is_valid() is True, "Unsolved grid is not valid"
    assert (
        unsolved_grid_9x9.is_solved() is False
    ), "Unsolved grid incorrectly marked as solved"

    _solved_grid_9x9 = row_major_to_grid(SOLVED_ROW_MAJOR_9X9, grid_type=(9, 9))
    assert _solved_grid_9x9 is not None, "Failed to convert row major to 9x9 grid"
    solved_grid_9x9 = Grid(_solved_grid_9x9, DEFAULT_9X9_SYMBOLS)
    print(solved_grid_9x9)
    assert solved_grid_9x9.is_valid() is True, "Solved grid is not valid"
    assert (
        solved_grid_9x9.is_solved() is True
    ), "Solved 9x9 grid incorrectly marked as unsolved"

    # 4x4
    UNSOLVED_ROW_MAJOR_4X4 = "1000000400200300"
    SOLVED_ROW_MAJOR_4X4 = "1432321441232341"

    _unsolved_grid_4x4 = row_major_to_grid(UNSOLVED_ROW_MAJOR_4X4, grid_type=(4, 4))
    assert _unsolved_grid_4x4 is not None, "Failed to convert row major to 4x4 grid"
    unsolved_grid_4x4 = Grid(_unsolved_grid_4x4, DEFAULT_4X4_SYMBOLS)
    print(unsolved_grid_4x4)
    assert unsolved_grid_4x4.is_valid() is True, "Unsolved grid is not valid"
    assert (
        unsolved_grid_4x4.is_solved() is False
    ), "Unsolved grid incorrectly marked as solved"

    _solved_grid_4x4 = row_major_to_grid(SOLVED_ROW_MAJOR_4X4, grid_type=(4, 4))
    assert _solved_grid_4x4 is not None, "Failed to convert row major to 4x4 grid"
    solved_grid_4x4 = Grid(_solved_grid_4x4, DEFAULT_4X4_SYMBOLS)
    print(solved_grid_4x4)
    assert solved_grid_4x4.is_valid() is True, "Solved 4x4 grid is not valid"
    assert (
        solved_grid_4x4.is_solved() is True
    ), "Solved 4x4 grid incorrectly marked as unsolved"
