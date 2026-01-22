"""Utils."""

from typing import Tuple
import numpy as np

from grid.symbols import Symbols, get_possible_symbols, get_unknown_symbol


def row_major_to_grid(row_major: str, grid_type: Tuple[int, int]) -> np.ndarray:
    """
    Convert a row-major string to a 2D grid.
    """
    grid = np.zeros(grid_type, dtype=str)
    for i, char in enumerate(row_major):
        row = i // grid_type[1]
        col = i % grid_type[1]
        grid[row, col] = str(char)

    return grid


def char_grid_to_solver_grid(char_grid: np.ndarray, symbols: Symbols) -> np.ndarray:
    """
    Convert a character grid (2D array of symbols) to a integer grid used by the solver.

    Note that it's not transforming the characters themselves,
    but rather the int representation of 2 to the power of the characters's position in the grid's symbol set.
    Example:
        - Symbols: ["?", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
        - Hex: [0x0, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000]
    """
    unknown = get_unknown_symbol(symbols)  # This is "?"
    possibles = get_possible_symbols(symbols)  # These are ["0", ..., "F"]

    symbols_map = {ch: (1 << i) for i, ch in enumerate(possibles)}
    symbols_map[unknown] = 0

    int_grid = np.zeros(char_grid.shape, dtype=int)
    for i in range(char_grid.shape[0]):
        for j in range(char_grid.shape[1]):
            int_grid[i, j] = symbols_map.get(char_grid[i, j], 0)
    return int_grid


if __name__ == "__main__":
    from grid.symbols import DEFAULT_9X9_SYMBOLS

    ROW_MAJOR = "467100805912835607085647192296351470708920351531408926073064510624519783159783064"
    grid = row_major_to_grid(ROW_MAJOR, (9, 9))
    assert grid is not None, "Failed to convert row major to grid"
    assert grid.shape == (9, 9), "Grid shape is not 9x9"
    print(grid)
    solver_grid = char_grid_to_solver_grid(grid, DEFAULT_9X9_SYMBOLS)
    assert solver_grid is not None, "Failed to convert char grid to solver grid"
    print(np.vectorize(np.binary_repr)(solver_grid, width=solver_grid.shape[1]))
