"""Grid."""

import numpy as np

from grid.symbols import Symbols
from grid.utils import validate_grid, validate_solution, row_major_to_grid, grid_to_row_major


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
    
    @classmethod
    def from_row_major(cls, row_major: str, grid_type: tuple[int, int], symbols: Symbols):
        """Create a Grid from a row major string."""
        grid_array = row_major_to_grid(row_major, grid_type)
        if grid_array is None:
            raise ValueError("Invalid row major string for the given grid type.")
        return cls(grid_array, symbols)
    
    @classmethod
    def from_peano(cls, peano: str, grid_type: tuple[int, int], symbols: Symbols):
        """Create a Grid from a Peano curve string."""
        row_major = grid_to_row_major(np.array(list(peano)).reshape(grid_type))
        if row_major is None:
            raise ValueError("Invalid Peano string for the given grid type.")
        return cls.from_row_major(row_major, grid_type, symbols)
    
    @classmethod
    def from_array(cls, array: np.ndarray, grid_type: tuple[int, int], symbols: Symbols):
        """Create a Grid from a 1xN where N is the number of cells in the grid."""
        if array.shape[0] != grid_type[0] * grid_type[1]:
            raise ValueError("Array length does not match the number of cells in the grid.")
        row_major = grid_to_row_major(array)
        if row_major is None:
            raise ValueError("Invalid array for the given grid type.")
        return cls.from_row_major(row_major, grid_type, symbols)
        


if __name__ == "__main__":
    from grid.symbols import DEFAULT_4X4_SYMBOLS, DEFAULT_9X9_SYMBOLS
    from grid.utils import row_major_to_grid

    # 9x9
    UNSOLVED_ROW_MAJOR_9X9 = "467100805912835607085647192296351470708920351531408926073064510624519783159783064"
    SOLVED_ROW_MAJOR_9X9 = "467192835912835647385647192296351478748926351531478926873264519624519783159783264"

    unsolved_grid_9x9 = Grid.from_row_major(UNSOLVED_ROW_MAJOR_9X9, grid_type=(9, 9), symbols=DEFAULT_9X9_SYMBOLS)
    print(unsolved_grid_9x9)
    assert unsolved_grid_9x9.is_valid() is True, "Unsolved grid is not valid"
    assert (
        unsolved_grid_9x9.is_solved() is False
    ), "Unsolved grid incorrectly marked as solved"

    solved_grid_9x9 = Grid.from_row_major(SOLVED_ROW_MAJOR_9X9, grid_type=(9, 9), symbols=DEFAULT_9X9_SYMBOLS)
    print(solved_grid_9x9)
    assert solved_grid_9x9.is_valid() is True, "Solved grid is not valid"
    assert (
        solved_grid_9x9.is_solved() is True
    ), "Solved 9x9 grid incorrectly marked as unsolved"

    # 4x4
    UNSOLVED_ROW_MAJOR_4X4 = "1000000400200300"
    SOLVED_ROW_MAJOR_4X4 = "1432321441232341"

    unsolved_grid_4x4 = Grid.from_row_major(UNSOLVED_ROW_MAJOR_4X4, grid_type=(4, 4), symbols=DEFAULT_4X4_SYMBOLS)
    print(unsolved_grid_4x4)
    assert unsolved_grid_4x4.is_valid() is True, "Unsolved grid is not valid"
    assert (
        unsolved_grid_4x4.is_solved() is False
    ), "Unsolved grid incorrectly marked as solved"

    solved_grid_4x4 = Grid.from_row_major(SOLVED_ROW_MAJOR_4X4, grid_type=(4, 4), symbols=DEFAULT_4X4_SYMBOLS)
    print(solved_grid_4x4)
    assert solved_grid_4x4.is_valid() is True, "Solved 4x4 grid is not valid"
    assert (
        solved_grid_4x4.is_solved() is True
    ), "Solved 4x4 grid incorrectly marked as unsolved"
