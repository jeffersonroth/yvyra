"""Classic Solver."""

import numpy as np


def solve_boxes_generic(grid: np.ndarray):
    """Generic solver for boxes."""
    n = grid.shape[0]
    box_size = int(np.sqrt(n))  # e.g., 3 for a 9x9 grid
    full_mask = (1 << n) - 1

    # 1. Reshape into a 4D view: (box_rows, box_cols, cell_rows, cell_cols)
    # For a 9x9, this becomes (3, 3, 3, 3)
    view = grid.reshape(box_size, box_size, box_size, box_size)

    # 2. Identify resolved cells (powers of 2)
    is_resolved = (view > 0) & ((view & (view - 1)) == 0)

    # 3. Combine resolved bits within each box
    # We reduce across the last two axes (the cells inside the box)
    box_resolved = np.bitwise_or.reduce(
        np.where(is_resolved, view, 0), axis=(2, 3), keepdims=True
    )

    # 4. Apply the "But-Not" logic
    # This updates the original 'grid' because 'view' is a view, not a copy
    view[:] = np.where(
        is_resolved, view, np.where(view == 0, full_mask, view) & ~box_resolved
    )


def solve_step_generic(grid: np.ndarray, axis: int):
    """Generic solver for rows (axis=1) or columns (axis=0)."""
    n = grid.shape[0]
    # If 4x4, mask is 1111 (15). If 9x9, mask is 111111111 (511)
    full_mask = (1 << n) - 1

    # 1. Identify cells that are ALREADY solved (exactly one bit set)
    # These are our constraints
    is_resolved = (grid > 0) & ((grid & (grid - 1)) == 0)

    # 2. Combine all resolved bits in the row/column using bitwise OR
    resolved_bits = np.bitwise_or.reduce(
        np.where(is_resolved, grid, 0), axis=axis, keepdims=True
    )

    # 3. Apply logic:
    # IF cell is already resolved: Keep it.
    # ELIF cell is 0: Give it all possibilities MINUS resolved bits.
    # ELSE: Take current candidates MINUS resolved bits.
    grid[:] = np.where(
        is_resolved, grid, np.where(grid == 0, full_mask, grid) & ~resolved_bits
    )


def solve_hidden_singles(group):
    """Finds hidden singles in a group (row, col, or box)."""
    # Check each bit from 0 to n-1 (for 4x4, that's bits 1, 2, 4, 8)
    n = len(group)
    for i in range(n):
        bit = 1 << i
        # Find which cells in this group could potentially be this bit
        mask = (group & bit) > 0

        # If this bit ONLY appears in one cell's candidate list...
        if np.sum(mask) == 1:
            idx = np.where(mask)[0][0]
            # ...and that cell isn't already solved...
            if bin(group[idx]).count("1") > 1:
                # ...then that cell MUST be this bit!
                group[idx] = bit


def solve_naked_pairs(group: np.ndarray):
    """
    If two cells have the same two bits and nothing else,
    remove those bits from all other cells in the group.
    """
    for i in range(len(group)):
        cell_val = group[i]
        # Check if the cell has exactly TWO bits set
        # bin(cell_val).count('1') == 2 is the easiest way in Python
        if bin(cell_val).count("1") == 2:
            # Look for a match in the rest of the group
            for j in range(i + 1, len(group)):
                if group[j] == cell_val:
                    # Found a Naked Pair!
                    # Now remove these two bits from everyone else
                    mask = ~cell_val
                    for k in range(len(group)):
                        if k != i and k != j:
                            group[k] &= mask

def solve_phistomephel_ring(grid: np.ndarray):
    # if grid.shape != (9, 9):
    #     return # Only applies to 9x9
        
    # # 1. Define the coordinates
    # # Outer 16: The four 2x2 corner blocks
    # outer_coords = [
    #     (0,0), (0,1), (1,0), (1,1), # Top-Left
    #     (0,7), (0,8), (1,7), (1,8), # Top-Right
    #     (7,0), (7,1), (8,0), (8,1), # Bottom-Left
    #     (7,7), (7,8), (8,7), (8,8)  # Bottom-Right
    # ]
    
    # # Inner 16: The ring surrounding the very center
    # # Rows 2 and 6 (cols 2-6), and Cols 2 and 6 (rows 2-6)
    # inner_coords = [
    #     (2,2), (2,3), (2,4), (2,5), (2,6),
    #     (6,2), (6,3), (6,4), (6,5), (6,6),
    #     (3,2), (4,2), (5,2),
    #     (3,6), (4,6), (5,6)
    # ]
    
    # def get_set_info(coords):
    #     cells = [grid[r, c] for r, c in coords]
    #     fixed = [c for c in cells if bin(c).count('1') == 1]
    #     return cells, fixed

    # outer_cells, outer_fixed = get_set_info(outer_coords)
    # inner_cells, inner_fixed = get_set_info(inner_coords)

    # # 2. Logic: If the total count of a digit is known in one ring, 
    # # it must match in the other.
    # for i in range(9):
    #     bit = 1 << i
    #     count_outer = outer_fixed.count(bit)
    #     count_inner = inner_fixed.count(bit)
        
    #     # If we found all instances of a digit in the outer ring (e.g., two 5s),
    #     # and the inner ring already has two 5s, remove 5 from all other inner candidates.
    #     # Note: This requires knowing the total count, which is usually found
    #     # by checking how many cells in the ring *could* still be that bit.
        
    #     # Simple version: if count is equal and maxed out, eliminate
    #     # (This is a complex set-theory deduction, often used in 'Set Equivalent Theory')
    pass


def solve_grid(grid):
    """Solves the Sudoku grid using a combination of techniques."""
    n = grid.shape[0]
    box_size = int(np.sqrt(n))

    while True:
        prev_state = grid.copy()

        # Phase 1: Material Nonimplication (Naked Singles)
        solve_step_generic(grid, axis=1)  # Rows
        solve_step_generic(grid, axis=0)  # Cols
        solve_boxes_generic(grid)  # Boxes

        # Phase 2: Hidden Singles (Uniqueness)
        # Rows and Columns
        for i in range(n):
            solve_hidden_singles(grid[i, :])  # Row i
            solve_hidden_singles(grid[:, i])  # Col i

        # Boxes
        # We create a view where each row is actually one subgrid box
        boxes_view = grid.reshape(box_size, box_size, box_size, box_size)
        boxes_view = boxes_view.transpose(0, 2, 1, 3).reshape(n, n)
        for box in boxes_view:
            solve_hidden_singles(box)

        # Phase 3: Naked Pairs
        # Rows and Columns
        for i in range(n):
            solve_naked_pairs(grid[i, :])  # Row i
            solve_naked_pairs(grid[:, i])  # Col i

        # Boxes
        # We create a view where each row is actually one subgrid box
        boxes_view = grid.reshape(box_size, box_size, box_size, box_size)
        boxes_view = boxes_view.transpose(0, 2, 1, 3).reshape(n, n)
        for box in boxes_view:
            solve_naked_pairs(box)

        if np.array_equal(prev_state, grid):
            print("Stable state reached.")
            break
        print(np.vectorize(np.binary_repr)(prev_state, width=prev_state.shape[1]))

    return grid


def is_solved(grid):
    """Checks if the Sudoku grid is solved."""
    # A grid is solved if every cell is a power of 2 (exactly one bit set)
    return np.all((grid > 0) & ((grid & (grid - 1)) == 0))


def is_invalid(grid):
    """Checks if the Sudoku grid is invalid."""
    # A grid is invalid if any cell has 0 possibilities
    return np.any(grid == 0)


def solve_recursive(grid):
    """Solves the Sudoku grid using backtracking."""
    # 1. Apply all your logic first to narrow down the search
    solve_grid(grid)  # This is your existing while loop function

    if is_invalid(grid):
        return None
    if is_solved(grid):
        return grid

    # 2. Find the cell with the fewest candidates (but > 1) to guess
    # We'll just take the first one for simplicity
    unsolved_indices = np.argwhere((grid & (grid - 1)) != 0)
    if len(unsolved_indices) == 0:
        return grid

    r, c = unsolved_indices[0]
    candidates = [1 << i for i in range(grid.shape[0]) if (grid[r, c] & (1 << i))]

    for guess in candidates:
        grid_copy = grid.copy()
        grid_copy[r, c] = guess

        result = solve_recursive(grid_copy)
        if result is not None:
            grid[:] = result[:]  # Update original grid with the solution
            return grid

    return None


if __name__ == "__main__":
    from grid.utils import char_grid_to_solver_grid, row_major_to_grid
    from grid.symbols import DEFAULT_9X9_SYMBOLS, DEFAULT_4X4_SYMBOLS

    # 4x4
    ROW_MAJOR = "1000000400200300"
    _grid = row_major_to_grid(ROW_MAJOR, (4, 4))
    grid = char_grid_to_solver_grid(_grid, DEFAULT_4X4_SYMBOLS)
    solve_recursive(grid)
    print(np.vectorize(np.binary_repr)(grid, width=grid.shape[1]))

    # Check if the grid is solved
    if is_solved(grid):
        print("4x4 Sudoku solved successfully!")
    else:
        print("4x4 Sudoku could not be solved.")

    # 9x9
    ROW_MAJOR = "467100805912835607085647192296351470708920351531408926073064510624519783159783064"
    ROW_MAJOR_SOLVED = "467192835912835647385647192296351478748926351531478926873264519624519783159783264"
    _grid = row_major_to_grid(ROW_MAJOR, (9, 9))
    grid = char_grid_to_solver_grid(_grid, DEFAULT_9X9_SYMBOLS)
    solve_recursive(grid)
    print(np.vectorize(np.binary_repr)(grid, width=grid.shape[1]))

    # Check if the grid is solved
    if is_solved(grid):
        print("9x9 Sudoku solved successfully!")
    else:
        print("9x9 Sudoku could not be solved.")

    _grid_solved = row_major_to_grid(ROW_MAJOR_SOLVED, (9, 9))
    grid_solved = char_grid_to_solver_grid(_grid_solved, DEFAULT_9X9_SYMBOLS)
    assert np.array_equal(grid, grid_solved), "9x9 Sudoku did not solve correctly."
