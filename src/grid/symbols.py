"""Symbols Utils.

type Symbols: An ordered set of symbols used in puzzles.
    - The first symbol corresponds to the unknown symbol (symbol used to represent unsolved cells)
    - The next symbols correspond to the ordered set of possible values for each cell (e.g., 1-9 for 9x9 Sudoku)

"""

from typing import List

type Symbols = List[str]


def symbols_from_chars(*chars: str) -> Symbols:
    """Create a Symbols set from a list of characters."""
    if not chars:
        raise ValueError("No characters provided")
    if len(set(chars)) != len(chars):
        raise ValueError("Duplicate characters provided")
    if any(len(char) != 1 for char in chars):
        raise ValueError("All characters must be single characters")
    return list(chars)


def get_unknown_symbol(symbols: Symbols) -> str:
    """Get the unknown symbol from a Symbols set."""
    return symbols[0]


def get_possible_symbols(symbols: Symbols) -> List[str]:
    """Get the possible symbols from a Symbols set."""
    return symbols[1:]


def update_unknown_symbol(symbols: Symbols, new_unknown: str) -> Symbols:
    """Update the unknown symbol in a Symbols set."""
    if new_unknown == get_unknown_symbol(symbols):
        return symbols
    possible_symbols = get_possible_symbols(symbols)
    if new_unknown in possible_symbols:
        raise ValueError("New unknown symbol already exists in symbols")
    new_symbols: Symbols = [new_unknown] + possible_symbols
    return new_symbols


# Common Symbols
DEFAULT_4X4_SYMBOLS: Symbols = symbols_from_chars("0", "1", "2", "3", "4")
DEFAULT_9X9_SYMBOLS: Symbols = symbols_from_chars(
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
)
DEFAULT_16X16_SYMBOLS: Symbols = symbols_from_chars(
    "?", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"
)

if __name__ == "__main__":
    print("Default 4x4 Symbols:", DEFAULT_4X4_SYMBOLS)
    assert get_unknown_symbol(DEFAULT_4X4_SYMBOLS) == "0"
    custom_4x4_symbols = update_unknown_symbol(DEFAULT_4X4_SYMBOLS, "?")
    assert get_unknown_symbol(custom_4x4_symbols) == "?"
    print("Default 9x9 Symbols:", DEFAULT_9X9_SYMBOLS)
    assert get_unknown_symbol(DEFAULT_9X9_SYMBOLS) == "0"
    custom_9x9_symbols = update_unknown_symbol(DEFAULT_9X9_SYMBOLS, "?")
    assert get_unknown_symbol(custom_9x9_symbols) == "?"
    print("Default 16x16 Symbols:", DEFAULT_16X16_SYMBOLS)
    assert get_unknown_symbol(DEFAULT_16X16_SYMBOLS) == "?"
    custom_16x16_symbols = update_unknown_symbol(DEFAULT_16X16_SYMBOLS, "?")
    assert get_unknown_symbol(custom_16x16_symbols) == "?"
