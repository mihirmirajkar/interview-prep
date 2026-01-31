from __future__ import annotations

from typing import Sequence, Set
import unittest


def maximize_unique_characters(words: Sequence[str]) -> Set[str]:
    """Return a subset of `words` that maximizes unique characters with no repeats.

    Rules:
    - You may select a subset of the given strings.
    - No character may appear more than once across the entire selected subset.
    - Any word that contains a duplicate character within itself is invalid and cannot be selected.

    Output:
    - Return the selected subset as a `Set[str]`.

    Errors:
    - Raise TypeError if `words` is not a sequence of strings (i.e., any element is not `str`).

    Notes:
    - This is an NP-hard optimization problem in general.
    - Your goal is to implement a correct solution that passes the tests.
    """

    # TODO: Implement.
    raise NotImplementedError


class TestMaximizeUniqueCharacters(unittest.TestCase):
    def _score(self, chosen: set[str]) -> int:
        used: set[str] = set()
        for w in chosen:
            used |= set(w)
        return len(used)

    def _assert_valid(self, words: list[str], chosen: set[str]) -> None:
        self.assertTrue(chosen.issubset(set(words)))

        used: set[str] = set()
        for w in chosen:
            # word must have no internal duplicates
            self.assertEqual(len(w), len(set(w)), msg=f"Word {w!r} has internal duplicates")
            # global uniqueness across chosen words
            for ch in w:
                self.assertNotIn(ch, used, msg=f"Character {ch!r} is reused")
                used.add(ch)

    def test_unique_optimum_basic(self):
        # Unique optimum is {"ab", "cd", "efg"} with 7 distinct chars.
        words = ["ab", "cd", "efg", "a", "cdef"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        self.assertEqual({"ab", "cd", "efg"}, chosen)
        self.assertEqual(7, self._score(chosen))

    def test_reject_internal_duplicate_words(self):
        # "aa" is invalid and cannot be selected.
        # Unique optimum is {"bc", "def"} => 5 distinct chars.
        words = ["aa", "bc", "def"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        self.assertEqual({"bc", "def"}, chosen)
        self.assertEqual(5, self._score(chosen))

    def test_digits_supported(self):
        # Strings may be digits; "1001" is invalid due to repeated digits.
        # Unique optimum is {"2357", "19", "04"} => 8 distinct chars.
        words = ["2357", "19", "04", "1001"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        self.assertEqual({"2357", "19", "04"}, chosen)
        self.assertEqual(8, self._score(chosen))

    def test_empty_input(self):
        words: list[str] = []

        chosen = maximize_unique_characters(words)
        self.assertEqual(set(), chosen)

    def test_empty_string_word(self):
        # The empty string is valid (uses 0 characters). It should not break anything.
        # Unique optimum is {"ab", "cd"}; including "" is allowed but doesn't increase score.
        words = ["", "ab", "cd", "a"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        # We assert the optimal set excludes "a" (conflicts with "ab"), and is uniquely optimal.
        # Whether "" is included or not doesn't change score, so we avoid tie by not requiring it.
        self.assertEqual({"ab", "cd"}, {w for w in chosen if w != ""})
        self.assertEqual(4, self._score(chosen))

    def test_all_conflict_best_single_word(self):
        # "abc" is uniquely optimal; all other options conflict in a way that prevents ties.
        words = ["abc", "ab", "bc", "ac"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        self.assertEqual({"abc"}, chosen)
        self.assertEqual(3, self._score(chosen))

    def test_duplicate_strings_in_input(self):
        # Duplicates in input; output is a set anyway.
        words = ["ab", "ab", "cd"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        self.assertEqual({"ab", "cd"}, chosen)
        self.assertEqual(4, self._score(chosen))

    def test_larger_unique_optimum(self):
        # Unique optimum: {"ab","cd","ef","gh","ijkl"} => 12 distinct chars.
        words = ["ab", "cd", "ef", "gh", "ijkl", "a", "c", "e", "g"]

        chosen = maximize_unique_characters(words)
        self._assert_valid(words, chosen)
        self.assertEqual({"ab", "cd", "ef", "gh", "ijkl"}, chosen)
        self.assertEqual(12, self._score(chosen))

    def test_non_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            maximize_unique_characters(["ab", 123])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()