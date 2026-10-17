import pytest

from continual_arc.grids import InvalidGrid, grid_hash, grids_equal, is_valid_grid, normalize_grid


def test_normalize_accepts_tuples_and_lists():
    assert normalize_grid(((1, 2), (3, 4))) == [[1, 2], [3, 4]]
    assert normalize_grid([[0]]) == [[0]]


def test_normalize_accepts_tolist_objects():
    class Arr:
        def tolist(self):
            return [[1, 2], [3, 4]]

    assert normalize_grid(Arr()) == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[]],
        [[1, 2], [3]],
        [[10]],
        [[-1]],
        [[True]],
        [[1.5]],
        "12",
        [[1] * 31],
        [[1]] * 31,
        None,
    ],
)
def test_normalize_rejects(bad):
    with pytest.raises(InvalidGrid):
        normalize_grid(bad)
    assert not is_valid_grid(bad)


def test_equality_and_hash():
    assert grids_equal([[1, 2]], ((1, 2),))
    assert not grids_equal([[1, 2]], [[2, 1]])
    assert not grids_equal([[1, 2]], "junk")
    assert grid_hash([[1, 2]]) == grid_hash([[1, 2]])
    assert grid_hash([[1, 2]]) != grid_hash([[2, 1]])
