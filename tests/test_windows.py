from adapter3r import reference_window


def test_reference_window_is_bounded_unique_and_legacy_compatible():
    result = reference_window(6, 2)
    assert result.tolist() == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]
    assert all(len(set(row)) == 5 for row in result.tolist())
    assert reference_window(2, 2).tolist() == [[0], [1]]


def test_reference_window_without_self():
    result = reference_window(4, 1, include_self=False)
    assert result.tolist() == [[1, 2], [0, 2], [1, 3], [2, 1]]
    assert all(
        index not in row for index, row in enumerate(result.tolist())
    )
