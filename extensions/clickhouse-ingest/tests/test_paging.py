"""at_smaller_pages: a failing listing is retried from page 1 at the next smaller size."""

import pytest

from spyre_clickhouse_ingest import paging


def _fetch(fail_over, calls):
    def fetch(size, attempts):
        calls.append((size, attempts))
        if size > fail_over:
            raise paging.ListingFailed(f"502 at {size}")
        return [size]

    return fetch


def test_the_first_size_that_works_is_returned_without_trying_smaller_ones():
    calls = []
    assert paging.at_smaller_pages(_fetch(100, calls)) == [100]
    assert calls == [(100, paging.STEP_ATTEMPTS)]


def test_it_steps_down_until_a_size_works():
    calls = []
    assert paging.at_smaller_pages(_fetch(30, calls)) == [25]
    assert [s for s, _ in calls] == [100, 50, 25]


def test_only_the_last_size_gets_the_long_retry():
    calls = []
    with pytest.raises(paging.ListingFailed):
        paging.at_smaller_pages(_fetch(0, calls))
    assert [a for _, a in calls] == [
        paging.STEP_ATTEMPTS,
        paging.STEP_ATTEMPTS,
        paging.STEP_ATTEMPTS,
        paging.LAST_SIZE_ATTEMPTS,
    ]


def test_the_error_of_the_last_size_is_the_one_raised():
    with pytest.raises(paging.ListingFailed, match="502 at 10"):
        paging.at_smaller_pages(_fetch(0, []))


def test_an_error_outside_failed_propagates_at_once():
    calls = []

    def fetch(size, attempts):
        calls.append(size)
        raise KeyError("not a listing failure")

    with pytest.raises(KeyError):
        paging.at_smaller_pages(fetch)
    assert calls == [100]


def test_failures_are_reported_for_each_size_that_failed():
    seen = []
    paging.at_smaller_pages(
        _fetch(25, []), on_failure=lambda size, err: seen.append(size)
    )
    assert seen == [100, 50]


def test_custom_sizes_and_attempts():
    calls = []
    with pytest.raises(paging.ListingFailed):
        paging.at_smaller_pages(
            _fetch(0, calls), sizes=(20, 5), step_attempts=1, last_attempts=3
        )
    assert calls == [(20, 1), (5, 3)]


def test_empty_sizes_is_a_programming_error():
    with pytest.raises(ValueError):
        paging.at_smaller_pages(lambda s, a: [], sizes=())
