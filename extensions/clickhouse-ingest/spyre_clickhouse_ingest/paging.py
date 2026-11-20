"""Listing a GitHub resource at a smaller page size when a large page keeps failing.

GitHub answers 502 for a big page of a many-job run however often it is asked, while a
smaller page succeeds, so retrying the same request cannot help.
"""

from typing import Any, Callable, Sequence

JOB_PAGE_SIZES = (100, 50, 25, 10)
STEP_ATTEMPTS = 2
LAST_SIZE_ATTEMPTS = 7


class ListingFailed(Exception):
    """A listing failed through every retry at one page size."""


def at_smaller_pages(
    fetch: Callable[[int, int], Any],
    sizes: Sequence[int] = JOB_PAGE_SIZES,
    *,
    step_attempts: int = STEP_ATTEMPTS,
    last_attempts: int = LAST_SIZE_ATTEMPTS,
    failed: tuple = (ListingFailed,),
    on_failure: Callable[[int, BaseException], None] | None = None,
) -> Any:
    """fetch(size, attempts) at each size in turn; the first result wins.

    fetch lists everything from page 1, so pages of different sizes are never combined. Only
    the last size gets last_attempts. An exception outside `failed` propagates at once.
    """
    if not sizes:
        raise ValueError("sizes is empty")
    for i, size in enumerate(sizes):
        last = i == len(sizes) - 1
        try:
            return fetch(size, last_attempts if last else step_attempts)
        except failed as err:
            if on_failure:
                on_failure(size, err)
            if last:
                raise
    raise AssertionError("unreachable")
