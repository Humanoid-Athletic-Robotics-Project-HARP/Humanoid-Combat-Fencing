"""One tiny real YouTube search. Deselect with `-m 'not network'`."""

from __future__ import annotations

import pytest

from fencing.scrape.youtube import search


@pytest.mark.network
def test_tiny_real_search_returns_entries_with_ids():
    entries = search("ytsearch3:sabre fencing footwork drill")
    assert entries, "search returned nothing"
    assert all(e.get("id") and e.get("title") for e in entries)
