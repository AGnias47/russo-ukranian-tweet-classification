#!/usr/bin/env python3

from utils.labels import int_to_label, label_to_int

import pytest


def test_int_to_label():
    assert int_to_label(1) == "pu"
    assert int_to_label(2) == "pr"
    assert int_to_label(3) == "au"
    assert int_to_label(4) == "ar"
    assert int_to_label(5) == "n"
    assert int_to_label(6) == "us"
    assert int_to_label(7) == "u"


def test_label_to_int():
    assert label_to_int("pu") == 1
    assert label_to_int("pr") == 2
    assert label_to_int("au") == 3
    assert label_to_int("ar") == 4
    assert label_to_int("n") == 5
    assert label_to_int("us") == 6
    assert label_to_int("u") == 7


if __name__ == "__main__":
    pytest.run()
