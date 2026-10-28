"""The APIs vaex 4.19 calls that its own dependencies have since removed.

vaex-jupyter 0.9.0 is the last release, so these cannot be fixed upstream and
_patch_vaex_widgets puts them back. Each one is restored only when it is
actually missing, which is what these check: a shim that fires on a working
dependency would be worse than the breakage.
"""

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import pytest
import vaex.jupyter
import vaex.jupyter.utils

from detanalysis.analyzer import _patch_vaex_widgets


class Kernel:
    """An ipykernel 6 kernel, which can still pump a single message."""

    def do_one_iteration(self):
        pass


class Shell:
    def __init__(self, kernel):
        self.kernel = kernel


@pytest.fixture
def cm_without_get_cmap(monkeypatch):
    monkeypatch.delattr(matplotlib.cm, 'get_cmap', raising=False)


def test_a_removed_get_cmap_is_restored(cm_without_get_cmap):
    """Without it the heatmap image is never built and the widget is blank."""
    _patch_vaex_widgets()

    assert matplotlib.cm.get_cmap is plt.get_cmap


def test_a_working_get_cmap_is_left_alone(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(matplotlib.cm, 'get_cmap', sentinel, raising=False)

    _patch_vaex_widgets()

    assert matplotlib.cm.get_cmap is sentinel


def test_restoring_get_cmap_twice_changes_nothing(cm_without_get_cmap):
    _patch_vaex_widgets()
    _patch_vaex_widgets()

    assert matplotlib.cm.get_cmap is plt.get_cmap


@pytest.fixture
def tick(monkeypatch):
    """The real kernel_tick, under both names vaex reaches it by."""
    original = vaex.jupyter.utils.kernel_tick
    monkeypatch.setattr(vaex.jupyter, 'kernel_tick', original)
    monkeypatch.setattr(vaex.jupyter.utils, 'kernel_tick', original)
    return original


def test_a_kernel_that_cannot_tick_gets_a_no_op(tick, monkeypatch):
    """ipykernel 7 dropped do_one_iteration when the kernel became async."""
    monkeypatch.setattr(vaex.jupyter.utils, 'ipython', Shell(object()))

    _patch_vaex_widgets()

    assert vaex.jupyter.kernel_tick is not tick
    assert vaex.jupyter.kernel_tick() is None
    # view.py reads vaex.jupyter, utils holds the definition; keep them equal.
    assert vaex.jupyter.utils.kernel_tick is vaex.jupyter.kernel_tick


def test_a_kernel_that_can_still_tick_is_left_alone(tick, monkeypatch):
    monkeypatch.setattr(vaex.jupyter.utils, 'ipython', Shell(Kernel()))

    _patch_vaex_widgets()

    assert vaex.jupyter.kernel_tick is tick


def test_the_tick_is_left_alone_outside_a_kernel(tick, monkeypatch):
    """vaex already skips it there, and nothing would be gained by replacing it."""
    monkeypatch.setattr(vaex.jupyter.utils, 'ipython', None)

    _patch_vaex_widgets()

    assert vaex.jupyter.kernel_tick is tick
