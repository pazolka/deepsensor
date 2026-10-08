"""Tests for per-set gridded-context masking in :mod:`deepsensor.model.gridded_tnp`.

The bug these pin down
----------------------
``mask_nans_nps`` collapses the channel axis of each context set to **one** flag per point
(``B.any(arr.mask, axis=1)``). ``Task.op`` applies an op once per set, so the granularity of that
collapse is the *set*, not the whole context. Before this change ``GriddedTNP.modify_task`` never
called it, so the grid's ``np.ma.MaskedArray`` reached ``convert_to_tensor``, which drops the mask —
every NaN cell survived as a real observation of ``0.0``.

The fix has two halves, and both are asserted here:

* the grid is masked per set (``mask_nans_nps``), and
* each gridded set is encoded as one **density channel** — ``[mask, values * mask]``, the
  ``PrependDensityChannel`` convention — so a set whose own channels are incomplete keeps its cells
  (flagged missing) instead of poisoning the others.

These tests use synthetic rasters only: what is under test is the mask algebra, not the geography.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepsensor.torch  # noqa: F401,E402  (assigns the backend before any model is built)
import torch  # noqa: E402

from deepsensor.data import DataProcessor, TaskLoader  # noqa: E402
from deepsensor.model.gridded_tnp import (  # noqa: E402
    GriddedTNP,
    _count_gridded_context_sets,
    convert_task_to_gridded_tnp_args,
)

LAT = np.linspace(50.0, 55.0, 24)
LON = np.linspace(-5.0, 1.5, 24)
TIMES = pd.date_range("2000-01-01", periods=4, freq="MS")
N_CELLS = len(LAT) * len(LON)

# Two disjoint-ish NaN footprints, so the per-set masks, their union, and the intersection are all
# different numbers. A test that only ever sees the union cannot tell per-set from per-point.
FOOTPRINT_A = np.zeros((len(LAT), len(LON)), dtype=bool)
FOOTPRINT_A[3:5, 6:10] = True  # 8 cells
FOOTPRINT_B = np.zeros((len(LAT), len(LON)), dtype=bool)
FOOTPRINT_B[3:5, 8:14] = True  # 12 cells; overlaps A on 4
N_NAN_A = int(FOOTPRINT_A.sum())  # 8
N_NAN_B = int(FOOTPRINT_B.sum())  # 12
N_NAN_UNION = int((FOOTPRINT_A | FOOTPRINT_B).sum())  # 16


def _raster(vars, footprint, lon=LON):
    """A ``time x lat x lon`` dataset whose ``vars`` are NaN wherever ``footprint`` is True."""
    rng = np.random.default_rng(0)
    data = {}
    for name in vars:
        values = rng.normal(size=(len(TIMES), len(LAT), len(lon)))
        values[:, footprint] = np.nan
        data[name] = (("time", "lat", "lon"), values)
    return xr.Dataset(data, coords={"time": TIMES, "lat": LAT, "lon": lon})


def _stations(processor):
    rng = np.random.default_rng(1)
    n = 40
    frame = pd.DataFrame(
        {
            "time": np.repeat(TIMES, n),
            "lat": np.tile(rng.uniform(50.0, 55.0, n), len(TIMES)),
            "lon": np.tile(rng.uniform(-5.0, 1.5, n), len(TIMES)),
            "gwsa": rng.normal(size=len(TIMES) * n),
        }
    ).set_index(["time", "lat", "lon"])
    return frame, processor(frame, method="mean_std")


def _build(n_gridded=2):
    """A ``DataProcessor``, ``TaskLoader`` and the processed sets for the synthetic rasters.

    ``n_gridded=2`` gives ``[covariates(3), abstraction(2), stations(1)]``; ``1`` merges the two
    rasters into a single 5-channel set, which is the pre-split layout.
    """
    processor = DataProcessor(x1_name="lat", x2_name="lon", x1_map=(0.0, 1.0), x2_map=(0.0, 1.0))
    _, stations_proc = _stations(processor)

    sets = [
        processor(_raster(["a1", "a2", "a3"], FOOTPRINT_A), method="mean_std"),
        processor(_raster(["b1", "b2"], FOOTPRINT_B), method="mean_std"),
    ]
    if n_gridded == 1:
        sets = [xr.merge(sets)]

    loader = TaskLoader(
        context=[*sets, stations_proc],
        target=stations_proc,
        links=[(len(sets), 0)],
    )
    return processor, loader, sets, stations_proc


# The repo's own architecture arguments (``build_model`` in the main pipeline). The model is
# indifferent to the geography, but *not* to ``points_per_dim`` / ``window_sizes``: those have to
# be given, or they are inferred from the grid density and the Swin padding breaks.
MODEL_KWARGS = dict(
    model_variant="ootg",
    grid_encoder_type="pseudo-token",
    num_fourier=10,
    p_basis_dropout=0.5,
    window_sizes=(4, 4),
    shift_sizes=(1, 1),
    points_per_dim=(64, 80),
    top_k_ctot=12,
    init_lengthscale=0.03125,
    d_model=64,
    num_heads=4,
    num_layers=4,
    norm_first=True,
    verbose=False,
    likelihood="het",
)


def _task(loader):
    return loader(TIMES[0], context_sampling=["all"] * len(loader.context), target_sampling="all")


def _mlp_in_dim(mlp):
    """Input width of an ``MLP``, read off its first ``Linear``."""
    for module in mlp.modules():
        if isinstance(module, torch.nn.Linear):
            return module.weight.shape[1]
    raise AssertionError("no Linear layer found")


@pytest.fixture(scope="module")
def split():
    """The two-gridded-set layout, built once."""
    return _build(n_gridded=2)


@pytest.fixture(scope="module")
def model(split):
    processor, loader, _, _ = split
    return GriddedTNP(processor, loader, **MODEL_KWARGS)


# ---------------------------------------------------------------------------------------------
# The mask is per set, and that is load-bearing
# ---------------------------------------------------------------------------------------------


def test_grid_mask_collapses_within_each_set(split):
    """One density channel per set, sized by that set's own footprint.

    This is the assertion that would have caught the original bug: a single collapsed context
    yields the *union* of the two footprints, so set 0 would lose 16 cells instead of 8.
    """
    _, loader, _, _ = split
    task = GriddedTNP.modify_task(_task(loader))
    yc_grid = convert_task_to_gridded_tnp_args(task, model_variant="ootg", dim_x=2)[3]

    # Blocks are [mask(1), values(3)] and [mask(1), values(2)].
    mask_a = yc_grid[..., 0]
    mask_b = yc_grid[..., 4]

    assert mask_a.shape == mask_b.shape == (1, len(LAT), len(LON))
    assert int(mask_a.sum()) == N_CELLS - N_NAN_A
    assert int(mask_b.sum()) == N_CELLS - N_NAN_B

    # The per-point collapse the split replaced, stated as a number it must not equal.
    assert int(mask_a.sum()) != N_CELLS - N_NAN_UNION
    assert int(mask_b.sum()) != N_CELLS - N_NAN_UNION


def test_missing_in_one_set_leaves_the_other_intact(split):
    """A cell NaN only in the abstraction set keeps its covariate values *and* its flag."""
    _, loader, _, _ = split
    task = GriddedTNP.modify_task(_task(loader))
    yc_grid = convert_task_to_gridded_tnp_args(task, model_variant="ootg", dim_x=2)[3]

    # A cell in B's footprint but outside A's.
    i, j = next(
        (i, j)
        for i in range(len(LAT))
        for j in range(len(LON))
        if FOOTPRINT_B[i, j] and not FOOTPRINT_A[i, j]
    )
    assert float(yc_grid[0, i, j, 0]) == 1.0  # set A present
    assert torch.isfinite(yc_grid[0, i, j, 1:4]).all()  # ... with real values
    assert float(yc_grid[0, i, j, 4]) == 0.0  # set B flagged missing
    assert torch.all(yc_grid[0, i, j, 5:7] == 0.0)  # ... and zero-filled

    # And the converse for a cell in A's footprint only.
    i, j = next(
        (i, j)
        for i in range(len(LAT))
        for j in range(len(LON))
        if FOOTPRINT_A[i, j] and not FOOTPRINT_B[i, j]
    )
    assert float(yc_grid[0, i, j, 0]) == 0.0
    assert float(yc_grid[0, i, j, 4]) == 1.0


def test_density_channels_are_binary_and_values_never_nan(split):
    """``NaN`` must not reach the encoder: a real NaN would poison the MLP, zeros do not."""
    _, loader, _, _ = split
    task = GriddedTNP.modify_task(_task(loader))
    yc_grid = convert_task_to_gridded_tnp_args(task, model_variant="ootg", dim_x=2)[3]

    assert torch.isfinite(yc_grid).all()
    for channel in (0, 4):
        assert set(torch.unique(yc_grid[..., channel]).tolist()) <= {0.0, 1.0}
    # A missing cell is zero everywhere in its block, so ``values * mask`` holds no stale data.
    assert torch.all(yc_grid[..., 1:4][yc_grid[..., 0] == 0] == 0.0)


# ---------------------------------------------------------------------------------------------
# Sizing: the density channels widen the grid encoder
# ---------------------------------------------------------------------------------------------


def test_grid_encoder_width_counts_one_channel_per_set(split, model):
    """``in_dim == dim_yc_grid + n_gridded``, and the built block is the same width."""
    _, loader, sets, _ = split
    assert tuple(model.config["dim_yc"]) == (3, 2, 1)
    assert model.config["num_gridded_contexts"] == 2
    assert _mlp_in_dim(model.model.encoder.y_grid_encoder) == 5 + 2

    task = GriddedTNP.modify_task(_task(loader))
    yc_grid = convert_task_to_gridded_tnp_args(task, model_variant="ootg", dim_x=2)[3]
    assert yc_grid.shape[-1] == _mlp_in_dim(model.model.encoder.y_grid_encoder)


def test_single_gridded_set_gets_one_density_channel():
    """The pre-split layout still works, and pays one channel rather than two."""
    processor, loader, _, _ = _build(n_gridded=1)
    model = GriddedTNP(processor, loader, **MODEL_KWARGS)
    assert tuple(model.config["dim_yc"]) == (5, 1)
    assert model.config["num_gridded_contexts"] == 1
    assert _mlp_in_dim(model.model.encoder.y_grid_encoder) == 5 + 1


def test_point_encoder_is_not_widened(split, model):
    """Only the grid gains density channels; the station block keeps its ``+1`` slot."""
    assert _mlp_in_dim(model.model.encoder.y_encoder) == 1 + 1


def test_convnp_pays_one_density_channel_per_set(split):
    """The convention this change copies is ``neuralprocesses``' own.

    ``_convgnp_init_dims`` sizes the conv input as ``sum(dim_yc) + len(dim_yc)`` — one density
    channel per context set, "``len(dim_yc)`` is equal to the number of density channels". So
    ``GriddedTNP``'s ``dim_yc_grid + num_gridded_contexts`` is that same rule, and splitting the
    context widens ConvNP's conv input by one with no library change at all.
    """
    from deepsensor.model import ConvNP

    processor, loader, _, _ = split
    model = ConvNP(processor, loader, unet_channels=(8, 8, 8), verbose=False)

    dim_yc = tuple(model.config["dim_yc"])
    assert dim_yc == (3, 2, 1)
    first_conv = next(m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d))
    assert first_conv.in_channels == sum(dim_yc) + len(dim_yc) == 6 + 3

    # The unsplit layout pays ``len((5, 1)) = 2``; the split pays 3. Same sum, one more channel.
    _, unsplit, _, _ = _build(n_gridded=1)
    unsplit_model = ConvNP(processor, unsplit, unet_channels=(8, 8, 8), verbose=False)
    assert tuple(unsplit_model.config["dim_yc"]) == (5, 1)
    assert sum((5, 1)) + 2 == 8 != first_conv.in_channels


def test_args_tuple_keeps_its_arity(split):
    """``xt`` stays last — ``logpdf`` reads ``model_args[-1]``, so a sixth element would be read
    as the target locations."""
    _, loader, _, _ = split
    task = GriddedTNP.modify_task(_task(loader))
    args = convert_task_to_gridded_tnp_args(task, model_variant="ootg", dim_x=2)
    assert len(args) == 5
    assert args[-1] is not None and args[-1].shape[-1] == 2


# ---------------------------------------------------------------------------------------------
# Inference and persistence
# ---------------------------------------------------------------------------------------------


def test_forward_and_logpdf_run_on_a_split_context(split, model):
    _, loader, _, _ = split
    task = GriddedTNP.modify_task(_task(loader))

    with torch.no_grad():
        dist = model(task)
        logpdf = model.logpdf(task)

    assert torch.isfinite(dist.mean).all()
    assert torch.isfinite(torch.as_tensor(logpdf)).all()


def test_num_gridded_contexts_survives_a_save_load_roundtrip(split, model, tmp_path):
    """``load`` rebuilds via ``construct_gridded_tnp(**config)`` and has no ``TaskLoader``.

    If the count were derived but not persisted it would default to 1, the grid encoder would be
    built one channel narrow, and ``load_state_dict(strict=True)`` would fail on shape.
    """
    model.save(str(tmp_path))

    with (tmp_path / "model_config.json").open() as handle:
        assert json.load(handle)["num_gridded_contexts"] == 2

    processor, loader, _, _ = split
    reloaded = GriddedTNP(processor, loader, str(tmp_path))
    assert reloaded.config["num_gridded_contexts"] == 2
    assert _mlp_in_dim(reloaded.model.encoder.y_grid_encoder) == 5 + 2
    assert reloaded.model.state_dict().keys() == model.model.state_dict().keys()


# ---------------------------------------------------------------------------------------------
# Plumbing: counting sets, and rejecting sets that disagree about the grid
# ---------------------------------------------------------------------------------------------


def test_count_gridded_context_sets(split):
    """xarray sets are counted, and counting stops at the first non-xarray set."""
    _, loader, _, _ = split
    assert _count_gridded_context_sets(loader) == 2

    _, single, _, _ = _build(n_gridded=1)
    assert _count_gridded_context_sets(single) == 1

    processor = DataProcessor(x1_name="lat", x2_name="lon", x1_map=(0.0, 1.0), x2_map=(0.0, 1.0))
    _, stations_proc = _stations(processor)
    point_only = TaskLoader(context=stations_proc, target=stations_proc)
    assert _count_gridded_context_sets(point_only) == 0


def test_point_only_loader_builds_a_gridded_model():
    """``model_variant='gridded'`` never touches the grid encoder, so zero sets must not raise."""
    processor = DataProcessor(x1_name="lat", x2_name="lon", x1_map=(0.0, 1.0), x2_map=(0.0, 1.0))
    _, stations_proc = _stations(processor)
    loader = TaskLoader(context=stations_proc, target=stations_proc)
    model = GriddedTNP(processor, loader, model_variant="gridded", points_per_dim=(64, 80), verbose=False)
    assert model.config["num_gridded_contexts"] == 0


def test_ootg_without_a_grid_raises():
    """The ootg encoder is *built* around a grid, so zero sets is a configuration error."""
    processor = DataProcessor(x1_name="lat", x2_name="lon", x1_map=(0.0, 1.0), x2_map=(0.0, 1.0))
    _, stations_proc = _stations(processor)
    loader = TaskLoader(context=stations_proc, target=stations_proc)
    with pytest.raises(ValueError, match="at least one gridded context set"):
        GriddedTNP(processor, loader, **MODEL_KWARGS)


def test_gridded_sets_on_different_coordinates_are_rejected():
    """Silently using the first set's grid would encode the second set's values at wrong cells."""
    processor = DataProcessor(x1_name="lat", x2_name="lon", x1_map=(0.0, 1.0), x2_map=(0.0, 1.0))
    _, stations_proc = _stations(processor)
    shifted = processor(_raster(["b1", "b2"], FOOTPRINT_B, lon=LON + 0.25), method="mean_std")
    loader = TaskLoader(
        context=[processor(_raster(["a1", "a2", "a3"], FOOTPRINT_A), method="mean_std"), shifted, stations_proc],
        target=stations_proc,
        links=[(2, 0)],
    )
    task = GriddedTNP.modify_task(_task(loader))
    with pytest.raises(ValueError, match="same grid"):
        convert_task_to_gridded_tnp_args(task, model_variant="ootg", dim_x=2)
