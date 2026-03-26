"""GriddedTNP model wrapper for DeepSensor.

This module wraps the gridded-tnp package's GriddedTNP model to work with DeepSensor's
data structures and prediction interface.
"""

from typing import Union, List, Optional, Tuple, Dict, Literal
import warnings
import json
import os.path
import torch


import numpy as np
import lab as B
from plum import dispatch

from deepsensor import backend
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task
from deepsensor.model.model import DeepSensorModel


def _compute_grid_range_from_task_loader(
    task_loader: TaskLoader, dim_x: int = 2
) -> Tuple[Tuple[float, float], ...]:
    """Compute the exact grid range from TaskLoader data with adaptive buffer.
    
    Extracts coordinate bounds from all context and target data, and adds a small
    buffer proportional to the data resolution to ensure full coverage.
    
    Args:
        task_loader: TaskLoader containing context and target data
        dim_x: Spatial dimension (e.g., 2 for lat/lon)
        
    Returns:
        Tuple of (min, max) bounds for each spatial dimension
    """
    import pandas as pd
    import xarray as xr
    
    # Collect all coordinate values from context and target data
    x1_coords = []
    x2_coords = []
    
    def extract_coords(data):
        """Extract x1 and x2 coordinates from xarray or pandas data."""
        if isinstance(data, (tuple, list)):
            for item in data:
                extract_coords(item)
        elif isinstance(data, xr.DataArray):
            if "x1" in data.coords and "x2" in data.coords:
                x1_coords.append(data.coords["x1"].values)
                x2_coords.append(data.coords["x2"].values)
        elif isinstance(data, xr.Dataset):
            # Extract from first variable
            for var in data.data_vars:
                da = data[var]
                if "x1" in da.coords and "x2" in da.coords:
                    x1_coords.append(da.coords["x1"].values)
                    x2_coords.append(da.coords["x2"].values)
                break
        elif isinstance(data, pd.DataFrame):
            # Get x1, x2 from index
            if "x1" in data.index.names and "x2" in data.index.names:
                df_reset = data.reset_index()
                x1_coords.append(df_reset["x1"].values)
                x2_coords.append(df_reset["x2"].values)
        elif isinstance(data, pd.Series):
            if "x1" in data.index.names and "x2" in data.index.names:
                series_reset = data.reset_index()
                x1_coords.append(series_reset["x1"].values)
                x2_coords.append(series_reset["x2"].values)
    
    # Extract from context and target
    extract_coords(task_loader.context)
    extract_coords(task_loader.target)
    
    if not x1_coords or not x2_coords:
        # Fallback to normalized [0, 1] range if no data found
        warnings.warn(
            "Could not extract coordinates from task_loader data. "
            "Using default [0, 1] range for grid_range.",
            UserWarning
        )
        return tuple((0.0, 1.0) for _ in range(dim_x))
    
    # Compute bounds
    x1_all = np.concatenate([np.ravel(x) for x in x1_coords])
    x2_all = np.concatenate([np.ravel(x) for x in x2_coords])
    
    x1_min, x1_max = np.nanmin(x1_all), np.nanmax(x1_all)
    x2_min, x2_max = np.nanmin(x2_all), np.nanmax(x2_all)
    
    # Compute adaptive buffer: 2% of the range or 0.02, whichever is smaller
    # This ensures we don't expand too much for small domains
    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
    
    x1_buffer = min(0.02 * x1_range, 0.02) if x1_range > 0 else 0.02
    x2_buffer = min(0.02 * x2_range, 0.02) if x2_range > 0 else 0.02
    
    # Apply buffer
    x1_bounds = (float(x1_min - x1_buffer), float(x1_max + x1_buffer))
    x2_bounds = (float(x2_min - x2_buffer), float(x2_max + x2_buffer))
    
    if dim_x == 2:
        return (x1_bounds, x2_bounds)
    else:
        # For higher dimensions, use same bounds for all
        warnings.warn(
            f"dim_x={dim_x} > 2: using x1 bounds for all dimensions. "
            "Consider providing explicit grid_range for higher-dimensional data.",
            UserWarning
        )
        return tuple(x1_bounds for _ in range(dim_x))


class GriddedTNP(DeepSensorModel):
    """A Gridded Transformer Neural Process (GriddedTNP) model.

    Wraps around the ``gridded-tnp`` package (``tnp`` module) to construct an
    approximately translation-equivariant GriddedTNP model based on the
    ``GriddedATETNP`` / ``OOTGGriddedATETNP`` family.
    See: https://github.com/cambridge-mlg/gridded-tnp

    **Key Features:**

    - **Approximate Translation Equivariance (default)**: Uses the ATETNP architecture.
    - **Swin + TE Attention**: Uses gridded TE self-attention and TE cross-attention decoding.
    - **Configurable Grid Encoder**:
        - ``'pseudo-token'`` (default): pseudo-token TE grid encoders
        - ``'kernel-interp'``: SetConv/OOTGSetConv kernel interpolation grid encoders
    - **Two Model Variants**:
        - ``'gridded'`` (default): point observations → internal grid → transformer → predictions
        - ``'ootg'`` (Off-The-Grid): point + explicit gridded context → transformer → predictions
    - **Efficient**: Processes spatial data on grids rather than all pairwise point interactions.

    **Model Variants:**

    1. **Gridded (default)**: Best for point-only observations
        - Uses approximately translation-equivariant ATETNP encoding
        - Supports pseudo-token or kernel-interpolation grid encoders
        - Grid resolution controlled by ``points_per_dim`` (e.g., 32×32)

    2. **OOTG (Off-The-Grid)**: Best for mixed point + gridded context
        - Handles point observations and gridded context separately
        - Preserves spatial structure of gridded data
        - More efficient when gridded context is available

    **Initialization Methods (via multiple dispatch):**

    The GriddedTNP class supports multiple initialization patterns:

    1. ``GriddedTNP(data_processor, task_loader, **kwargs)``
        Auto-infers dimensions from data.

    2. ``GriddedTNP(*args, **kwargs)``
        Direct instantiation with explicit hyperparameters.

    3. ``GriddedTNP(model_ID)``
        Load from saved weights and config.

    4. ``GriddedTNP(data_processor, task_loader, model_ID)``
        Load with data objects for prediction.

    Args:
        data_processor (:class:`~.data.processor.DataProcessor`, optional):
            DataProcessor for normalizing/unnormalizing data. Required for ``.predict()`` method.
        task_loader (:class:`~.data.loader.TaskLoader`, optional):
            TaskLoader for inferring model dimensions (``dim_yc``, ``dim_yt``, ``dim_aux_t``).
        model_ID (str, optional):
            Path to folder containing saved model weights (``model.pt``) and config (``model_config.json``).
        dim_x (int, optional):
            Spatial dimension (e.g., 2 for lat/lon). Defaults to 2.
        dim_yc (int | tuple[int], optional):
            Context variable dimensions. Inferred from ``task_loader`` if not provided.
        dim_yt (int, optional):
            Target variable dimensions. Inferred from ``task_loader`` if not provided.
        d_model (int, optional):
            Transformer embedding dimension. Defaults to 128.
        num_heads (int, optional):
            Number of attention heads. Defaults to 8.
        num_layers (int, optional):
            Number of transformer layers. Defaults to 6.
        model_variant ({'gridded', 'ootg'}, optional):
            Model architecture variant. Defaults to 'gridded'.
        grid_encoder_type ({'pseudo-token', 'kernel-interp'}, optional):
            Grid encoder family. Defaults to 'pseudo-token'.
        num_fourier (int, optional):
            Number of Fourier features used by ATETNP basis function module.
            Defaults to 16.
        points_per_dim (tuple[int, ...], optional):
            Internal grid resolution. Defaults to (32, 32) for 2D.
        grid_range (tuple[tuple[float, float], ...], optional):
            Spatial bounds for internal grid and basis domain.
            Defaults to ((0, 1), (0, 1)) for 2D.
        heteroscedastic (bool, optional):
            Use heteroscedastic (input-dependent) noise. Defaults to True.
        verbose (bool, optional):
            Print inferred parameters during initialization. Defaults to True.

    Example:
        >>> from deepsensor.model import GriddedTNP
        >>> from deepsensor.data import DataProcessor, TaskLoader
        >>> import xarray as xr
        >>>
        >>> # Load and process data
        >>> ds = xr.tutorial.open_dataset("air_temperature")
        >>> dp = DataProcessor(x1_name="lat", x2_name="lon")
        >>> ds_normalized = dp(ds)
        >>>
        >>> # Create task loader and model
        >>> tl = TaskLoader(context=ds_normalized, target=ds_normalized)
        >>> model = GriddedTNP(dp, tl, d_model=128, num_layers=6)
        >>>
        >>> # Train
        >>> from deepsensor.train import Trainer
        >>> trainer = Trainer(model, lr=5e-5)
        >>> task = tl("2013-01-01", context_sampling=50)
        >>> loss = trainer(task)
        >>>
        >>> # Predict
        >>> pred = model.predict(task, X_t=ds)

    Example (OOTG variant with gridded context):
        >>> # For mixed point + gridded context (e.g., stations + satellite)
        >>> tl = TaskLoader(
        ...     context=[stations_df, satellite_xr],  # point + gridded
        ...     target=stations_df
        ... )
        >>> model = GriddedTNP(dp, tl, model_variant='ootg', num_layers=8)

    See Also:
        :func:`construct_gridded_tnp`: Lower-level constructor for building the model.
        :class:`ConvNP`: Convolutional Neural Process alternative.
    """

    @dispatch
    def __init__(self, *args, **kwargs):
        """Generate a new model with default or specified parameters."""
        super().__init__()
        self.model, self.config = construct_gridded_tnp(*args, **kwargs)

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        *args,
        verbose: bool = True,
        **kwargs,
    ):
        """Instantiate model from TaskLoader, using data to infer model parameters.

        Args:
            data_processor (:class:`~.data.processor.DataProcessor`):
                DataProcessor object for unnormalising predictions.
            task_loader (:class:`~.data.loader.TaskLoader`):
                TaskLoader object for inferring model hyperparameters.
            verbose (bool, optional):
                Whether to print inferred model parameters. Defaults to True.
        """
        super().__init__(data_processor, task_loader)

        # Infer dimensions from TaskLoader
        if "dim_yc" not in kwargs:
            dim_yc = task_loader.context_dims
            if verbose:
                print(f"dim_yc inferred from TaskLoader: {dim_yc}")
            kwargs["dim_yc"] = dim_yc

        if "dim_yt" not in kwargs:
            dim_yt = sum(task_loader.target_dims)
            if verbose:
                print(f"dim_yt inferred from TaskLoader: {dim_yt}")
            kwargs["dim_yt"] = dim_yt

        if "dim_aux_t" not in kwargs:
            dim_aux_t = task_loader.aux_at_target_dims
            if verbose:
                print(f"dim_aux_t inferred from TaskLoader: {dim_aux_t}")
            kwargs["dim_aux_t"] = dim_aux_t

        # Infer grid parameters from data
        if "points_per_dim" not in kwargs:
            # Compute data density and use it to set grid resolution
            from deepsensor.model.defaults import compute_greatest_data_density
            
            internal_density = compute_greatest_data_density(task_loader)
            # Set grid resolution based on data density
            # Use sqrt(density) as a heuristic for grid points per dimension
            # Add 10% buffer to ensure sufficient resolution
            points_per_side = int(np.sqrt(internal_density) * 1.1)
            
            dim_x = kwargs.get("dim_x", 2)
            points_per_dim = tuple(points_per_side for _ in range(dim_x))
            
            if verbose:
                print(f"points_per_dim inferred from data density ({internal_density} ppu): {points_per_dim}")
            kwargs["points_per_dim"] = points_per_dim

        if "grid_range" not in kwargs:
            # Compute exact grid range from the data with a small adaptive buffer
            dim_x = kwargs.get("dim_x", 2)
            grid_range = _compute_grid_range_from_task_loader(task_loader, dim_x)
            
            if verbose:
                print(f"grid_range inferred from data: {grid_range}")
            kwargs["grid_range"] = grid_range

        if "init_lengthscale" not in kwargs and "points_per_dim" in kwargs:
            # Set initial lengthscale based on grid resolution
            # Use 2x the grid spacing as a reasonable default
            points_per_dim = kwargs["points_per_dim"]
            avg_grid_spacing = 1.0 / np.mean(points_per_dim)
            init_lengthscale = 2.0 * avg_grid_spacing
            
            if verbose:
                print(f"init_lengthscale inferred from grid resolution: {init_lengthscale:.4f}")
            kwargs["init_lengthscale"] = init_lengthscale

        self.model, self.config = construct_gridded_tnp(*args, **kwargs)

    @dispatch
    def __init__(self, model_ID: str):
        """Instantiate a model from a folder containing model weights and config."""
        super().__init__()
        self.load(model_ID)

    @dispatch
    def __init__(
        self,
        data_processor: DataProcessor,
        task_loader: TaskLoader,
        model_ID: str,
    ):
        """Instantiate a model from a folder with data processor and task loader."""
        super().__init__(data_processor, task_loader)
        self.load(model_ID)

    def save(self, model_ID: str):
        """Save the model weights and config to a folder.

        Args:
            model_ID (str): Folder to save the model to.
        """
        os.makedirs(model_ID, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_ID, "model.pt"))

        config_fpath = os.path.join(model_ID, "model_config.json")
        with open(config_fpath, "w") as f:
            json.dump(self.config, f, indent=4, sort_keys=False)

    def load(self, model_ID: str):
        """Load a model from a folder containing model weights and config.

        Args:
            model_ID (str): Folder to load the model from.
        """
        config_fpath = os.path.join(model_ID, "model_config.json")
        with open(config_fpath, "r") as f:
            self.config = json.load(f)

        self.model, _ = construct_gridded_tnp(**self.config)
        self.model.load_state_dict(torch.load(os.path.join(model_ID, "model.pt")))

    @classmethod
    def modify_task(cls, task: Task):
        """Prepare task for GriddedTNP model (add batch dim, convert to tensor, etc.).

        Args:
            task (:class:`~.data.task.Task`): The task to modify.

        Returns:
            :class:`~.data.task.Task`: The modified task.
        """
        if "batch_dim" not in task["ops"]:
            task = task.add_batch_dim()
        if "float32" not in task["ops"]:
            task = task.cast_to_float32()
        if "numpy_mask" not in task["ops"]:
            task = task.mask_nans_numpy()
        if "tensor" not in task["ops"]:
            task = task.convert_to_tensor()

        return task

    def __call__(self, task: Task, n_samples: Optional[int] = None, requires_grad: bool = False):
        """Compute GriddedTNP distribution.

        Args:
            task (:class:`~.data.task.Task`): Task containing context and targets.
            n_samples (int, optional): Number of samples to draw.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to False.

        Returns:
            Distribution: The GriddedTNP distribution.
        """
        task = GriddedTNP.modify_task(task)
        dist = run_gridded_tnp_model(self.model, task, n_samples, requires_grad)
        return dist

    def _cast_numpy_and_squeeze(self, x: B.Numeric, squeeze_axes: List[int] = (0, 1)):
        """Convert tensor to numpy and squeeze dimensions."""
        x_np = B.to_numpy(x)
        axes = sorted(set(int(ax) for ax in squeeze_axes), reverse=True)
        for axis in axes:
            if -x_np.ndim <= axis < x_np.ndim and x_np.shape[axis] == 1:
                x_np = np.squeeze(x_np, axis=axis)
        return x_np

    def _normalise_prediction_shape(self, arr: np.ndarray) -> np.ndarray:
        """Ensure prediction arrays follow (N_dims, N_targets) convention."""
        if arr.ndim != 2:
            return arr

        n_target_dims = sum(len(var_ids) for var_ids in self.task_loader.target_var_IDs)
        if arr.shape[0] == n_target_dims:
            return arr
        if arr.shape[1] == n_target_dims:
            return arr.T
        return arr

    # Prediction methods
    @dispatch
    def mean(self, dist):
        """Extract mean from GriddedTNP distribution."""
        mean = dist.mean
        mean = self._cast_numpy_and_squeeze(mean)
        return self._normalise_prediction_shape(mean)

    @dispatch
    def mean(self, task: Task):
        """Mean values at target locations.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            :class:`numpy:numpy.ndarray`: Mean values.
        """
        dist = self(task)
        return self.mean(dist)

    @dispatch
    def variance(self, dist):
        """Extract variance from GriddedTNP distribution."""
        variance = dist.variance
        variance = self._cast_numpy_and_squeeze(variance)
        return self._normalise_prediction_shape(variance)

    @dispatch
    def variance(self, task: Task):
        """Variance values at target locations.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            :class:`numpy:numpy.ndarray`: Variance values.
        """
        dist = self(task)
        return self.variance(dist)

    @dispatch
    def std(self, dist):
        """Extract standard deviation from GriddedTNP distribution."""
        return np.sqrt(self.variance(dist))

    @dispatch
    def std(self, task: Task):
        """Standard deviation values at target locations.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            :class:`numpy:numpy.ndarray`: Standard deviation values.
        """
        dist = self(task)
        return self.std(dist)

    @dispatch
    def sample(self, dist, n_samples: int = 1):
        """Draw samples from GriddedTNP distribution."""
        samples = dist.sample((n_samples,))
        # Be careful to keep sample dimension in position 0
        return self._cast_numpy_and_squeeze(samples, squeeze_axes=(1, 2))

    @dispatch
    def sample(self, task: Task, n_samples: int = 1):
        """Draw samples from the model.

        Args:
            task (:class:`~.data.task.Task`): The task.
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            :class:`numpy:numpy.ndarray`: Samples with shape (n_samples, n_features, *n_targets).
        """
        dist = self(task)
        return self.sample(dist, n_samples)

    @dispatch
    def logpdf(self, dist, task: Task):
        """Compute log probability density.

        Args:
            dist: The distribution from model forward pass.
            task (:class:`~.data.task.Task`): The task containing target observations.

        Returns:
            torch.Tensor: The log probability density (scalar, averaged over batch).
        """
        task = GriddedTNP.modify_task(task)
        # task["Y_t"] is a list with one element: the batch tensor
        # We extract it to get shape [batch, n_targets, dim_yt]
        Y_t = task["Y_t"][0]
        # Compute log prob and average over batch
        return dist.log_prob(Y_t).mean()

    @dispatch
    def logpdf(self, task: Task):
        """Compute log probability density.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            float: The log probability density.
        """
        dist = self(task, requires_grad=True)
        return self.logpdf(dist, task)

    def loss_fn(self, task: Task, **kwargs):
        """Compute the loss of a task.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            float: The loss (negative log probability).
        """
        return -self.logpdf(task)

    # Optional: implement entropy, covariance methods if needed
    def mean_marginal_entropy(self, task: Task):
        """Mean marginal entropy over target points.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            float: The mean marginal entropy.
        """
        dist = self(task)
        # TODO: Implement entropy calculation
        raise NotImplementedError("Entropy not yet implemented for GriddedTNP")

    def joint_entropy(self, task: Task):
        """Joint entropy over target points.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            float: The joint entropy.
        """
        dist = self(task)
        # TODO: Implement entropy calculation
        raise NotImplementedError("Entropy not yet implemented for GriddedTNP")

    def covariance(self, task: Task):
        """Covariance matrix over target points.

        Args:
            task (:class:`~.data.task.Task`): The task.

        Returns:
            :class:`numpy:numpy.ndarray`: Covariance matrix.
        """
        # TODO: Implement covariance calculation
        raise NotImplementedError("Covariance not yet implemented for GriddedTNP")


def construct_gridded_tnp(
    dim_x: int = 2,
    dim_yc: int = 1,
    dim_yt: int = 1,
    dim_aux_t: Optional[int] = None,
    d_model: int = 128,
    num_heads: int = 8,
    num_layers: int = 6,
    head_dim: Optional[int] = None,
    p_dropout: float = 0.0,
    grid_range: Optional[Tuple[Tuple[float, float], ...]] = None,
    points_per_dim: Optional[Tuple[int, ...]] = None,
    init_lengthscale: float = 0.1,
    train_noise: bool = True,
    min_noise: float = 1e-4,
    heteroscedastic: bool = True,
    xy_encoder_num_layers: int = 2,
    xy_encoder_width: int = 128,
    z_decoder_num_layers: int = 2,
    z_decoder_width: int = 128,
    model_variant: Literal["gridded", "ootg"] = "gridded",
    grid_encoder_type: Literal["pseudo-token", "kernel-interp"] = "pseudo-token",
    num_fourier: int = 16,
    p_basis_dropout: float = 0.5,
    top_k_ctot: Optional[int] = None,
    norm_first: bool = True,
    window_sizes: Optional[Tuple[int, ...]] = None,
    shift_sizes: Optional[Tuple[int, ...]] = None,
    roll_dims: Optional[Tuple[int, ...]] = None,
    margin: Optional[Tuple[float, ...]] = None,
    likelihood: Literal["normal", "het", "bernoulli-gamma"] = "het",
    **kwargs,
):
    """Construct a GriddedTNP model with specified architecture and hyperparameters.

    This function builds an approximately translation-equivariant GriddedTNP model by
    assembling ATETNP encoder, Swin/TE transformer, decoder, and likelihood components.
    It supports both ``gridded`` and ``ootg`` context layouts plus a selectable grid encoder
    family (pseudo-token or kernel-interpolation).

    **Architecture Overview:**

    1. **Encoder**: Projects observations to gridded embeddings (ATETNP)
        - Optional Fourier basis augmentation (controlled by ``num_fourier``)
        - ``'pseudo-token'``: TE pseudo-token grid encoder
        - ``'kernel-interp'``: SetConv/OOTGSetConv grid encoder

    2. **Transformer**: Swin + TE attention via ``GriddedTransformerEncoder``
        - TE self-attention over windows
        - TE cross-attention grid decoder (optionally top-k)
        - ``num_layers`` stacked transformer blocks

    3. **Decoder**: TNPDecoder projects transformer outputs to predictions
        - MLP mapping from embeddings to likelihood parameters

    4. **Likelihood**: One of the following:
        - Bernoulli-Gamma 
        - Gaussian likelihood: Heteroscedastic (input-dependent noise) or homoscedastic (default = Gaussian homoscedastic)

    **Model Variants:**

    - ``'gridded'`` (default):
        - Point observations on an internal grid
        - Gridded context is flattened to points in task conversion
        - Best for: Point-only observations and interpolation

    - ``'ootg'`` (Off-The-Grid):
        - Keeps gridded context explicit (no flattening)
        - Best for: Mixed point + gridded observations (e.g., stations + satellite)

    Args:
        dim_x (int, optional):
            Spatial input dimensionality (e.g., 2 for lat/lon, 3 for lat/lon/altitude).
            Defaults to 2.
        dim_yc (int | tuple[int], optional):
            Context observation dimensionality. Can be a tuple for multiple context variables.
            Defaults to 1.
        dim_yt (int, optional):
            Target observation dimensionality. Defaults to 1.
        dim_aux_t (int, optional):
            Auxiliary target variable dimensionality (e.g., time features).
            Currently unused by default builder. Defaults to None.
        d_model (int, optional):
            Transformer embedding dimension. Higher values increase model capacity but also
            computational cost. Defaults to 128.
        num_heads (int, optional):
            Number of attention heads in multi-head self-attention. Must divide ``d_model``
            evenly if ``head_dim`` is not specified. Defaults to 8.
        num_layers (int, optional):
            Number of transformer encoder layers. More layers increase capacity for
            modeling complex spatial dependencies. Defaults to 6.
        head_dim (int, optional):
            Dimension of each attention head. If None, computed as ``d_model // num_heads``.
            Defaults to None.
        p_dropout (float, optional):
            Dropout probability in transformer layers. Defaults to 0.0.
        grid_range (tuple[tuple[float, float], ...], optional):
            Spatial bounds for the internal grid (gridded variant only).
            Must have length ``dim_x``. Example: ``((0.0, 1.0), (0.0, 1.0))`` for 2D.
            Defaults to unit hypercube ``[(0, 1)] * dim_x``.
        points_per_dim (tuple[int, ...], optional):
            Grid resolution along each spatial dimension.
            Example: ``(32, 32)`` creates a 32×32 grid. Defaults to ``(32,) * dim_x``.
        init_lengthscale (float, optional):
            Initial lengthscale for SetConv RBF kernels. Controls the spatial extent
            of context influence. Defaults to 0.1.
        train_noise (bool, optional):
            Whether observation noise is trainable (homoscedastic likelihood only).
            Defaults to True.
        min_noise (float, optional):
            Minimum observation noise (homoscedastic) or noise floor (heteroscedastic).
            Prevents numerical instability. Defaults to 1e-4.
        heteroscedastic (bool, optional):
            Use heteroscedastic (input-dependent) noise model. If False, uses fixed
            (trainable) noise. Defaults to True.
        xy_encoder_num_layers (int, optional):
            Number of MLP layers in xy_encoder (maps [x, y, mask] to embeddings).
            Defaults to 2.
        xy_encoder_width (int, optional):
            Hidden width of xy_encoder MLP. Defaults to 128.
        z_decoder_num_layers (int, optional):
            Number of MLP layers in decoder (maps embeddings to likelihood parameters).
            Defaults to 2.
        z_decoder_width (int, optional):
            Hidden width of decoder MLP. Defaults to 128.
        model_variant ({'gridded', 'ootg'}, optional):
            Model architecture variant. See above for detailed comparison.
            Defaults to 'gridded'.
        grid_encoder_type ({'pseudo-token', 'kernel-interp'}, optional):
            Choice of grid encoder family. Defaults to 'pseudo-token'.
        num_fourier (int, optional):
            Number of Fourier features for basis augmentation. Defaults to 16.
        p_basis_dropout (float, optional):
            Dropout probability applied to basis features. Defaults to 0.5.
        top_k_ctot (int, optional):
            Restrict TE cross-attention decoder to nearest grid points. Defaults to None.
        norm_first (bool, optional):
            LayerNorm-first transformer blocks. Defaults to True.
        window_sizes (tuple[int, ...], optional):
            Swin window sizes per spatial dimension. If None, inferred from ``points_per_dim``.
        shift_sizes (tuple[int, ...], optional):
            Swin shift sizes per dimension. If None, inferred from ``window_sizes``.
        roll_dims (tuple[int, ...], optional):
            Optional periodic/rolled dimensions for local-neighbour operations.
        margin (tuple[float, ...], optional):
            Optional margin for pseudo-token TE grid extent in gridded mode.
        likelihood ({'normal', 'het', 'bernoulli-gamma'}, optional):
            Likelihood function for the model:
            
            - ``'normal'``: Gaussian with fixed (trainable) noise
            - ``'het'``: Heteroscedastic Gaussian (input-dependent noise) - default
            - ``'bernoulli-gamma'``: Bernoulli-Gamma for spike-and-slab modeling
              (point masses at zero + continuous positive values)
            
            Note: Bernoulli-Gamma requires custom implementation in gridded-tnp.
            Defaults to 'het'.

    Returns:
        tuple[torch.nn.Module, dict]: A tuple containing:
            - **model**: GriddedATETNP or OOTGGriddedATETNP PyTorch module
            - **config**: Dictionary of all hyperparameters for reproducibility/saving

    Raises:
        NotImplementedError:
            If backend is not PyTorch (only PyTorch is currently supported).
        ValueError:
            If ``model_variant`` is not 'gridded' or 'ootg', or if dimension mismatches occur.

    Example:
        >>> from deepsensor.model.gridded_tnp import construct_gridded_tnp
        >>>
        >>> # Build a gridded model with custom hyperparameters
        >>> model, config = construct_gridded_tnp(
        ...     dim_x=2,
        ...     dim_yc=3,  # 3 context variables (e.g., temp, pressure, humidity)
        ...     dim_yt=1,  # 1 target variable
        ...     d_model=256,
        ...     num_heads=8,
        ...     num_layers=12,
        ...     points_per_dim=(64, 64),  # Higher resolution grid
        ...     heteroscedastic=True
        ... )
        >>>
        >>> # Build an OOTG model for mixed point + gridded context
        >>> model_ootg, config_ootg = construct_gridded_tnp(
        ...     dim_x=2,
        ...     dim_yc=1,
        ...     dim_yt=1,
        ...     model_variant='ootg',
        ...     num_layers=8
        ... )

    See Also:
        :class:`GriddedTNP`: High-level wrapper class with DeepSensor integration.
        :func:`convert_task_to_gridded_tnp_args`: Convert DeepSensor Tasks to model inputs.
    """
    if backend.str != "torch":
        raise NotImplementedError(
            f"GriddedTNP only supports PyTorch backend, got {backend.str}"
        )

    if kwargs:
        raise ValueError(
            f"Unexpected keyword arguments for GriddedTNP construction: {list(kwargs.keys())}"
        )

    if model_variant not in ["gridded", "ootg"]:
        raise ValueError(
            f"model_variant must be one of ['gridded', 'ootg'], got {model_variant!r}"
        )
    if grid_encoder_type not in ["pseudo-token", "kernel-interp"]:
        raise ValueError(
            "grid_encoder_type must be one of ['pseudo-token', 'kernel-interp'], "
            f"got {grid_encoder_type!r}"
        )

    if isinstance(dim_yc, (tuple, list)):
        dim_yc_values = tuple(int(d) for d in dim_yc)
        dim_yc_total = int(sum(dim_yc_values))
        if model_variant == "ootg" and len(dim_yc_values) >= 2:
            dim_yc_grid = int(dim_yc_values[0])
            dim_yc_point = int(sum(dim_yc_values[1:]))
        else:
            dim_yc_grid = dim_yc_total
            dim_yc_point = dim_yc_total
    else:
        dim_yc_total = int(dim_yc)
        dim_yc_grid = dim_yc_total
        dim_yc_point = dim_yc_total

    if dim_aux_t not in (None, 0):
        warnings.warn(
            "dim_aux_t is currently not used by the default GriddedTNP builder.",
            stacklevel=2,
        )

    if num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")

    if head_dim is None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) when head_dim is not set."
            )
        head_dim = d_model // num_heads

    if grid_range is None:
        grid_range = tuple((0.0, 1.0) for _ in range(dim_x))
    else:
        grid_range = tuple(tuple(bounds) for bounds in grid_range)
    if points_per_dim is None:
        points_per_dim = tuple(32 for _ in range(dim_x))
    else:
        points_per_dim = tuple(int(v) for v in points_per_dim)

    if len(grid_range) != dim_x:
        raise ValueError(
            f"grid_range must have length dim_x={dim_x}, got {len(grid_range)}"
        )
    if len(points_per_dim) != dim_x:
        raise ValueError(
            f"points_per_dim must have length dim_x={dim_x}, got {len(points_per_dim)}"
        )

    try:
        from tnp.likelihoods.gaussian import (
            HeteroscedasticNormalLikelihood,
            NormalLikelihood,
        )
        from tnp.models.gridded_atetnp import GriddedATETNP
        from tnp.models.gridded_atetnp import GriddedATETNPEncoder
        from tnp.models.gridded_atetnp import OOTGGriddedATETNP
        from tnp.models.gridded_atetnp import OOTGGriddedATETNPEncoder
        from tnp.models.tnp import TNPDecoder
        from tnp.networks.grid_encoders import (
            OOTGPseudoTokenTEGridEncoder,
            OOTGSetConv,
            PseudoTokenTEGridEncoder,
            SetConv,
        )
        from tnp.networks.mlp import MLP
        from tnp.networks.modules import ModuleOnFourierExpandedInput
        from tnp.networks.swin_attention import SWINAttentionLayer
        from tnp.networks.te_grid_decoders import TEMHCAGridDecoder
        from tnp.networks.teattention_layers import (
            GriddedMultiHeadSelfTEAttentionLayer,
            MultiHeadCrossTEAttentionLayer,
        )
        from tnp.networks.transformer import GriddedTransformerEncoder
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import gridded-tnp components. Ensure `gridded-tnp` and its dependencies "
            "(notably `check_shapes`) are installed in the active environment."
        ) from exc
    
    # Try to import Bernoulli-Gamma likelihood (may not exist in all installations)
    try:
        from tnp.likelihoods.bernoulli_gamma import BernoulliGammaLikelihood
    except ImportError:
        BernoulliGammaLikelihood = None

    y_encoder = MLP(
        in_dim=dim_yc_point + 1,
        out_dim=d_model,
        num_layers=xy_encoder_num_layers,
        width=xy_encoder_width,
    )

    if window_sizes is None:
        window_sizes = tuple(max(1, min(4, int(points))) for points in points_per_dim)
    else:
        window_sizes = tuple(int(v) for v in window_sizes)
    if len(window_sizes) != dim_x:
        raise ValueError(
            f"window_sizes must have length dim_x={dim_x}, got {len(window_sizes)}"
        )

    if shift_sizes is None:
        shift_sizes = tuple(0 if w <= 1 else w // 2 for w in window_sizes)
    else:
        shift_sizes = tuple(int(v) for v in shift_sizes)
    if len(shift_sizes) != dim_x:
        raise ValueError(
            f"shift_sizes must have length dim_x={dim_x}, got {len(shift_sizes)}"
        )

    if roll_dims is not None:
        roll_dims = tuple(int(d) for d in roll_dims)

    if margin is not None:
        margin = tuple(float(m) for m in margin)
        if len(margin) != dim_x:
            raise ValueError(f"margin must have length dim_x={dim_x}, got {len(margin)}")

    mhca_kernel_decoder = MLP(
        in_dim=dim_x,
        out_dim=num_heads,
        num_layers=2,
        width=num_heads,
    )
    te_mhca_layer = MultiHeadCrossTEAttentionLayer(
        embed_dim=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=d_model,
        norm_first=norm_first,
        p_dropout=p_dropout,
        kernel=mhca_kernel_decoder,
    )
    grid_decoder = TEMHCAGridDecoder(
        mhca_layer=te_mhca_layer,
        top_k_ctot=top_k_ctot,
        roll_dims=roll_dims,
    )
    mhsa_layer = GriddedMultiHeadSelfTEAttentionLayer(
        embed_dim=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        grid_shape=window_sizes,
        feedforward_dim=d_model,
        norm_first=norm_first,
        p_dropout=p_dropout,
    )
    swin_layer = SWINAttentionLayer(
        mhsa_layer=mhsa_layer,
        window_sizes=window_sizes,
        shift_sizes=shift_sizes,
        roll_dims=roll_dims,
    )
    transformer_encoder = GriddedTransformerEncoder(
        num_layers=num_layers,
        grid_decoder=grid_decoder,
        mhsa_layer=swin_layer,
    )

    basis_fn_mlp = MLP(
        in_dim=dim_x * num_fourier,
        out_dim=d_model,
        num_layers=2,
        width=d_model,
    )
    basis_fn = ModuleOnFourierExpandedInput(
        module=basis_fn_mlp,
        x_range=grid_range,
        num_fourier=num_fourier,
    )

    if grid_encoder_type == "pseudo-token":
        mhca_kernel_encoder = MLP(
            in_dim=dim_x,
            out_dim=num_heads,
            num_layers=2,
            width=num_heads,
        )
        grid_encoder_te_mhca_layer = MultiHeadCrossTEAttentionLayer(
            embed_dim=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            feedforward_dim=d_model,
            norm_first=norm_first,
            p_dropout=p_dropout,
            kernel=mhca_kernel_encoder,
        )

        if model_variant == "gridded":
            points_per_unit_dim = []
            for axis, bounds in enumerate(grid_range):
                axis_range = float(bounds[1] - bounds[0])
                if axis_range <= 0:
                    raise ValueError(
                        f"grid_range[{axis}] has non-positive span {axis_range}; cannot derive points_per_unit_dim."
                    )
                ppu = max(1, int(round(points_per_dim[axis] / axis_range)))
                points_per_unit_dim.append(ppu)

            grid_encoder = PseudoTokenTEGridEncoder(
                embed_dim=d_model,
                mhca_layer=grid_encoder_te_mhca_layer,
                points_per_unit_dim=tuple(points_per_unit_dim),
                margin=margin,
            )
        else:
            grid_encoder = OOTGPseudoTokenTEGridEncoder(
                embed_dim=d_model,
                mhca_layer=grid_encoder_te_mhca_layer,
                grid_shape=points_per_dim,
            )
    else:
        if model_variant == "gridded":
            grid_encoder = SetConv(
                dims=dim_x,
                grid_range=grid_range,
                points_per_dim=points_per_dim,
                init_lengthscale=init_lengthscale,
            )
        else:
            grid_encoder = OOTGSetConv(
                dims=dim_x,
                grid_shape=points_per_dim,
                init_lengthscale=init_lengthscale,
            )

    if model_variant == "gridded":
        encoder = GriddedATETNPEncoder(
            tetransformer_encoder=transformer_encoder,
            grid_encoder=grid_encoder,
            y_encoder=y_encoder,
            basis_fn=basis_fn,
            p_basis_dropout=p_basis_dropout,
        )
    else:
        y_grid_encoder = MLP(
            in_dim=dim_yc_grid,
            out_dim=d_model,
            num_layers=xy_encoder_num_layers,
            width=xy_encoder_width,
        )
        encoder = OOTGGriddedATETNPEncoder(
            tetransformer_encoder=transformer_encoder,
            grid_encoder=grid_encoder,
            y_encoder=y_encoder,
            y_grid_encoder=y_grid_encoder,
            basis_fn=basis_fn,
            p_basis_dropout=p_basis_dropout,
        )

    # Determine likelihood output dimension and construct likelihood object
    if likelihood == "bernoulli-gamma":
        if BernoulliGammaLikelihood is None:
            raise ImportError(
                "Bernoulli-Gamma likelihood requested but not found in gridded-tnp. "
                "Ensure you've added tnp/likelihoods/bernoulli_gamma.py to your gridded-tnp installation."
            )
        likelihood_out_dim = 3 * dim_yt  # Bernoulli logits + Gamma shape + Gamma rate
        likelihood_obj = BernoulliGammaLikelihood()
    elif likelihood == "het":
        likelihood_out_dim = 2 * dim_yt  # Mean + variance
        likelihood_obj = HeteroscedasticNormalLikelihood(min_noise=min_noise)
    elif likelihood == "normal":
        likelihood_out_dim = dim_yt  # Mean only (fixed noise)
        likelihood_obj = NormalLikelihood(noise=min_noise, train_noise=train_noise)
    elif heteroscedastic:  # Backward compatibility: heteroscedastic flag
        likelihood_out_dim = 2 * dim_yt
        likelihood_obj = HeteroscedasticNormalLikelihood(min_noise=min_noise)
    else:
        likelihood_out_dim = dim_yt
        likelihood_obj = NormalLikelihood(noise=min_noise, train_noise=train_noise)

    z_decoder = MLP(
        in_dim=d_model,
        out_dim=likelihood_out_dim,
        num_layers=z_decoder_num_layers,
        width=z_decoder_width,
    )
    decoder = TNPDecoder(z_decoder=z_decoder)

    if model_variant == "gridded":
        model = GriddedATETNP(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood_obj,
        )
    else:
        model = OOTGGriddedATETNP(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood_obj,
        )

    model = model.float()
    model._deepsensor_model_variant = model_variant

    config = {
        "dim_x": dim_x,
        "dim_yc": dim_yc,
        "dim_yt": dim_yt,
        "dim_aux_t": dim_aux_t,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "p_dropout": p_dropout,
        "grid_range": [list(bounds) for bounds in grid_range],
        "points_per_dim": list(points_per_dim),
        "init_lengthscale": init_lengthscale,
        "train_noise": train_noise,
        "min_noise": min_noise,
        "heteroscedastic": heteroscedastic,
        "xy_encoder_num_layers": xy_encoder_num_layers,
        "xy_encoder_width": xy_encoder_width,
        "z_decoder_num_layers": z_decoder_num_layers,
        "z_decoder_width": z_decoder_width,
        "model_variant": model_variant,
        "grid_encoder_type": grid_encoder_type,
        "num_fourier": num_fourier,
        "p_basis_dropout": p_basis_dropout,
        "top_k_ctot": top_k_ctot,
        "norm_first": norm_first,
        "window_sizes": list(window_sizes),
        "shift_sizes": list(shift_sizes),
        "roll_dims": None if roll_dims is None else list(roll_dims),
        "margin": None if margin is None else list(margin),
        "likelihood": likelihood,
    }
    return model, config


def run_gridded_tnp_model(
    model,
    task: Task,
    n_samples: Optional[int] = None,
    requires_grad: bool = False,
):
    """Run GriddedTNP model on a task.

    Args:
        model: The GriddedTNP model.
        task (:class:`~.data.task.Task`): Task containing context and targets.
        n_samples (int, optional): Number of samples to draw.
        requires_grad (bool, optional): Whether to compute gradients. Defaults to False.

    Returns:
        Distribution: The model's output distribution.
    """

    # Convert task to GriddedTNP format
    model_variant = getattr(model, "_deepsensor_model_variant", None)
    if model_variant is None:
        model_variant = "ootg" if "OOTG" in model.__class__.__name__ else "gridded"
    model_args = convert_task_to_gridded_tnp_args(task, model_variant=model_variant)

    # Convert all tensors to float32 for consistency with model
    def _to_float32(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float32:
            return x.to(dtype=torch.float32)
        return x

    model_args = tuple(_to_float32(arg) for arg in model_args)

    # Run forward pass with or without gradients
    if not requires_grad:
        with torch.no_grad():
            dist = model(*model_args)
    else:
        dist = model(*model_args)

    return dist


def _to_torch_float(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.float()

    if hasattr(value, "y") and hasattr(value, "mask"):
        value = value.y

    if isinstance(value, np.ma.MaskedArray):
        value = value.filled(np.nan)

    if hasattr(value, "data") and not isinstance(value, np.ndarray):
        value = value.data

    arr = np.asarray(value)
    if arr.dtype == np.object_:
        arr = np.ma.asarray(arr)
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan)
        arr = np.asarray(arr, dtype=np.float32)

    return torch.as_tensor(arr).float()


def convert_task_to_gridded_tnp_args(
    task: Task,
    model_variant: Literal["gridded", "ootg"] = "gridded",
):
    """Convert DeepSensor Task to GriddedTNP format.

    GriddedTNP expects:
        - xc: [batch, n_context, dim_x] - context locations
        - yc: [batch, n_context, dim_y] - context observations
        - xt: [batch, n_target, dim_x] - target locations

    Args:
        task (:class:`~.data.task.Task`): DeepSensor task.

    Returns:
        tuple:
            - For ``model_variant='gridded'``: ``(xc, yc, xt)``
            - For ``model_variant='ootg'``: ``(xc, yc, xc_grid, yc_grid, xt)``
    """
    if model_variant not in ["gridded", "ootg"]:
        raise ValueError(
            f"model_variant must be one of ['gridded', 'ootg'], got {model_variant!r}"
        )

    def point_x_to_tnp(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[None, ...]
        if x.ndim != 3:
            raise ValueError(f"Expected point X with ndim 2 or 3, got shape {tuple(x.shape)}")
        # DeepSensor convention is typically (batch, dim_x, n_points)
        if x.shape[1] <= x.shape[2]:
            return x.transpose(1, 2)
        return x

    def point_y_to_tnp(y: torch.Tensor) -> torch.Tensor:
        y = _to_torch_float(y)
        if y.ndim == 2:
            y = y[None, ...]
        if y.ndim != 3:
            raise ValueError(f"Expected point Y with ndim 2 or 3, got shape {tuple(y.shape)}")
        if y.shape[1] <= y.shape[2]:
            return y.transpose(1, 2)
        return y

    def grid_y_to_tnp(y_grid: torch.Tensor) -> torch.Tensor:
        y_grid = _to_torch_float(y_grid)
        if y_grid.ndim == 3:
            y_grid = y_grid[None, ...]
        if y_grid.ndim != 4:
            raise ValueError(
                f"Expected gridded Y with ndim 3 or 4, got shape {tuple(y_grid.shape)}"
            )
        return y_grid.permute(0, 2, 3, 1)

    def grid_x_tuple_to_tnp(
        x_grid_tuple: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if len(x_grid_tuple) != 2:
            raise NotImplementedError(
                "Only 2D gridded context coordinates are currently supported for OOTG variant."
            )
        x1 = _to_torch_float(x_grid_tuple[0])
        x2 = _to_torch_float(x_grid_tuple[1])

        def _coord_to_1d(coord: torch.Tensor, axis_hint: str) -> torch.Tensor:
            if coord.ndim == 0:
                return coord.unsqueeze(0)
            if coord.ndim == 1:
                return coord
            if coord.ndim >= 3:
                # Typical batched case from concat_tasks: (batch, 1, n) or (batch, n, 1)
                c0 = coord[0]
                if c0.ndim == 2 and (c0.shape[0] == 1 or c0.shape[1] == 1):
                    return c0.reshape(-1)

                # Fallback for higher-rank inputs: collapse batch and recurse.
                return _coord_to_1d(c0, axis_hint)
            if coord.ndim == 2:
                if coord.shape[0] == 1 or coord.shape[1] == 1:
                    return coord.reshape(-1)

                if axis_hint == "x1":
                    col0 = coord[:, :1]
                    if torch.allclose(coord, col0.expand_as(coord)):
                        return coord[:, 0]
                    return coord[:, 0]

                row0 = coord[:1, :]
                if torch.allclose(coord, row0.expand_as(coord)):
                    return coord[0, :]
                return coord[0, :]

            return coord.reshape(-1)

        x1 = _coord_to_1d(x1, axis_hint="x1")
        x2 = _coord_to_1d(x2, axis_hint="x2")
        gx1, gx2 = torch.meshgrid(x1, x2, indexing="ij")
        xc_grid = torch.stack((gx1, gx2), dim=-1)[None, ...]
        if batch_size > 1:
            xc_grid = xc_grid.repeat(batch_size, 1, 1, 1)
        return xc_grid

    def flatten_grid_to_points(xc_grid: torch.Tensor, yc_grid: torch.Tensor):
        batch_size = xc_grid.shape[0]
        xc_point = xc_grid.reshape(batch_size, -1, xc_grid.shape[-1])
        yc_point = yc_grid.reshape(batch_size, -1, yc_grid.shape[-1])
        return xc_point, yc_point

    def target_x_to_tnp(x_t, batch_size: int) -> torch.Tensor:
        if isinstance(x_t, tuple):
            xt_grid = grid_x_tuple_to_tnp(x_t, batch_size=batch_size)
            return xt_grid.reshape(batch_size, -1, xt_grid.shape[-1])

        x_t = _to_torch_float(x_t)
        if x_t.ndim == 2:
            x_t = x_t[None, ...]
        if x_t.ndim != 3:
            raise ValueError(f"Expected target X with ndim 2 or 3, got shape {tuple(x_t.shape)}")
        if x_t.shape[1] <= x_t.shape[2]:
            return x_t.transpose(1, 2)
        return x_t

    # Extract context data
    X_c = task["X_c"]  # List of context location arrays
    Y_c = task["Y_c"]  # List of context observation arrays

    point_xc = []
    point_yc = []
    gridded_contexts = []

    for x_ci, y_ci in zip(X_c, Y_c):
        if isinstance(x_ci, tuple):
            yc_grid = grid_y_to_tnp(y_ci)
            batch_size = yc_grid.shape[0]
            xc_grid = grid_x_tuple_to_tnp(x_ci, batch_size=batch_size)
            if model_variant == "gridded":
                xc_point, yc_point = flatten_grid_to_points(xc_grid, yc_grid)
                point_xc.append(xc_point)
                point_yc.append(yc_point)
            else:
                gridded_contexts.append((xc_grid, yc_grid))
        else:
            point_xc.append(point_x_to_tnp(x_ci))
            point_yc.append(point_y_to_tnp(y_ci))

    if len(point_xc) == 0:
        if model_variant == "gridded":
            raise ValueError(
                "No usable point contexts were found. For gridded-only contexts, use model_variant='ootg'."
            )
        if len(gridded_contexts) == 0:
            raise ValueError("Task has no context sets in X_c/Y_c.")

    if len(point_xc) > 1:
        xc = torch.cat(point_xc, dim=1)
        yc = torch.cat(point_yc, dim=1)
    elif len(point_xc) == 1:
        xc = point_xc[0]
        yc = point_yc[0]
    else:
        # OOTG with only gridded context
        batch_size = gridded_contexts[0][0].shape[0]
        device = gridded_contexts[0][0].device
        dtype = gridded_contexts[0][0].dtype
        dy_grid = gridded_contexts[0][1].shape[-1]
        xc = torch.empty((batch_size, 0, gridded_contexts[0][0].shape[-1]), device=device, dtype=dtype)
        yc = torch.empty((batch_size, 0, dy_grid), device=device, dtype=dtype)

    # Extract target locations
    X_t = task["X_t"]
    if X_t is None:
        raise ValueError("Task must have target locations (X_t)")

    if len(X_t) == 1:
        xt = target_x_to_tnp(X_t[0], batch_size=xc.shape[0])
    else:
        # Multiple target sets - need to handle this case
        # For now, just use the first one
        warnings.warn("Multiple target sets detected, using first one only")
        xt = target_x_to_tnp(X_t[0], batch_size=xc.shape[0])

    if model_variant == "gridded":
        return xc, yc, xt

    if len(gridded_contexts) == 0:
        raise ValueError(
            "model_variant='ootg' requires at least one gridded context set (tuple-valued X_c)."
        )
    if len(gridded_contexts) > 1:
        raise NotImplementedError(
            "Multiple gridded context sets are not yet supported for model_variant='ootg'."
        )

    xc_grid, yc_grid = gridded_contexts[0]
    return xc, yc, xc_grid, yc_grid, xt
