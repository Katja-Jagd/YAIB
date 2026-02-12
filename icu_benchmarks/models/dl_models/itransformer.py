import gin
from torch import nn as nn
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import CustomDLPredictionWrapper, SSLWrapper

# From iTransformer
import torch
import numpy as np
from x_transformers import Encoder

# Import prediction heads and utility functions from BAT to avoid duplication
from icu_benchmarks.models.dl_models.bat import (
    BinaryClassificationHead,
    RegressionHead,
    ForecastingHead,
    TimeseriesClassificationHead,
)


# Own encoder prediction class that plugs the inverted encoder to different prediction heads
class EncoderPredictionInverted(nn.Module):
    """
    Note: This class is NOT gin.configurable because it's instantiated
    programmatically by iTransformer/SSL_iTransformer wrappers with explicit parameters.
    """
    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}

        # Handling input dim for head depending on the output dimension of the encoding model
        if isinstance(self.encoder_class, (EncoderClassifierInverted, AutoregressiveEncoderClassifierInverted)):
            if self.encoder_class.use_static:
                self.input_dim = (
                    self.encoder_class.sensors_count * self.encoder_class.time_axis_dim
                    + self.encoder_class.static_out
                )
            else:
                self.input_dim = self.encoder_class.sensors_count * self.encoder_class.time_axis_dim
        else:
            raise ValueError("Unknown encoder class: cannot determine input dimension.")

        # Prediction head initialization
        self.head = self.prediction_head(
            input_dim=self.input_dim,
            **self.prediction_head_kwargs
        )

    def forward(self, x, static, time, sensor_mask):
        features = self.encoder_class(x, static, time, sensor_mask)

        # Sanity check shape
        # For autoregressive encoder: features shape is (N, T, E)
        # For regular encoder: features shape is (N, E)
        # Check if autoregressive by looking at output dimensionality
        if len(features.shape) == 3:  # Autoregressive: (N, T, E)
            if features.shape[2] != self.input_dim:
                raise ValueError(
                    f"Mismatch between computed input_dim ({self.input_dim}) and actual ({features.shape[2]})"
                )
        else:  # Regular: (N, E)
            if features.shape[1] != self.input_dim:
                raise ValueError(
                    f"Mismatch between computed input_dim ({self.input_dim}) and actual ({features.shape[1]})"
                )

        return self.head(features)


# Note: masked_mean_pooling and masked_max_pooling are NOT imported from BAT
# because they are not used in iTransformer (uses unrolling instead of pooling)


class EncoderClassifierInverted(nn.Module):
    """
    Inverted Transformer encoder that operates on the sensor dimension.

    Unlike BAT which attends over sensors and time separately,
    iTransformer embeds the time dimension and applies attention over sensors.
    This inverts the typical time-series transformer architecture.

    Note: This class is NOT gin.configurable because it's instantiated
    programmatically by iTransformer/SSL_iTransformer wrappers with explicit parameters.
    """

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        time_embed_size=9,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        return_intermediates=False,
        use_mask=False,
        use_static=True,
        obs_strategy="both",
        **kwargs
    ):
        super().__init__()

        self.return_intermediates = return_intermediates
        self.obs_strategy = obs_strategy
        self.pooling = pooling
        self.device = device
        self.use_mask = use_mask
        self.sensors_count = sensors_count
        self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count
        self.use_static = use_static

        # Determine time dimension based on observation strategy
        if self.obs_strategy in ("indicator_only", "obs_only"):
            self.time_axis_dim_in = self.max_timepoint_count
        else:
            self.time_axis_dim_in = 2 * self.max_timepoint_count

        # Compressed time dimension for attention (configurable hyperparameter)
        self.time_axis_dim = min(self.time_axis_dim_in, time_embed_size)
        self.static_out = self.static_count + 4

        # Attention layers over sensors (inverted from BAT)
        self.attn_layers_1 = Encoder(
            dim=self.time_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
            attn_flash=False,  # Compatible with A100
        )

        # Process entire time series at once (not per-timepoint)
        self.time_embedding = nn.Linear(self.time_axis_dim_in, self.time_axis_dim)

        self.use_static = use_static
        
        # Static feature handling
        if self.use_static:
            print("use static in inverted transformer")
            self.static_embedding = nn.Linear(self.static_count, self.static_out)
            self.nonlinear_merger = nn.Linear(
                self.sensors_count * self.time_axis_dim + self.static_out,
                self.sensors_count * self.time_axis_dim + self.static_out,
            )
        else:
            print("do not use static in inverted transformer")
            self.nonlinear_merger = nn.Linear(
                self.sensors_count * self.time_axis_dim,
                self.sensors_count * self.time_axis_dim,
            )

    def forward(self, x, static, time, sensor_mask, **kwargs):
        """
        Forward pass of inverted transformer.

        Args:
            x: (N, F, T) - batch, sensors, time
            static: (N, static_count) - static features
            time: (N, T) - time encodings
            sensor_mask: (N, F, T) - mask for missing values

        Returns:
            features: (N, E) - encoded representation
        """

        x_sensor = torch.clone(x)  # (N, F, T)

        # Add indication for missing sensor values
        x_time_mask = torch.clone(sensor_mask)  # (N, F, T)
        if self.obs_strategy == "indicator_only":
            x_sensor = x_time_mask.float()
        elif self.obs_strategy == "obs_only":
            x_sensor = x_sensor
        elif self.obs_strategy == "both":
            x_sensor = torch.cat([x_sensor, x_time_mask], axis=2)  # (N, F, 2T)
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

        # Free memory
        del x_time_mask

        # Pad time dimension to time_axis_dim_in if shorter (SSL obs windows are variable length;
        # for non-SSL this is a no-op because T already equals max_timepoint_count)
        if x_sensor.shape[2] < self.time_axis_dim_in:
            x_sensor = torch.nn.functional.pad(x_sensor, (0, self.time_axis_dim_in - x_sensor.shape[2]))

        # Make embedding of time dimension - process entire time series at once
        # This compresses the time dimension from time_axis_dim_in to time_axis_dim
        x_sensor = self.time_embedding(x_sensor)  # (N, F, time_axis_dim)

        # Run sensor attention (this is the "inverted" part - attending over sensors)
        if self.return_intermediates:
            x_sensor, sensor_intermediates = self.attn_layers_1(
                x_sensor, return_hiddens=True
            )
        else:
            x_sensor = self.attn_layers_1(x_sensor)  # (N, F, time_axis_dim)

        # Unroll the sensor and time dimensions
        x_sensor = x_sensor.reshape(x_sensor.shape[0], -1)  # (N, F * time_axis_dim)

        # Concatenate with static features if used
        if self.use_static:
            static_embedded = self.static_embedding(static)  # (N, static_out)
            x_merged = torch.cat((x_sensor, static_embedded), axis=1)
        else:
            x_merged = x_sensor

        # Apply nonlinear transformation so it can be fit to prediction head
        nonlinear_merged = self.nonlinear_merger(x_merged).relu()

        if self.return_intermediates:
            return sensor_intermediates.attn_intermediates[0].post_softmax_attn, None

        return nonlinear_merged


class AutoregressiveEncoderClassifierInverted(nn.Module):
    """
    Autoregressive version of the inverted transformer encoder.

    Makes per-timestep predictions by masking out future timepoints.
    At each timestep t, only information from timesteps <= t is used.

    This is achieved by:
    1. Masking future timepoints (setting values to 0 and marking as missing)
    2. Processing each timestep separately with the inverted transformer
    3. Returning per-timestep representations (N, T, E) instead of pooled (N, E)

    Note: This class is NOT gin.configurable because it's instantiated
    programmatically by Autoregressive_iTransformer wrapper with explicit parameters.
    """

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        time_embed_size=9,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        return_intermediates=False,
        use_mask=False,
        use_static=True,
        obs_strategy="both",
        **kwargs
    ):
        super().__init__()

        self.return_intermediates = return_intermediates
        self.obs_strategy = obs_strategy
        self.pooling = pooling
        self.device = device
        self.use_mask = use_mask
        self.sensors_count = sensors_count
        self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count
        self.use_static = use_static

        # Determine time dimension based on observation strategy
        if self.obs_strategy in ("indicator_only", "obs_only"):
            self.time_axis_dim_in = self.max_timepoint_count
        else:
            self.time_axis_dim_in = 2 * self.max_timepoint_count

        # Compressed time dimension for attention (configurable hyperparameter)
        self.time_axis_dim = min(self.time_axis_dim_in, time_embed_size)
        self.static_out = self.static_count + 4

        # Attention layers over sensors (inverted from BAT)
        self.attn_layers_1 = Encoder(
            dim=self.time_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
            attn_flash=False,  # Compatible with A100
        )

        # Process entire time series at once (not per-timepoint)
        self.time_embedding = nn.Linear(self.time_axis_dim_in, self.time_axis_dim)

        self.use_static = use_static

        # Static feature handling
        if self.use_static:
            print("use static in autoregressive inverted transformer")
            self.static_embedding = nn.Linear(self.static_count, self.static_out)
            self.nonlinear_merger = nn.Linear(
                self.sensors_count * self.time_axis_dim + self.static_out,
                self.sensors_count * self.time_axis_dim + self.static_out,
            )
        else:
            print("do not use static in autoregressive inverted transformer")
            self.nonlinear_merger = nn.Linear(
                self.sensors_count * self.time_axis_dim,
                self.sensors_count * self.time_axis_dim,
            )

    def forward(self, x, static, time, sensor_mask, **kwargs):
        """
        Autoregressive forward pass with future masking.

        Args:
            x: (N, F, T) - batch, sensors, time
            static: (N, static_count) - static features
            time: (N, T) - time encodings
            sensor_mask: (N, F, T) - mask for missing values

        Returns:
            features: (N, T, E) - per-timestep encoded representations
        """
        N, F, T = x.shape

        # We'll process each timestep t by masking all future timesteps > t
        # and applying the encoder to get representation for timestep t
        output_dim = (
            self.sensors_count * self.time_axis_dim + self.static_out
            if self.use_static
            else self.sensors_count * self.time_axis_dim
        )

        # Initialize output tensor
        outputs = torch.zeros(N, T, output_dim, device=x.device)

        # Process each timestep
        for t in range(T):
            # Create masked version of input where all timesteps > t are masked
            x_masked = x.clone()
            sensor_mask_masked = sensor_mask.clone()

            # Mask future timepoints (t+1 onwards)
            if t < T - 1:
                x_masked[:, :, t+1:] = 0  # Set to missing value indicator
                sensor_mask_masked[:, :, t+1:] = False  # Mark as missing

            # Process this masked input through the encoder
            x_sensor = torch.clone(x_masked)  # (N, F, T)

            # Add indication for missing sensor values
            x_time_mask = torch.clone(sensor_mask_masked)  # (N, F, T)
            if self.obs_strategy == "indicator_only":
                x_sensor = x_time_mask.float()
            elif self.obs_strategy == "obs_only":
                x_sensor = x_sensor
            elif self.obs_strategy == "both":
                x_sensor = torch.cat([x_sensor, x_time_mask], axis=2)  # (N, F, 2T)
            else:
                raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

            # Free memory
            del x_time_mask

            # Make embedding of time dimension
            x_sensor = self.time_embedding(x_sensor)  # (N, F, time_axis_dim)

            # Run sensor attention
            if self.return_intermediates:
                x_sensor, sensor_intermediates = self.attn_layers_1(
                    x_sensor, return_hiddens=True
                )
            else:
                x_sensor = self.attn_layers_1(x_sensor)  # (N, F, time_axis_dim)

            # Unroll the sensor and time dimensions
            x_sensor = x_sensor.reshape(x_sensor.shape[0], -1)  # (N, F * time_axis_dim)

            # Concatenate with static features if used
            if self.use_static:
                static_embedded = self.static_embedding(static)  # (N, static_out)
                x_merged = torch.cat((x_sensor, static_embedded), axis=1)
            else:
                x_merged = x_sensor

            # Apply nonlinear transformation
            nonlinear_merged = self.nonlinear_merger(x_merged).relu()

            # Store the output for this timestep
            outputs[:, t, :] = nonlinear_merged

        if self.return_intermediates:
            return sensor_intermediates.attn_intermediates[0].post_softmax_attn, None

        return outputs  # (N, T, E)


@gin.configurable
class iTransformer(CustomDLPredictionWrapper):
    """
    Inverted Transformer wrapper for YAIB framework.

    Automatically selects between regular and autoregressive encoder based on
    the TIMESTEP_LEVEL_PREDICTIONS gin parameter:
    - If True: Uses AutoregressiveEncoderClassifierInverted for per-timestep predictions (AKI, LOS)
    - If False: Uses EncoderClassifierInverted with pooling for single prediction (Mortality)
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        time_embed_size,
        layers,
        heads,
        dropout,
        attn_dropout,
        use_mask,
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)

        self.save_hyperparameters()

        # Extract dimensions from dataset
        # input_size is (N, F, T) where N=batch, F=sensors, T=actual_timepoints
        sensors_count = input_size[1]
        actual_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)

        # Check if we should use autoregressive (per-timestep) mode
        try:
            skip_pooling = gin.query_parameter("%TIMESTEP_LEVEL_PREDICTIONS")
        except Exception:
            skip_pooling = False

        # Instantiate appropriate encoder
        if skip_pooling:
            # Per-timestep predictions for tasks like AKI, LOS
            encoder = AutoregressiveEncoderClassifierInverted(
                device=self.device,
                pooling="mean",  # Not used in autoregressive mode
                time_embed_size=time_embed_size,
                layers=layers,
                heads=heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_mask=use_mask,
                sensors_count=sensors_count,
                max_timepoint_count=actual_timepoint_count,
                static_count=static_count,
            )
        else:
            # Single prediction with pooling for tasks like Mortality
            encoder = EncoderClassifierInverted(
                device=self.device,
                pooling="mean",
                time_embed_size=time_embed_size,
                layers=layers,
                heads=heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_mask=use_mask,
                sensors_count=sensors_count,
                max_timepoint_count=actual_timepoint_count,
                static_count=static_count,
            )

        # Compose full prediction model
        self.model = EncoderPredictionInverted(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)


@gin.configurable
class SSL_iTransformer(SSLWrapper):
    """
    Self-Supervised Learning wrapper for iTransformer.

    This enables the inverted transformer to be used for SSL pretraining
    with forecasting tasks, similar to SSL_BAT.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        time_embed_size,
        layers,
        heads,
        dropout,
        attn_dropout,
        use_mask,
        prediction_head=ForecastingHead,
        prediction_head_kwargs={"sensors_count": 48, "forecast_len": 2},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)
        self.save_hyperparameters()

        raise NotImplementedError(
        "SSL_iTransformer is currently under development due to "
        "architectural restrictions regarding timepoints."
        )

        """
        # Extract dimensions from dataset
        # input_size is (N, F, T) where N=batch, F=sensors, T=actual_timepoints
        # For SSL the observation window is variable (12–max_obs per batch), so we
        # size the linear projection to max_obs (24) rather than the first batch's T.
        sensors_count = input_size[1]
        max_obs = 24  # must match SSLPolarsDataset.max_obs
        static_count = kwargs.get("static_count", 4)
        use_static = kwargs.get("use_static", True)
        obs_strategy = kwargs.get("obs_strategy", "both")

        # Instantiate inverted encoder
        encoder = EncoderClassifierInverted(
            device=self.device,
            pooling="mean",
            time_embed_size=time_embed_size,
            layers=layers,
            heads=heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_mask=use_mask,
            sensors_count=sensors_count,
            max_timepoint_count=max_obs,
            static_count=static_count,
            use_static=use_static,
            obs_strategy=obs_strategy,
        )

        # Compose full prediction model
        self.model = EncoderPredictionInverted(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)
    """
        

@gin.configurable
class Autoregressive_iTransformer(CustomDLPredictionWrapper):
    """
    Autoregressive inverted Transformer wrapper for per-timestep predictions.

    This model makes predictions at each timestep using only information
    from that timestep and earlier. It's suitable for tasks like:
    - Length of stay regression (per-timestep)
    - AKI prediction (per-timestep classification)
    - Any task requiring causal predictions over time
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        time_embed_size,
        layers,
        heads,
        dropout,
        attn_dropout,
        use_mask,
        prediction_head=RegressionHead,
        prediction_head_kwargs={"output_dim": 1},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)

        self.save_hyperparameters()

        # Extract dimensions from dataset
        # input_size is (N, F, T) where N=batch, F=sensors, T=actual_timepoints
        sensors_count = input_size[1]
        actual_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)

        # Instantiate autoregressive inverted encoder
        encoder = AutoregressiveEncoderClassifierInverted(
            device=self.device,
            pooling="mean",
            time_embed_size=time_embed_size,
            layers=layers,
            heads=heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_mask=use_mask,
            sensors_count=sensors_count,
            max_timepoint_count=actual_timepoint_count,
            static_count=static_count,
        )

        # Compose full prediction model
        self.model = EncoderPredictionInverted(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

        # Helps CustomDLPredictionWrapper with setting metrics
        # For classification, set num_classes; for regression, this is just a dummy
        if prediction_head == TimeseriesClassificationHead:
            num_classes = prediction_head_kwargs.get("num_classes", 2)
            self.logit = nn.Linear(1, num_classes)  # dummy shape
        else:
            self.logit = nn.Linear(1, 1)  # dummy shape for regression

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)
