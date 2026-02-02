import gin
import torch
import numpy as np
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import CustomDLPredictionWrapper, SSLWrapper
from torch import nn
from x_transformers import Encoder

# Import prediction heads and utility functions from BAT to avoid duplication
from icu_benchmarks.models.dl_models.bat import (
    BinaryClassificationHead,
    RegressionHead,
    ForecastingHead,
    TimeseriesClassificationHead,
    PositionalEncodingTF
)

# TODO: implement autoregressive capabilities for AKI and LOS prediction

# Local pooling functions for RadV Transformer (3D tensors: N, T, E)
def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    Designed for (N, T, E) tensors - pools over time dimension
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
    data_summed = torch.sum(datatensor * mask_expanded, dim=1)

    # find out number of existing timepoints
    data_counts = mask_expanded.sum(1)
    data_counts = torch.clamp(data_counts, min=1e-9)  # put on min clamp

    # Calculate average:
    averaged = data_summed / (data_counts)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked max for time dimension of tensor
    Designed for (N, T, E) tensors - pools over time dimension
    """
    # eliminate all values learned from nonexistant timepoints
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

    datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    maxed = torch.max(datatensor, 1)[0]

    return maxed

# Own encoder prediction class that plugs the encoder to different prediction heads
@gin.configurable
class EncoderPrediction(nn.Module):
    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}
        self.is_autoregressive = isinstance(self.encoder_class, AutoregressiveEncoderRegular)

        # Handling input dim for head depending on the output dimension of the encoding model
        if isinstance(self.encoder_class, (EncoderClassifierRegular, AutoregressiveEncoderRegular)):
            if self.encoder_class.use_static:
                encoder_output_dim = self.encoder_class.sensor_axis_dim + self.encoder_class.static_out
            else:
                encoder_output_dim = self.encoder_class.sensor_axis_dim
        else:
            raise NotImplementedError("Encoder class not supported for automatic head dimension extraction.")

        # Prediction head initialization
        self.head = prediction_head(
            input_dim=encoder_output_dim,
            **self.prediction_head_kwargs
        )

    def forward(self, x, static, time, sensor_mask, **kwargs):
        encoded = self.encoder_class(
            x,
            static,
            time,
            sensor_mask,
            **kwargs
        )

        # Sanity check shape
        if self.is_autoregressive:
            # For autoregressive encoder: features shape is (N, T, E)
            expected_dim = self.encoder_class.sensor_axis_dim
            if self.encoder_class.use_static:
                expected_dim += self.encoder_class.static_out
            if encoded.shape[2] != expected_dim:
                raise ValueError(
                    f"Mismatch between computed input_dim ({expected_dim}) and actual ({encoded.shape[2]})"
                )
        else:
            # For regular encoder: features shape is (N, E)
            expected_dim = self.encoder_class.sensor_axis_dim
            if self.encoder_class.use_static:
                expected_dim += self.encoder_class.static_out
            if encoded.shape[1] != expected_dim:
                raise ValueError(
                    f"Mismatch between computed input_dim ({expected_dim}) and actual ({encoded.shape[1]})"
                )

        return self.head(encoded)

@gin.configurable
class EncoderClassifierRegular(nn.Module):

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
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
        self.obs_strategy = obs_strategy  # BINARY
        self.pooling = pooling
        self.device = device
        self.use_mask = use_mask
        self.sensors_count = sensors_count
        self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count

        if self.obs_strategy in ("indicator_only", "obs_only"):  # BINARY
            # print("binary_values_only")
            self.sensor_axis_dim_in = self.sensors_count
        else:  # BINARY
            self.sensor_axis_dim_in = 2 * self.sensors_count

        self.sensor_axis_dim = self.sensor_axis_dim_in
        if self.sensor_axis_dim % 2 != 0:
            self.sensor_axis_dim += 1

        # self.time_axis_dim_in = 2 * self.max_timepoint_count
        # self.time_axis_dim = min(2 * self.max_timepoint_count, 500)
        self.static_out = self.static_count + 4

        self.attn_layers_2 = Encoder(
            dim=self.sensor_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
        )

        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)

        self.use_static = use_static

        if self.use_static:
            self.static_embedding = nn.Linear(self.static_count, self.static_out)
            self.nonlinear_merger = nn.Linear(
                self.sensor_axis_dim + self.static_out,
                self.sensor_axis_dim + self.static_out,
            )
            self.classifier = nn.Linear(
                self.sensor_axis_dim + self.static_out, num_classes
            )
        else:
            self.nonlinear_merger = nn.Linear(
                self.sensor_axis_dim,
                self.sensor_axis_dim,
            )
            self.classifier = nn.Linear(self.sensor_axis_dim, num_classes)

        self.pos_encoder = PositionalEncodingTF(self.sensor_axis_dim)

    def forward(self, x, static, time, sensor_mask, **kwargs):

        x_time = torch.clone(x)  # (N, F, T)
        x_time = torch.permute(x_time, (0, 2, 1))  # (N, T, F)
        mask = (
            torch.count_nonzero(x_time, dim=2)
        ) > 0  # mask for sum of all sensors for each person/at each timepoint

        # add indication for missing sensor values
        x_sensor_mask = torch.clone(sensor_mask)  # (N, F, T)
        x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)
        if self.obs_strategy == "indicator_only":  # Binary
            x_time = x_sensor_mask.float()  # Binary
        elif self.obs_strategy == "obs_only":
            x_time = x_time
        elif self.obs_strategy == "both":  # Binary
            x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F) #Binary
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")
        # make sensor embeddings
        x_time = self.sensor_embedding(x_time)  # (N, T, F)
        # add positional encodings
        with torch.no_grad():
            pe = self.pos_encoder(time).to(x_time.device)  # (N, T, pe)
        x_time = torch.add(x_time, pe)  # (N, T, F)
        # run time attention
        mask_attention = mask if self.use_mask else None
        if self.return_intermediates:
            x_time, time_intermediates = self.attn_layers_2(
                x_time, mask=mask_attention, return_hiddens=True
            )
        else:
            x_time = self.attn_layers_2(x_time, mask=mask_attention)

        # Apply pooling
        mask_pooling = mask if self.use_mask else None
        if self.pooling == "mean":
            if mask_pooling is not None:
                x_time = masked_mean_pooling(x_time, mask_pooling)
            else:
                x_time = torch.mean(x_time, dim=1)
        elif self.pooling == "median":
            x_time = torch.median(x_time, dim=1)[0]
        elif self.pooling == "sum":
            x_time = torch.sum(x_time, dim=1)  # sum on time
        elif self.pooling == "max":
            if mask_pooling is not None:
                x_time = masked_max_pooling(x_time, mask_pooling)
            else:
                x_time = torch.max(x_time, dim=1)[0]

        # concatenate pooled attended tensors
        if self.use_static:
            static = self.static_embedding(static)
            x_merged = torch.cat((x_time, static), axis=1)
        else:
            x_merged = x_time

        nonlinear_merged = self.nonlinear_merger(x_merged).relu()

        return nonlinear_merged

        """
        # classify!
        if self.return_intermediates:
            return None, time_intermediates.attn_intermediates[0].post_softmax_attn
        return self.classifier(nonlinear_merged)
        """

@gin.configurable
class AutoregressiveEncoderRegular(nn.Module):
    """
    Autoregressive version of the RADV transformer encoder.
    Returns per-timestep representations instead of pooled representations.

    For each timestep t, processes input with all future timepoints (>t) masked,
    ensuring the representation at timestep t only depends on information up to t.
    """

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
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

        if self.obs_strategy in ("indicator_only", "obs_only"):
            self.sensor_axis_dim_in = self.sensors_count
        else:
            self.sensor_axis_dim_in = 2 * self.sensors_count

        self.sensor_axis_dim = self.sensor_axis_dim_in
        if self.sensor_axis_dim % 2 != 0:
            self.sensor_axis_dim += 1

        self.static_out = self.static_count + 4

        # Use regular Encoder with explicit future masking
        self.attn_layers = Encoder(
            dim=self.sensor_axis_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
            attn_flash=False,
        )

        self.sensor_embedding = nn.Linear(self.sensor_axis_dim_in, self.sensor_axis_dim)
        self.use_static = use_static

        if self.use_static:
            print("use static in autoregressive encoder")
            self.static_embedding = nn.Linear(self.static_count, self.static_out)
            self.nonlinear_merger = nn.Linear(
                self.sensor_axis_dim + self.static_out,
                self.sensor_axis_dim + self.static_out,
            )
        else:
            self.nonlinear_merger = nn.Linear(
                self.sensor_axis_dim,
                self.sensor_axis_dim,
            )

        self.pos_encoder = PositionalEncodingTF(self.sensor_axis_dim)

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

        # Calculate output dimension
        output_dim = (
            self.sensor_axis_dim + self.static_out
            if self.use_static
            else self.sensor_axis_dim
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
            x_time = torch.clone(x_masked)  # (N, F, T)
            x_time = torch.permute(x_time, (0, 2, 1))  # (N, T, F)

            # Make mask of all empty (missing) timepoints
            mask = (torch.count_nonzero(x_time, dim=2)) > 0  # (N, T)

            # Add indication for missing sensor values
            x_sensor_mask = torch.clone(sensor_mask_masked)  # (N, F, T)
            x_sensor_mask = torch.permute(x_sensor_mask, (0, 2, 1))  # (N, T, F)

            if self.obs_strategy == "indicator_only":
                x_time = x_sensor_mask.float()
            elif self.obs_strategy == "obs_only":
                x_time = x_time
            elif self.obs_strategy == "both":
                x_time = torch.cat([x_time, x_sensor_mask], axis=2)  # (N, T, 2F)
            else:
                raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

            # Free memory
            del x_sensor_mask

            # Make sensor embeddings
            x_time = self.sensor_embedding(x_time)  # (N, T, E)

            # Add positional encodings (only for the time slice up to t)
            with torch.no_grad():
                pe = self.pos_encoder(time).to(x_time.device)  # (N, T, E)
            x_time = torch.add(x_time, pe)  # (N, T, E)
            del pe

            # Run attention
            mask_attention = mask if self.use_mask else None
            if self.return_intermediates:
                x_time, time_intermediates = self.attn_layers(
                    x_time, mask=mask_attention, return_hiddens=True
                )
            else:
                x_time = self.attn_layers(x_time, mask=mask_attention)  # (N, T, E)

            # Pool only the current timestep representation
            # Take the representation at timestep t
            x_current = x_time[:, t, :]  # (N, E)

            # Concatenate with static features if used
            if self.use_static:
                static_embedded = self.static_embedding(static)  # (N, static_out)
                x_merged = torch.cat((x_current, static_embedded), axis=1)
            else:
                x_merged = x_current

            # Apply nonlinear transformation
            nonlinear_merged = self.nonlinear_merger(x_merged).relu()

            # Store the output for this timestep
            outputs[:, t, :] = nonlinear_merged

        if self.return_intermediates:
            return time_intermediates.attn_intermediates[0].post_softmax_attn

        return outputs  # (N, T, E)

@gin.configurable
class RadVTransformer(CustomDLPredictionWrapper):
    """
    RADV Transformer wrapper for YAIB framework.

    Automatically selects between regular and autoregressive encoder based on
    the TIMESTEP_LEVEL_PREDICTIONS gin parameter:
    - If True: Uses AutoregressiveEncoderRegular for per-timestep predictions (AKI, LOS)
    - If False: Uses EncoderClassifierRegular with pooling for single prediction (Mortality)
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        value_embed_size,
        layers,
        heads,
        dropout,
        attn_dropout,
        use_mask,
        pooling="max",
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(
            lr=lr,
            optimizer=optimizer,
            *args,
            **kwargs
        )

        self.save_hyperparameters()

        # Extract dimensions from dataset
        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)  # fallback if static shape isn't passed

        # Check if we should use autoregressive (per-timestep) mode
        try:
            skip_pooling = gin.query_parameter("%TIMESTEP_LEVEL_PREDICTIONS")
        except Exception:
            skip_pooling = False

        # Instantiate appropriate encoder
        if skip_pooling:
            # Per-timestep predictions for tasks like AKI, LOS
            encoder = AutoregressiveEncoderRegular(
                device=self.device,
                pooling=pooling,  # Not used in autoregressive mode, but kept for consistency
                layers=layers,
                heads=heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_mask=use_mask,
                sensors_count=sensors_count,
                max_timepoint_count=max_timepoint_count,
                static_count=static_count,
            )
        else:
            # Single prediction with pooling for tasks like Mortality
            encoder = EncoderClassifierRegular(
                device=self.device,
                pooling=pooling,
                layers=layers,
                heads=heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_mask=use_mask,
                sensors_count=sensors_count,
                max_timepoint_count=max_timepoint_count,
                static_count=static_count,
            )

        # Compose full prediction model
        self.model = EncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs
        )

        # Helps CustomDLPredictionWrapper with setting binary classification metrics
        self.logit = nn.Linear(1, prediction_head_kwargs.get("num_classes", 2))  # dummy shape

    def forward(self, data, static, time, sensor_mask):
        return self.model(
            data,
            static,
            time,
            sensor_mask
        )

@gin.configurable
class SSL_RadVTransformer(SSLWrapper):
    """
    Self-supervised learning wrapper for RADV Transformer.

    Uses the same architecture as RadVTransformer but wrapped in SSLWrapper
    for self-supervised pretraining tasks like forecasting.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        value_embed_size,
        layers,
        heads,
        dropout,
        attn_dropout,
        use_mask,
        pooling="max",
        prediction_head=ForecastingHead,
        prediction_head_kwargs={"sensors_count": 48, "forecast_len": 2},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)
        self.save_hyperparameters()

        # Extract dimensions from dataset
        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)  # fallback if static shape isn't passed

        # Instantiate encoder (use regular encoder with pooling for SSL)
        encoder = EncoderClassifierRegular(
            device=self.device,
            pooling=pooling,
            layers=layers,
            heads=heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_mask=use_mask,
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
        )

        # Compose full prediction model
        self.model = EncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)