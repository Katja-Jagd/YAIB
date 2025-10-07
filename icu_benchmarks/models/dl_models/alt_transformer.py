import gin
import torch
from torch import nn
from x_transformers import Encoder

from icu_benchmarks.models.dl_models.layers import (
    PositionalEncodingScaled,
    masked_mean_pooling,
    masked_max_pooling,
)
from icu_benchmarks.models.wrappers import DLPredictionWrapper
from icu_benchmarks.constants import RunMode


@gin.configurable
class AltTransformer(DLPredictionWrapper):
    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,          # (B, T, F)
        num_classes,
        sensors_count=37,
        static_count=8,
        layers=1,
        heads=1,
        dropout=0.2,
        attn_dropout=0.2,
        pooling="mean",
        use_static=True,
        obs_strategy="both",
        device="cpu",
        return_intermediates=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pooling = pooling
        self.device = device
        self.return_intermediates = return_intermediates
        self.obs_strategy = obs_strategy
        self.use_static = use_static

        # RADV input dims
        if obs_strategy in ("indicator_only", "obs_only"):
            sensor_dim_in = sensors_count
        else:
            sensor_dim_in = 2 * sensors_count

        sensor_dim = sensor_dim_in if sensor_dim_in % 2 == 0 else sensor_dim_in + 1
        self.sensor_embedding = nn.Linear(sensor_dim_in, sensor_dim)

        self.pos_encoder = PositionalEncodingTF(sensor_dim)

        # Encoder layers (x_transformers)
        self.encoder = Encoder(
            dim=sensor_dim,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
        )

        # Static branch
        if use_static:
            self.static_embedding = nn.Linear(static_count, static_count + 4)
            self.nonlinear_merger = nn.Linear(sensor_dim + static_count + 4,
                                              sensor_dim + static_count + 4)
            self.classifier = nn.Linear(sensor_dim + static_count + 4, num_classes)
        else:
            self.nonlinear_merger = nn.Linear(sensor_dim, sensor_dim)
            self.classifier = nn.Linear(sensor_dim, num_classes)

    def forward(self, x, static=None, time=None, sensor_mask=None):
        # x: (B, T, F)
        # time: (B, T)
        # sensor_mask: (B, T, F)

        mask = (x.abs().sum(dim=2) > 0)  # valid timesteps

        if self.obs_strategy == "indicator_only":
            x = sensor_mask.float()
        elif self.obs_strategy == "both":
            x = torch.cat([x, sensor_mask], dim=2)

        # sensor embedding
        x = self.sensor_embedding(x)

        # positional encoding
        if time is not None:
            pe = self.pos_encoder(time).to(x.device)
            x = x + pe

        # transformer encoder
        x = self.encoder(x, mask=mask)

        # pooling
        if self.pooling == "mean":
            x = masked_mean_pooling(x, mask)
        elif self.pooling == "max":
            x = masked_max_pooling(x, mask)
        else:
            x = x.mean(dim=1)

        # merge static
        if self.use_static and static is not None:
            static = self.static_embedding(static)
            x = torch.cat([x, static], dim=1)

        x = self.nonlinear_merger(x).relu()
        return self.classifier(x)
