"""
Deep Set Attention model for ICU time series prediction.

Based on SEFT (Set Functions for Time Series) architecture with attention mechanisms.
Adapted from Patient_Journey_Classification repository for YAIB framework.

Reference: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
"""

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Callable
import math
import inspect

import torch_scatter

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import CustomDLPredictionWrapper
from icu_benchmarks.models.dl_models.bat import BinaryClassificationHead, RegressionHead, PositionalEncodingTF


# Utility Functions (from seft_utils.py)

def segment_softmax(data, segment_ids, eps=1e-7):
    """Compute softmax within segments."""

    max_values = torch_scatter.scatter_max(data, segment_ids, dim=0)[0]
    max_values = max_values[segment_ids]
    normalized = data - max_values

    numerator = torch.exp(normalized)
    denominator = torch_scatter.scatter_add(numerator, segment_ids, dim=0)
    denominator = denominator[segment_ids]

    softmax = numerator / (denominator + eps)

    return softmax


class PaddedToSegments(nn.Module):
    """Convert a padded tensor with mask to a stacked tensor with segments."""

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        valid_observations = torch.nonzero(mask).squeeze().to(inputs.device)
        collected_values = inputs[mask]
        return collected_values, valid_observations[:, 0]


class Segmentpooling(nn.Module):
    """Pooling operation on segments."""
    def __init__(self, pooling_fn: str = "sum", cumulative: bool = False):
        super().__init__()
        self.cumulative = cumulative
        #self.pooling_fn_name = pooling_fn
        self.pooling_fn = self._get_pooling_fn(pooling_fn)

    def _get_pooling_fn(self, pooling_fn: str) -> Callable:
        if not self.cumulative:
            if pooling_fn == "sum":
                return lambda x, ids: torch.scatter_add(
                    torch.zeros(ids.max() + 1, *x.shape[1:], device=x.device),
                    0,
                    ids.unsqueeze(-1).expand(-1, x.shape[1]),
                    x,
                )
            elif pooling_fn == "mean":
                return lambda x, ids: torch.scatter_add(
                    torch.zeros(ids.max() + 1, *x.shape[1:], device=x.device),
                    0,
                    ids.unsqueeze(-1).expand(-1, x.shape[1]),
                    x,
                ) / torch.bincount(ids).float().unsqueeze(-1)
            elif pooling_fn == "max":
                return lambda x, ids: torch.scatter_reduce(
                    torch.full(
                        (ids.max() + 1, *x.shape[1:]), float("-inf"), device=x.device
                    ),
                    0,
                    ids.unsqueeze(-1).expand(-1, x.shape[1]),
                    x,
                    reduce="amax",
                )
            else:
                raise ValueError(f"Invalid pooling function")
        else:
            if pooling_fn == "sum":
                return lambda x, ids: torch.cumsum(x, dim=0)
            elif pooling_fn == "mean":
                return lambda x, ids: torch.cumsum(x, dim=0) / torch.arange(
                    1, x.size(0) + 1, device=x.device
                ).unsqueeze(-1)
            else:
                raise ValueError(f"Invalid pooling function for cumulative mode")

    def forward(self, data: torch.Tensor, segment_ids: torch.Tensor):
        return self.pooling_fn(data, segment_ids)

# Helper function to map activation names to PyTorch functions
def get_activation_fn(activation_name):
    """Map activation names to PyTorch functions."""
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name is None:
        return None
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

# Custom function to initialize weights
def initialize_weights(layer, initializer_name):
    """Initialize layer weights."""
    if initializer_name == "he_uniform":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    elif initializer_name == "glorot_uniform":
        nn.init.xavier_uniform_(layer.weight)
    else:
        raise ValueError(f"Unsupported initializer: {initializer_name}")


class MySequential(nn.Module):
    """Sequential module that can handle segment_ids parameter."""
    def __init__(self, layers):
        super(MySequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, segment_ids=None):
        outputs = inputs  # handle the corner case where self.layers is empty
        for layer in self.layers:
            # Check if the layer's forward method supports 'segment_ids'
            kwargs = {}
            sig = inspect.signature(layer.forward)
            if "segment_ids" in sig.parameters:
                kwargs["segment_ids"] = segment_ids

            # Call the layer with or without segment_ids depending on its signature
            outputs = layer(inputs, **kwargs) if kwargs else layer(inputs)
            # Prepare the input for the next layer
            inputs = outputs

        return outputs

# Updated function to handle activation and kernel_initializer (dense_kwargs)
def build_dense_dropout_model(input_size, n_layers, width, dropout, dense_kwargs):
    """
    Build a Sequential model composed of stacked Linear and Dropout blocks.

    Args:
        input_size: Input dimension
        n_layers: Number of layers to stack
        width: Width of the layers
        dropout: Dropout probability
        dense_kwargs: Dictionary for additional layer settings (activation, initializer)

    Returns:
        MySequential model of stacked Linear Dropout layers
    """
    layers = []

    activation_fn = get_activation_fn(dense_kwargs.get("activation", None))
    initializer = dense_kwargs.get("kernel_initializer", None)

    for i in range(n_layers):
        if i == 0:
            linear_layer = nn.Linear(input_size, width)
        else:
            linear_layer = nn.Linear(width, width)

        # Apply kernel initializer if specified
        if initializer:
            initialize_weights(linear_layer, initializer)

        layers.append(linear_layer)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Add activation function if specified
        if activation_fn:
            layers.append(activation_fn)

    return MySequential(layers)

# where deep_set_attention.py starts
# Note: PositionalEncodingTF is now imported from bat.py to avoid redundancy

# Attention Layers

class SetAttentionLayer(nn.Module):
    """
    Set attention layer with multi-head attention mechanism.
    """
    def __init__(
        self,
        n_layers: int = 2,
        width: int = 128,
        latent_width: int = 128,
        pooling_function: str = "mean",
        dot_prod_dim: int = 64,
        n_heads: int = 4,
        attn_dropout: float = 0.3,
        psi_input_size: int = 32,
    ):
        super().__init__()
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi_input_size = psi_input_size

        dense_options = {"activation": "relu", "kernel_initializer": "he_uniform"}

        self.psi = build_dense_dropout_model(
            psi_input_size, n_layers, width, 0.0, dense_kwargs=dense_options
        )
        self.psi.layers.append(nn.LazyLinear(latent_width))
        self.psi_pooling = Segmentpooling(pooling_function)
        self.rho = nn.LazyLinear(latent_width)

        self.W_k = nn.Parameter(
            torch.empty(psi_input_size + latent_width, self.dot_prod_dim * self.n_heads)
        )
        nn.init.kaiming_uniform_(self.W_k, a=math.sqrt(5))

        # Weight W_q initialized to zeros
        self.W_q = nn.Parameter(torch.zeros(self.n_heads, self.dot_prod_dim))

    def forward(
        self, inputs: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor
    ) -> List[torch.Tensor]:
        encoded = self.psi(inputs)
        agg = self.psi_pooling(encoded, segment_ids)
        agg = self.rho(agg)
        agg_scattered = agg[segment_ids]
        combined = torch.cat([inputs, agg_scattered], dim=-1)
        keys = torch.matmul(combined, self.W_k).view(
            -1, self.n_heads, 1, self.dot_prod_dim
        )
        queries = self.W_q.unsqueeze(0).unsqueeze(-1)

        preattn = torch.matmul(keys, queries) / (self.dot_prod_dim**0.5)
        preattn = preattn.squeeze(-1)

        if self.training and self.attn_dropout > 0:
            mask = torch.rand_like(preattn) < self.attn_dropout
            preattn = preattn.masked_fill(mask, -1e9)

        return [
            segment_softmax(pre_attn, segment_ids) for pre_attn in preattn.unbind(1)
        ]


class DeepSetAttentionEncoder(nn.Module):
    """
    Deep Set Attention encoder for irregularly sampled time series.

    Uses set-based attention mechanisms to handle variable-length,
    irregularly sampled observations.
    """
    def __init__(
        self,
        device="cpu",
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        seft_n_phi_layers=2,
        seft_phi_width=128,
        seft_n_psi_layers=2,
        seft_psi_width=128,
        seft_psi_latent_width=128,
        seft_dot_prod_dim=64,
        heads=4,
        attn_dropout=0.3,
        seft_latent_width=128,
        seft_phi_dropout=0.2,
        seft_n_rho_layers=2,
        seft_rho_width=128,
        seft_rho_dropout=0.2,
        seft_max_timescales=500,
        seft_n_positional_dims=16,
        obs_strategy="both",
        use_static=True,
        **kwargs
    ):
        super().__init__()

        self.obs_strategy = obs_strategy
        self.use_static = use_static
        self.device = device
        self.sensors_count = sensors_count
        self.n_modalities = sensors_count

        if sensors_count > 100:
            self.modality_embedding = nn.LazyLinear(64)
            self.n_modalities = 64

        phi_input_dim = self.n_modalities + seft_n_positional_dims + 1

        dense_options = {"activation": "relu", "kernel_initializer": "he_uniform"}

        self.phi = build_dense_dropout_model(
            phi_input_dim,
            seft_n_phi_layers,
            seft_phi_width,
            seft_phi_dropout,
            dense_kwargs=dense_options,
        )
        self.phi.layers.append(nn.LazyLinear(seft_latent_width))
        self.latent_width = seft_latent_width
        self.n_heads = heads

        self.positional_encoding = PositionalEncodingTF(
            d_model=seft_n_positional_dims, max_len=seft_max_timescales
        )

        self.attention = SetAttentionLayer(
            seft_n_psi_layers,
            seft_psi_width,
            seft_psi_latent_width,
            dot_prod_dim=seft_dot_prod_dim,
            n_heads=heads,
            attn_dropout=attn_dropout,
            psi_input_size=phi_input_dim,
        )

        self.pooling = Segmentpooling(pooling_fn="sum", cumulative=False)
        if self.use_static:
            self.demo_encoder = nn.Sequential(
                nn.LazyLinear(seft_phi_width), # First dense layer
                nn.ReLU(), # ReLU activation
                nn.LazyLinear(phi_input_dim), # Second dense layer
            )

        self.rho = build_dense_dropout_model(
            heads * seft_latent_width,
            seft_n_rho_layers,
            seft_rho_width,
            seft_rho_dropout,
            dense_kwargs=dense_options,
        )

        self.to_segments = PaddedToSegments()

        # Store output dimension for prediction head
        self.output_dim = seft_rho_width

    def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:
        """
        Forward pass of Deep Set Attention encoder.

        Args:
            x: (N, F, T) - batch, sensors, time
            static: (N, static_count) - static features
            time: (N, T) - time encodings
            sensor_mask: (N, F, T) - mask for missing values

        Returns:
            encoded: (N, output_dim) - encoded representation
        """

        if self.obs_strategy == "both":
            pass
        elif self.obs_strategy == "indicator_only":
            x = sensor_mask.clone().float()
        elif self.obs_strategy == "obs_only":
            sensor_mask = torch.ones_like(sensor_mask)
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

        x = x.permute(0, 2, 1)
        sensor_mask = sensor_mask.permute(0, 2, 1)

        time, x, sensor_mask, static, lengths = self.flatten_unaligned_measurements(
            x, static, time, sensor_mask
        )

        time = time.squeeze(-1)

        transformed_times = self.positional_encoding(time).squeeze(1)
        transformed_measurements = F.one_hot(sensor_mask, self.n_modalities).float()

        combined_values = torch.cat(
            (transformed_times, x, transformed_measurements), dim=-1
        )

        if self.use_static:
            demo_encoded = self.demo_encoder(static)
            combined_with_demo = torch.cat(
                [demo_encoded.unsqueeze(1), combined_values], dim=1
            )
        else:
            combined_with_demo = combined_values

        if lengths.dim() == 2:
            lengths = lengths.squeeze(-1)

        mask = torch.arange(combined_with_demo.size(1), device=combined_with_demo.device).unsqueeze(0) < (
            lengths + 1
        ).unsqueeze(1)
        collected_values, segment_ids = self.to_segments(combined_with_demo, mask)

        encoded = self.phi(collected_values)
        attentions = self.attention(collected_values, segment_ids, lengths)

        weighted_values = [encoded * attention for attention in attentions]

        pooled_values = self.pooling(
            torch.cat(weighted_values, dim=-1), segment_ids
        )
        return self.rho(pooled_values)

    def flatten_unaligned_measurements(self, x, static, time, sensor_mask):
        """
        Flatten irregularly sampled measurements into a continuous representation.

        Args:
            x: (N, T, F) - values
            static: (N, static_count) - static features
            time: (N, T) - time points
            sensor_mask: (N, T, F) - observation mask

        Returns:
            Tuple of flattened and padded tensors
        """
        all_gather_y = []
        all_demo = []
        all_gather_x = []
        all_y_indices = []
        all_lengths = []

        for batch_ind in range(x.shape[0]):
            demo = static[batch_ind]
            X = time[batch_ind]
            Y = x[batch_ind]
            measurements = sensor_mask[batch_ind]

            X = X.unsqueeze(-1)
            measurement_positions = torch.nonzero(measurements)
            X_indices = measurement_positions[:, 0]
            Y_indices = measurement_positions[:, 1]

            gathered_X = X[X_indices]
            gathered_Y = Y[
                measurement_positions[:, 0], measurement_positions[:, 1]
            ].unsqueeze(-1)

            length = X_indices.shape[0]

            all_gather_y.append(gathered_Y)
            all_demo.append(demo)
            all_gather_x.append(gathered_X)
            all_y_indices.append(Y_indices)
            all_lengths.append(length)

        # Pad tensors to the same length
        padded_gather_y = pad_sequence(
            all_gather_y, batch_first=True, padding_value=0.0
        )
        padded_gather_x = pad_sequence(
            all_gather_x, batch_first=True, padding_value=0.0
        )
        padded_y_indices = pad_sequence(
            all_y_indices, batch_first=True, padding_value=0
        )

        # Convert all_demos and all_lengths into tensors
        all_demo = torch.stack(all_demo) # No padding needed for demo
        all_lengths = torch.tensor(all_lengths, device=x.device)  # Lengths are already a 1D tensor

        return padded_gather_x, padded_gather_y, padded_y_indices, all_demo, all_lengths


class DeepSetAttentionEncoderPrediction(nn.Module):
    """
    Combines DeepSetAttention encoder with prediction head.
    """
    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}

        # Input dim is the output of the rho network
        self.input_dim = self.encoder_class.output_dim

        # Initialize prediction head
        self.head = self.prediction_head(
            input_dim=self.input_dim,
            **self.prediction_head_kwargs
        )

    def forward(self, x, static, time, sensor_mask):
        features = self.encoder_class(x, static, time, sensor_mask)

        # Sanity check shape
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Mismatch between computed input_dim ({self.input_dim}) and actual ({features.shape[1]})"
            )

        return self.head(features)


@gin.configurable
class DeepSetAttentionModel(CustomDLPredictionWrapper):
    """
    Deep Set Attention wrapper for YAIB framework.

    This model uses set-based attention mechanisms to handle
    irregularly sampled time series data from ICU environments.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        seft_n_phi_layers=2,
        seft_phi_width=128,
        seft_n_psi_layers=2,
        seft_psi_width=128,
        seft_psi_latent_width=128,
        seft_dot_prod_dim=64,
        heads=4,
        attn_dropout=0.3,
        seft_latent_width=128,
        seft_phi_dropout=0.2,
        seft_n_rho_layers=2,
        seft_rho_width=128,
        seft_rho_dropout=0.2,
        seft_max_timescales=500,
        seft_n_positional_dims=16,
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs=None,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)

        if prediction_head_kwargs is None:
            prediction_head_kwargs = {"num_classes": 2}

        self.save_hyperparameters()

        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)

        encoder = DeepSetAttentionEncoder(
            device=self.device,
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
            seft_n_phi_layers=seft_n_phi_layers,
            seft_phi_width=seft_phi_width,
            seft_n_psi_layers=seft_n_psi_layers,
            seft_psi_width=seft_psi_width,
            seft_psi_latent_width=seft_psi_latent_width,
            seft_dot_prod_dim=seft_dot_prod_dim,
            heads=heads,
            attn_dropout=attn_dropout,
            seft_latent_width=seft_latent_width,
            seft_phi_dropout=seft_phi_dropout,
            seft_n_rho_layers=seft_n_rho_layers,
            seft_rho_width=seft_rho_width,
            seft_rho_dropout=seft_rho_dropout,
            seft_max_timescales=seft_max_timescales,
            seft_n_positional_dims=seft_n_positional_dims,
        )

        self.model = DeepSetAttentionEncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)
