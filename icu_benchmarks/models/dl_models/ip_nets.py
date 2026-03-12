"""
Interpolation-Prediction Networks (IP-Nets) for ICU time series prediction.

Based on: https://github.com/mlds-lab/interp-net
Paper: "Interpolation-Prediction Networks for Irregularly Sampled Time Series"

Adapted for YAIB framework.
"""

import gin
import torch
import torch.nn as nn
import torch.jit as jit
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import CustomDLPredictionWrapper, SSLWrapper
from icu_benchmarks.models.dl_models.bat import BinaryClassificationHead, RegressionHead, TimeseriesClassificationHead, ForecastingHead


def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """

    if mask is not None:
        # eliminate all values learned from nonexistant timepoints
        mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
        data_summed = torch.sum(datatensor * mask_expanded, dim=1)

        # find out number of existing timepoints
        data_counts = mask_expanded.sum(1)
        data_counts = torch.clamp(data_counts, min=1e-9)  # put on min clamp

        # Calculate average:
        averaged = data_summed / (data_counts)
    else:
        averaged = datatensor.mean(dim=1)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """

    if mask is not None:

        # eliminate all values learned from nonexistant timepoints
        mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

        datatensor[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    maxed = torch.max(datatensor, 1)[0]

    return maxed


class SingleChannelInterp(jit.ScriptModule):
    """
    Single-channel interpolation layer.
    Performs weighted interpolation within each channel independently.
    """
    def __init__(self, input_dim):
        super(SingleChannelInterp, self).__init__()
        assert input_dim % 3 == 0
        self.d_dim = input_dim // 3
        self.kernel = nn.Parameter(torch.zeros(self.d_dim))

    @jit.script_method
    def forward(self, x, interpolation_grid, reconstruction: bool = False):
        x_t = x[:, : self.d_dim, :]
        m = x[:, self.d_dim : 2 * self.d_dim, :]
        d = x[:, 2 * self.d_dim : 3 * self.d_dim, :]
        time_stamp = x.shape[2]

        if reconstruction:
            ref_t = d.unsqueeze(2).expand(-1, -1, time_stamp, -1)
            output_dim = time_stamp
        else:
            ref_t = interpolation_grid.unsqueeze(1).unsqueeze(1)  # Expand grid
            output_dim = interpolation_grid.shape[-1]

        d = d.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        mask = m.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        x_t = x_t.unsqueeze(-1).expand(-1, -1, -1, output_dim)

        norm = (d - ref_t) ** 2
        a = torch.ones((self.d_dim, time_stamp, output_dim), device=x.device)
        pos_kernel = torch.log(1 + torch.exp(self.kernel))  # Positive kernel
        alpha = a * pos_kernel.view(-1, 1, 1)

        w = torch.logsumexp(-alpha * norm + torch.log(mask + 1e-9), dim=2)
        w1 = w.unsqueeze(2).expand(-1, -1, time_stamp, -1)
        w1 = torch.exp(-alpha * norm + torch.log(mask + 1e-9) - w1)

        y = (w1 * x_t).sum(dim=2)

        if reconstruction:
            rep1 = torch.cat([y, w], dim=1)
        else:
            w_t = torch.logsumexp(-10.0 * alpha * norm + torch.log(mask + 1e-9), dim=2)
            w_t = w_t.unsqueeze(2).expand(-1, -1, time_stamp, -1)
            w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask + 1e-9) - w_t)
            y_trans = (w_t * x_t).sum(dim=2)
            rep1 = torch.cat([y, w, y_trans], dim=1)

        return rep1


class CrossChannelInterp(jit.ScriptModule):
    """
    Cross-channel interpolation layer.
    Learns relationships between different channels/sensors.
    """
    def __init__(self, input_dim):
        super(CrossChannelInterp, self).__init__()
        self.d_dim = input_dim // 3
        self.cross_channel_interp = nn.Parameter(torch.eye(self.d_dim))

    @jit.script_method
    def forward(self, x, reconstruction: bool = False):
        output_dim = x.shape[-1]
        y = x[:, : self.d_dim, :]
        w = x[:, self.d_dim : 2 * self.d_dim, :]
        intensity = torch.exp(w)

        y = y.transpose(1, 2)
        w = w.transpose(1, 2)
        w2 = w

        w = w.unsqueeze(-1).expand(-1, -1, -1, self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)

        mean = y.mean(dim=1, keepdim=True)
        mean = mean.expand(-1, output_dim, -1)
        w2 = torch.matmul(w * (y - mean), self.cross_channel_interp) + mean

        rep1 = w2.transpose(1, 2)

        if not reconstruction:
            y_trans = x[:, 2 * self.d_dim : 3 * self.d_dim, :]
            y_trans = y_trans - rep1
            rep1 = torch.cat([rep1, intensity, y_trans], dim=1)

        return rep1


class AutoregressiveSingleChannelInterp(jit.ScriptModule):
    """
    Autoregressive single-channel interpolation layer.
    Only uses past timepoints for interpolation at each position.
    """
    def __init__(self, input_dim):
        super(AutoregressiveSingleChannelInterp, self).__init__()
        assert input_dim % 3 == 0
        self.d_dim = input_dim // 3
        self.kernel = nn.Parameter(torch.zeros(self.d_dim))

    @jit.script_method
    def forward(self, x, interpolation_grid, reconstruction: bool = False):
        x_t = x[:, : self.d_dim, :]
        m = x[:, self.d_dim : 2 * self.d_dim, :]
        d = x[:, 2 * self.d_dim : 3 * self.d_dim, :]
        time_stamp = x.shape[2]

        if reconstruction:
            # For reconstruction, use causal mask
            ref_t = d.unsqueeze(2).expand(-1, -1, time_stamp, -1)
            output_dim = time_stamp
        else:
            ref_t = interpolation_grid.unsqueeze(1).unsqueeze(1)
            output_dim = interpolation_grid.shape[-1]

        d = d.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        mask = m.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        x_t = x_t.unsqueeze(-1).expand(-1, -1, -1, output_dim)

        norm = (d - ref_t) ** 2
        a = torch.ones((self.d_dim, time_stamp, output_dim), device=x.device)
        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha = a * pos_kernel.view(-1, 1, 1)

        # Create causal mask: only allow attention to past timepoints
        # For each output position j, only attend to input positions i where time[i] <= time[j]
        if not reconstruction:
            # ref_t has shape (batch, d_dim, 1, output_dim)
            # d has shape (batch, d_dim, time_stamp, output_dim)
            # Create mask where d <= ref_t (past timepoints)
            causal_mask = (d <= ref_t).float()
            mask = mask * causal_mask
        else:
            # For reconstruction, mask future positions
            # Create lower triangular mask for causal attention
            time_indices = torch.arange(time_stamp, device=x.device)
            causal_mask = (time_indices.unsqueeze(0) <= time_indices.unsqueeze(1)).float()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(x.shape[0], self.d_dim, -1, -1)
            mask = mask * causal_mask

        w = torch.logsumexp(-alpha * norm + torch.log(mask + 1e-9), dim=2)
        w1 = w.unsqueeze(2).expand(-1, -1, time_stamp, -1)
        w1 = torch.exp(-alpha * norm + torch.log(mask + 1e-9) - w1)

        y = (w1 * x_t).sum(dim=2)

        if reconstruction:
            rep1 = torch.cat([y, w], dim=1)
        else:
            w_t = torch.logsumexp(-10.0 * alpha * norm + torch.log(mask + 1e-9), dim=2)
            w_t = w_t.unsqueeze(2).expand(-1, -1, time_stamp, -1)
            w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask + 1e-9) - w_t)
            y_trans = (w_t * x_t).sum(dim=2)
            rep1 = torch.cat([y, w, y_trans], dim=1)

        return rep1


class IPNetsEncoder(nn.Module):
    """
    IP-Nets encoder combining single-channel and cross-channel interpolation
    with GRU for temporal modeling.
    """
    def __init__(
        self,
        device="cpu",
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        recurrent_n_units=128,
        ipnets_imputation_stepsize=0.25,
        dropout=0.2,
        recurrent_dropout=0.2,
        ipnets_reconst_fraction=0.5,
        obs_strategy="both",
        use_static=True,
        pooling="hidden",  # One of hidden, mean, max
        **kwargs
    ):
        super(IPNetsEncoder, self).__init__()

        self.obs_strategy = obs_strategy
        self.use_static = use_static
        self.pool = pooling
        self.device = device

        self.imputation_stepsize = ipnets_imputation_stepsize
        self.reconst_fraction = ipnets_reconst_fraction
        self.eps = 1e-9
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.n_units = recurrent_n_units
        self.sensors_count = sensors_count
        self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count

        self.interp_dim = sensors_count * 3

        self.single_channel_interp = SingleChannelInterp(input_dim=self.interp_dim)
        self.cross_channel_interp = CrossChannelInterp(input_dim=self.interp_dim)

        if self.use_static:
            self.demo_encoder = nn.Sequential(
                nn.LazyLinear(recurrent_n_units),
                nn.ReLU(),
                nn.Linear(recurrent_n_units, recurrent_n_units)
            )

        self.gru = nn.GRU(
            input_size=self.interp_dim,
            hidden_size=recurrent_n_units,
            dropout=self.recurrent_dropout,
            batch_first=True
        )

        self.input_dropout_layer = nn.Dropout(self.dropout)
        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout)

    def forward(self, x, static, time, sensor_mask, **kwargs):
        """
        Forward pass of IPNets encoder.

        Args:
            x: (N, F, T) - batch, sensors, time
            static: (N, static_count) - static features
            time: (N, T) - time encodings
            sensor_mask: (N, F, T) - mask for missing values

        Returns:
            pooled: (N, recurrent_n_units) - encoded representation
            reconstruction_loss: scalar - reconstruction loss for training
        """

        if self.obs_strategy == "both":
            pass
        elif self.obs_strategy == "indicator_only":
            x = sensor_mask.clone().float()
        elif self.obs_strategy == "obs_only":
            sensor_mask = torch.ones_like(sensor_mask)
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

        times, values, measurements, grid, grid_lengths, static_processed = (
            self.create_timepoint_grid(x, static, time, sensor_mask)
        )

        layer_input = torch.cat((values, measurements.float(), times), dim=1)

        sic_output = self.single_channel_interp(layer_input, grid)
        crc_output = self.cross_channel_interp(sic_output)

        rnn_input = crc_output.transpose(1, 2)

        if self.training:
            dropout_mask = self.input_dropout_layer(torch.ones_like(rnn_input[:, 0, :]))
            rnn_input = rnn_input * dropout_mask.unsqueeze(1)

        if self.use_static:
            demo_encoded = self.demo_encoder(static_processed)
            hidden_state = demo_encoded
        else:
            hidden_state = torch.zeros((x.shape[0], self.n_units), device=x.device)

        rnn_output = self.gru(rnn_input, hidden_state.unsqueeze(0))[0]

        # Since we did not mask during the RNN pass, we take the last non-zero
        # time reading based on pooling strategy
        if self.pool == "hidden":
            idx = (
                (grid_lengths - 1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, 1, self.n_units)
                    .long()
            )
            pooled = rnn_output.gather(1, idx).squeeze(1)
        elif self.pool == "max":
            pooled = masked_max_pooling(rnn_output, None)
        elif self.pool == "mean":
            pooled = masked_mean_pooling(rnn_output, None)
        else:
            raise NotImplementedError(f"Pooling function {self.pool} not supported.")

        # Reconstruction loss calculation
        reconst_mask = (
            torch.rand(measurements.shape, device=measurements.device)
            > self.reconst_fraction
        )
        context_measurements = measurements & reconst_mask

        # Check for missing observations, fill first time step
        nothing_observed = context_measurements.sum(dim=-1, keepdim=True) == 0
        context_measurements = context_measurements | nothing_observed

        # Reconstruction pass; improve imputation
        reconst_input = torch.cat((values, context_measurements.float(), times), dim=1)
        sic_reconst = self.single_channel_interp(
            reconst_input, grid, reconstruction=True
        )
        crc_reconst = self.cross_channel_interp(sic_reconst, reconstruction=True)

        target_measurements = measurements & (~context_measurements)
        squared_error = target_measurements.float() * (values - crc_reconst) ** 2
        instance_wise_reconst_error = squared_error.sum(dim=[1, 2]) / (
            target_measurements.float().sum(dim=[1, 2]) + self.eps
        )

        # Loss calculation based on training phase
        if self.training:
            reconstruction_loss = instance_wise_reconst_error.mean()
        else:
            reconstruction_loss = torch.zeros(
                (), dtype=torch.float32, device=x.device
            )

        return pooled, reconstruction_loss

    def create_timepoint_grid(self, x, static, time, sensor_mask):
        """
        Create interpolation grid for IP-Nets.

        Args:
            x: (N, F, T) - input values
            static: (N, static_count) - static features
            time: (N, T) - time encodings
            sensor_mask: (N, F, T) - observation mask

        Returns:
            Tuple of tensors for interpolation
        """

        all_demo = static
        all_x = []
        all_y = []
        all_measurements = []
        all_grid = []
        all_grid_length = []

        end_time = torch.max(time)

        for batch_ind in range(x.shape[0]):
            # Bit of notation wonkiness; x is actually considered the time with y the value.
            y_ind = x[batch_ind].permute(1, 0)
            x_ind = time[batch_ind]
            sensor_mask_ind = sensor_mask[batch_ind].permute(1, 0).bool()

            length = torch.nonzero(sensor_mask_ind.sum(dim=0)).shape[0]

            X = x_ind.unsqueeze(-1)

            # Check if a value was never measured. If this is the case, add an
            # observation at timepoint t=0 with the mean, assuming mean-centered data (mean = 0).
            n_observed_values = (sensor_mask_ind == False).sum(dim=0)
            nothing_ever_observed = torch.where(n_observed_values == length)[0]

            # Update Y and measurements to add observations at t=0
            indices = torch.stack(
                [torch.zeros_like(nothing_ever_observed), nothing_ever_observed], dim=1
            )
            Y = y_ind.clone()
            measurements = sensor_mask_ind.clone()

            # WHAT IS THIS?
            if indices.shape[0] > 0:
                Y = Y.index_put(
                    (indices[:, 0], indices[:, 1]),
                    torch.zeros(indices.shape[0], device=y_ind.device),
                )
                measurements = measurements.index_put(
                    (indices[:, 0], indices[:, 1]),
                    torch.ones(
                        indices.shape[0], dtype=torch.bool, device=sensor_mask_ind.device
                    ),
                )

            # Generate a grid for imputation
            grid = torch.arange(
                0,
                end_time + self.imputation_stepsize,
                step=self.imputation_stepsize,
                device=x_ind.device,
            )
            grid_length = torch.tensor(
                grid.shape[0], dtype=torch.int32, device=grid.device
            )

            X = X.repeat(1, Y.shape[-1])

            X = X.transpose(0, 1)
            Y = Y.transpose(0, 1)
            measurements = measurements.transpose(0, 1)

            all_y.append(Y)
            all_x.append(X)
            all_measurements.append(measurements)
            all_grid.append(grid)
            all_grid_length.append(grid_length)

        all_grid = torch.stack(all_grid)
        all_y = torch.stack(all_y)
        all_x = torch.stack(all_x)
        all_measurements = torch.stack(all_measurements)
        all_grid_length = torch.stack(all_grid_length)

        return all_x, all_y, all_measurements, all_grid, all_grid_length, all_demo


class AutoregressiveIPNetsEncoder(nn.Module):
    """
    Autoregressive IP-Nets encoder that makes per-timestep predictions.
    Uses causal masking in the interpolation network and does not pool the GRU output.
    """
    def __init__(
        self,
        device="cpu",
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        recurrent_n_units=128,
        ipnets_imputation_stepsize=0.25,
        dropout=0.2,
        recurrent_dropout=0.2,
        ipnets_reconst_fraction=0.5,
        obs_strategy="both",
        use_static=True,
        **kwargs
    ):
        super(AutoregressiveIPNetsEncoder, self).__init__()

        self.obs_strategy = obs_strategy
        self.use_static = use_static
        self.device = device

        self.imputation_stepsize = ipnets_imputation_stepsize
        self.reconst_fraction = ipnets_reconst_fraction
        self.eps = 1e-9
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.n_units = recurrent_n_units
        self.sensors_count = sensors_count
        self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count

        self.interp_dim = sensors_count * 3

        # Use autoregressive interpolation layers
        self.single_channel_interp = AutoregressiveSingleChannelInterp(input_dim=self.interp_dim)
        self.cross_channel_interp = CrossChannelInterp(input_dim=self.interp_dim)

        if self.use_static:
            self.demo_encoder = nn.Sequential(
                nn.LazyLinear(recurrent_n_units),
                nn.ReLU(),
                nn.Linear(recurrent_n_units, recurrent_n_units)
            )

        self.gru = nn.GRU(
            input_size=self.interp_dim,
            hidden_size=recurrent_n_units,
            dropout=self.recurrent_dropout,
            batch_first=True
        )

        self.input_dropout_layer = nn.Dropout(self.dropout)
        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout)

    def forward(self, x, static, time, sensor_mask, **kwargs):
        """
        Forward pass of autoregressive IPNets encoder.

        Args:
            x: (N, F, T) - batch, sensors, time
            static: (N, static_count) - static features
            time: (N, T) - time encodings
            sensor_mask: (N, F, T) - mask for missing values

        Returns:
            rnn_output: (N, T_grid, recurrent_n_units) - per-timestep encoded representations
            reconstruction_loss: scalar - reconstruction loss for training
        """

        if self.obs_strategy == "both":
            pass
        elif self.obs_strategy == "indicator_only":
            x = sensor_mask.clone().float()
        elif self.obs_strategy == "obs_only":
            sensor_mask = torch.ones_like(sensor_mask)
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

        times, values, measurements, grid, grid_lengths, static_processed = (
            self.create_timepoint_grid(x, static, time, sensor_mask)
        )

        layer_input = torch.cat((values, measurements.float(), times), dim=1)

        # Use autoregressive interpolation
        sic_output = self.single_channel_interp(layer_input, grid)
        crc_output = self.cross_channel_interp(sic_output)

        rnn_input = crc_output.transpose(1, 2)

        if self.training:
            dropout_mask = self.input_dropout_layer(torch.ones_like(rnn_input[:, 0, :]))
            rnn_input = rnn_input * dropout_mask.unsqueeze(1)

        if self.use_static:
            demo_encoded = self.demo_encoder(static_processed)
            hidden_state = demo_encoded
        else:
            hidden_state = torch.zeros((x.shape[0], self.n_units), device=x.device)

        # Get all timestep outputs from GRU (no pooling)
        rnn_output = self.gru(rnn_input, hidden_state.unsqueeze(0))[0]
        # rnn_output shape: (N, T_grid, recurrent_n_units)

        # Reconstruction loss calculation (similar to non-autoregressive version)
        reconst_mask = (
            torch.rand(measurements.shape, device=measurements.device)
            > self.reconst_fraction
        )
        context_measurements = measurements & reconst_mask

        # Check for missing observations, fill first time step
        nothing_observed = context_measurements.sum(dim=-1, keepdim=True) == 0
        context_measurements = context_measurements | nothing_observed

        # Reconstruction pass with causal masking
        reconst_input = torch.cat((values, context_measurements.float(), times), dim=1)
        sic_reconst = self.single_channel_interp(
            reconst_input, grid, reconstruction=True
        )
        crc_reconst = self.cross_channel_interp(sic_reconst, reconstruction=True)

        target_measurements = measurements & (~context_measurements)
        squared_error = target_measurements.float() * (values - crc_reconst) ** 2
        instance_wise_reconst_error = squared_error.sum(dim=[1, 2]) / (
            target_measurements.float().sum(dim=[1, 2]) + self.eps
        )

        if self.training:
            reconstruction_loss = instance_wise_reconst_error.mean()
        else:
            reconstruction_loss = torch.zeros(
                (), dtype=torch.float32, device=x.device
            )

        return rnn_output, reconstruction_loss

    def create_timepoint_grid(self, x, static, time, sensor_mask):
        """
        Create interpolation grid for autoregressive IP-Nets.
        Same as the regular IP-Nets grid creation.
        """
        all_demo = static
        all_x = []
        all_y = []
        all_measurements = []
        all_grid = []
        all_grid_length = []

        end_time = torch.max(time)

        for batch_ind in range(x.shape[0]):
            y_ind = x[batch_ind].permute(1, 0)
            x_ind = time[batch_ind]
            sensor_mask_ind = sensor_mask[batch_ind].permute(1, 0).bool()

            length = torch.nonzero(sensor_mask_ind.sum(dim=0)).shape[0]

            X = x_ind.unsqueeze(-1)

            n_observed_values = (sensor_mask_ind == False).sum(dim=0)
            nothing_ever_observed = torch.where(n_observed_values == length)[0]

            indices = torch.stack(
                [torch.zeros_like(nothing_ever_observed), nothing_ever_observed], dim=1
            )
            Y = y_ind.clone()
            measurements = sensor_mask_ind.clone()

            if indices.shape[0] > 0:
                Y = Y.index_put(
                    (indices[:, 0], indices[:, 1]),
                    torch.zeros(indices.shape[0], device=y_ind.device),
                )
                measurements = measurements.index_put(
                    (indices[:, 0], indices[:, 1]),
                    torch.ones(
                        indices.shape[0], dtype=torch.bool, device=sensor_mask_ind.device
                    ),
                )

            grid = torch.arange(
                0,
                end_time + self.imputation_stepsize,
                step=self.imputation_stepsize,
                device=x_ind.device,
            )
            grid_length = torch.tensor(
                grid.shape[0], dtype=torch.int32, device=grid.device
            )

            X = X.repeat(1, Y.shape[-1])

            X = X.transpose(0, 1)
            Y = Y.transpose(0, 1)
            measurements = measurements.transpose(0, 1)

            all_y.append(Y)
            all_x.append(X)
            all_measurements.append(measurements)
            all_grid.append(grid)
            all_grid_length.append(grid_length)

        all_grid = torch.stack(all_grid)
        all_y = torch.stack(all_y)
        all_x = torch.stack(all_x)
        all_measurements = torch.stack(all_measurements)
        all_grid_length = torch.stack(all_grid_length)

        return all_x, all_y, all_measurements, all_grid, all_grid_length, all_demo


class IPNetsEncoderPrediction(nn.Module):
    """
    Combines IPNets encoder with prediction head.
    """
    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}

        # Input dim is the output of the GRU
        self.input_dim = self.encoder_class.n_units

        # Initialize prediction head
        self.head = self.prediction_head(
            input_dim=self.input_dim,
            **self.prediction_head_kwargs
        )

    def forward(self, x, static, time, sensor_mask):
        features, reconstruction_loss = self.encoder_class(x, static, time, sensor_mask)

        # Sanity check shape
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Mismatch between computed input_dim ({self.input_dim}) and actual ({features.shape[1]})"
            )

        predictions = self.head(features)

        # Return both predictions and reconstruction loss for custom loss handling
        return predictions, reconstruction_loss # use this for pre-training
        #return predictions # Use this for finetuning

class AutoregressiveIPNetsEncoderPrediction(nn.Module):
    """
    Combines autoregressive IPNets encoder with per-timestep prediction head.
    """
    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}

        # Input dim is the output of the GRU
        self.input_dim = self.encoder_class.n_units

        # Initialize prediction head
        self.head = self.prediction_head(
            input_dim=self.input_dim,
            **self.prediction_head_kwargs
        )

    def forward(self, x, static, time, sensor_mask):
        # features shape: (N, T, recurrent_n_units)
        features, reconstruction_loss = self.encoder_class(x, static, time, sensor_mask)

        # Sanity check shape
        if features.shape[2] != self.input_dim:
            raise ValueError(
                f"Mismatch between computed input_dim ({self.input_dim}) and actual ({features.shape[2]})"
            )

        # predictions shape: (N, T, num_classes) or (N, T) for regression
        predictions = self.head(features)

        return predictions, reconstruction_loss


@gin.configurable
class IPNetsModel(CustomDLPredictionWrapper):
    """
    Interpolation-Prediction Networks wrapper for YAIB framework.

    IP-Nets uses learned interpolation to handle irregularly sampled time series,
    followed by a GRU for temporal modeling.

    Automatically switches between pooled and autoregressive modes based on task:
    - If TIMESTEP_LEVEL_PREDICTIONS=True (e.g., AKI): Uses autoregressive encoder
    - Otherwise: Uses pooled encoder
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        recurrent_n_units=128,
        ipnets_imputation_stepsize=0.25,
        dropout=0.2,
        recurrent_dropout=0.2,
        ipnets_reconst_fraction=0.5,
        pooling="hidden",
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

        # Check if we should use autoregressive mode (like BAT and GRU-D do)
        try:
            skip_pooling = gin.query_parameter("%TIMESTEP_LEVEL_PREDICTIONS")
        except Exception:
            skip_pooling = False

        # Choose encoder based on task requirements
        if skip_pooling:
            # Per-timestep mode (e.g., AKI) - use regular encoder but without pooling
            # The GRU is already causal, so we don't need causal interpolation
            encoder = IPNetsEncoder(
                device=self.device,
                pooling=None,  # No pooling - return all timesteps
                sensors_count=sensors_count,
                max_timepoint_count=max_timepoint_count,
                static_count=static_count,
                recurrent_n_units=recurrent_n_units,
                ipnets_imputation_stepsize=ipnets_imputation_stepsize,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                ipnets_reconst_fraction=ipnets_reconst_fraction,
            )
            # Use appropriate prediction head for per-timestep predictions
            if prediction_head == BinaryClassificationHead:
                prediction_head = TimeseriesClassificationHead

            self.model = AutoregressiveIPNetsEncoderPrediction(
                encoder_class=encoder,
                prediction_head=prediction_head,
                prediction_head_kwargs=prediction_head_kwargs,
            )
        else:
            # Pooled mode for single prediction per patient (e.g., Mortality24, Sepsis)
            encoder = IPNetsEncoder(
                device=self.device,
                pooling=pooling,
                sensors_count=sensors_count,
                max_timepoint_count=max_timepoint_count,
                static_count=static_count,
                recurrent_n_units=recurrent_n_units,
                ipnets_imputation_stepsize=ipnets_imputation_stepsize,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                ipnets_reconst_fraction=ipnets_reconst_fraction,
            )

            self.model = IPNetsEncoderPrediction(
                encoder_class=encoder,
                prediction_head=prediction_head,
                prediction_head_kwargs=prediction_head_kwargs,
            )

        # Store reconstruction loss weight
        self.reconstruction_loss_weight = kwargs.get("reconstruction_loss_weight", 1.0)

        # Store configuration for debugging
        self.skip_pooling = skip_pooling

    def forward(self, data, static, time, sensor_mask):
        predictions, reconstruction_loss = self.model(
            data, static=static, time=time, sensor_mask=sensor_mask
        )

        # Store reconstruction loss for use in training_step
        self.last_reconstruction_loss = reconstruction_loss

        return predictions

    def training_step(self, batch, batch_idx):
        """
        Custom training step to include reconstruction loss.
        """
        # Call parent's training_step
        loss_dict = super().training_step(batch, batch_idx)

        # Add reconstruction loss if it exists
        if hasattr(self, 'last_reconstruction_loss'):
            task_loss = loss_dict if isinstance(loss_dict, torch.Tensor) else loss_dict.get('loss', torch.tensor(0.0))
            total_loss = task_loss + self.reconstruction_loss_weight * self.last_reconstruction_loss

            # Log losses
            self.log('train_task_loss', task_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_reconst_loss', self.last_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)

            return total_loss

        return loss_dict


@gin.configurable
class AutoregressiveIPNetsModel(CustomDLPredictionWrapper):
    """
    Autoregressive Interpolation-Prediction Networks wrapper for YAIB framework.

    Makes per-timestep predictions using only past information through:
    1. Causal masking in the interpolation network
    2. Per-timestep GRU outputs (no pooling)
    3. Per-timestep prediction heads (TimeseriesClassificationHead or RegressionHead)
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        recurrent_n_units=128,
        ipnets_imputation_stepsize=0.25,
        dropout=0.2,
        recurrent_dropout=0.2,
        ipnets_reconst_fraction=0.5,
        prediction_head=None,
        prediction_head_kwargs=None,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)

        # Import here to avoid circular imports
        from icu_benchmarks.models.dl_models.bat import TimeseriesClassificationHead, RegressionHead

        # Set default prediction head based on run mode if not provided
        if prediction_head is None:
            if hasattr(self, 'run_mode'):
                if self.run_mode == RunMode.classification:
                    prediction_head = TimeseriesClassificationHead
                    if prediction_head_kwargs is None:
                        prediction_head_kwargs = {"num_classes": 2}
                else:  # regression
                    prediction_head = RegressionHead
                    if prediction_head_kwargs is None:
                        prediction_head_kwargs = {"output_dim": 1}
            else:
                # Default to classification
                prediction_head = TimeseriesClassificationHead
                if prediction_head_kwargs is None:
                    prediction_head_kwargs = {"num_classes": 2}

        if prediction_head_kwargs is None:
            prediction_head_kwargs = {"num_classes": 2}

        self.save_hyperparameters()

        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)

        encoder = AutoregressiveIPNetsEncoder(
            device=self.device,
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
            recurrent_n_units=recurrent_n_units,
            ipnets_imputation_stepsize=ipnets_imputation_stepsize,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            ipnets_reconst_fraction=ipnets_reconst_fraction,
        )

        self.model = AutoregressiveIPNetsEncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

        # Store reconstruction loss weight
        self.reconstruction_loss_weight = kwargs.get("reconstruction_loss_weight", 1.0)

    def forward(self, data, static, time, sensor_mask):
        predictions, reconstruction_loss = self.model(
            data, static=static, time=time, sensor_mask=sensor_mask
        )

        # Store reconstruction loss for use in training_step
        self.last_reconstruction_loss = reconstruction_loss

        return predictions

    def training_step(self, batch, batch_idx):
        """
        Custom training step to include reconstruction loss.
        """
        # Call parent's training_step
        loss_dict = super().training_step(batch, batch_idx)

        # Add reconstruction loss if it exists
        if hasattr(self, 'last_reconstruction_loss'):
            task_loss = loss_dict if isinstance(loss_dict, torch.Tensor) else loss_dict.get('loss', torch.tensor(0.0))
            total_loss = task_loss + self.reconstruction_loss_weight * self.last_reconstruction_loss

            # Log losses
            self.log('train_task_loss', task_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_reconst_loss', self.last_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)

            return total_loss

        return loss_dict


@gin.configurable
class SSL_IPNets(SSLWrapper):
    """
    Self-Supervised Learning wrapper for IP-Nets model.

    This enables the Interpolation-Prediction Networks model to be used for SSL
    pretraining with forecasting tasks, similar to SSL_BAT, SSL_iTransformer,
    and SSL_DeepSetAttention.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        recurrent_n_units=128,
        ipnets_imputation_stepsize=0.25,
        dropout=0.2,
        recurrent_dropout=0.2,
        ipnets_reconst_fraction=0.5,
        pooling="hidden",
        prediction_head=ForecastingHead,
        prediction_head_kwargs=None,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)

        if prediction_head_kwargs is None:
            prediction_head_kwargs = {"sensors_count": 48, "forecast_len": 2}

        self.save_hyperparameters()

        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)

        encoder = IPNetsEncoder(
            device=self.device,
            pooling=pooling,
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
            recurrent_n_units=recurrent_n_units,
            ipnets_imputation_stepsize=ipnets_imputation_stepsize,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            ipnets_reconst_fraction=ipnets_reconst_fraction,
        )

        self.model = IPNetsEncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

        # Store reconstruction loss weight
        self.reconstruction_loss_weight = kwargs.get("reconstruction_loss_weight", 1.0)

    def forward(self, data, static, time, sensor_mask):
        predictions, reconstruction_loss = self.model(
            data, static=static, time=time, sensor_mask=sensor_mask
        )

        # Store reconstruction loss for use in training_step
        self.last_reconstruction_loss = reconstruction_loss

        return predictions

    def training_step(self, batch, batch_idx):
        """
        Custom training step to include reconstruction loss.
        """
        # Call parent's training_step
        loss_dict = super().training_step(batch, batch_idx)

        # Add reconstruction loss if it exists
        if hasattr(self, 'last_reconstruction_loss'):
            task_loss = loss_dict if isinstance(loss_dict, torch.Tensor) else loss_dict.get('loss', torch.tensor(0.0))
            total_loss = task_loss + self.reconstruction_loss_weight * self.last_reconstruction_loss

            # Log losses
            self.log('train_task_loss', task_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_reconst_loss', self.last_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)

            return total_loss

        return loss_dict
