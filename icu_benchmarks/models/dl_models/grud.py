import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from typing import List
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import CustomDLPredictionWrapper, SSLWrapper, ImputationWrapper
from icu_benchmarks.models.dl_models.bat import ForecastingHead


def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()
    data_summed = torch.sum(datatensor * mask_expanded, dim=1)

    data_counts = mask_expanded.sum(1)
    data_counts = torch.clamp(data_counts, min=1e-9)

    averaged = data_summed / (data_counts)

    return averaged


def masked_max_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    """
    mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float()

    datatensor[mask_expanded == 0] = -1e9
    maxed = torch.max(datatensor, 1)[0]

    return maxed


def exp_relu(x):
    return torch.exp(-F.relu(x))


def get_activation(identifier):
    if identifier is None:
        return None
    if identifier == "exp_relu":
        return exp_relu
    return getattr(F, identifier.lower(), None)


@jit.script
def generate_masks(inputs: torch.Tensor, dropout_prob: float, training: bool, num_units: int, count: int):
    masks = torch.jit.annotate(List[torch.Tensor], [])

    for _ in range(count):
        if not training or dropout_prob == 0:
            mask = torch.ones_like(inputs[:, :num_units], dtype=torch.float32)
        else:
            mask = F.dropout(
                torch.ones_like(inputs[:, :num_units], dtype=torch.float32),
                p=dropout_prob,
                training=training,
            )
        masks.append(mask)

    return torch.stack(masks)


class GRUDCell(jit.ScriptModule):
    def get_dropout_mask_for_cell(
        self, inputs, dropout_prob: float, training: bool, num_units: int, count: int
    ):
        if not self.input_dropout_masks[0].numel() == 1:
            return self.input_dropout_masks
        else:
            return generate_masks(inputs, dropout_prob, training, num_units, count)

    def get_rdropout_mask_for_cell(
        self, inputs, dropout_prob: float, training: bool, num_units: int, count: int
    ):
        if not self.recurrent_dropout_masks[0].numel() == 1:
            return self.recurrent_dropout_masks
        else:
            return generate_masks(inputs, dropout_prob, training, num_units, count)

    def get_mdropout_mask_for_cell(
        self, inputs, dropout_prob: float, training: bool, num_units: int, count: int
    ):
        if not self.feed_dropout_masks[0].numel() == 1:
            return self.feed_dropout_masks
        else:
            return generate_masks(inputs, dropout_prob, training, num_units, count)

    def reset_masks(self):
        self.recurrent_dropout_masks = torch.tensor([1, 1, 1])
        self.input_dropout_masks = torch.tensor([1, 1, 1])
        self.feed_dropout_masks = torch.tensor([1, 1, 1])

    def __init__(
        self,
        input_size,
        hidden_size,
        device,
        x_imputation="zero",
        input_decay="exp_relu",
        hidden_decay="exp_relu",
        activation="tanh",
        recurrent_activation="hardsigmoid",
        use_decay_bias=True,
        feed_masking=True,
        masking_decay=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        use_bias=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.x_imputation = x_imputation
        self.input_decay = get_activation(input_decay)
        self.hidden_decay = get_activation(hidden_decay)
        self.activation = get_activation(activation)
        self.recurrent_activation = get_activation(recurrent_activation)
        self.use_decay_bias = use_decay_bias
        self.feed_masking = feed_masking
        self.masking_decay = get_activation(masking_decay)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.use_input_decay: bool = bool(input_decay)
        self.use_hidden_decay: bool = bool(hidden_decay)
        self.use_masking_decay: bool = bool(masking_decay)

        self.input_dropout_masks = torch.tensor([1, 1, 1])
        self.recurrent_dropout_masks = torch.tensor([1, 1, 1])
        self.feed_dropout_masks = torch.tensor([1, 1, 1])

        self.kernel_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.kernel_r = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.kernel_h = nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.recurrent_kernel_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.recurrent_kernel_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.recurrent_kernel_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))

        self.use_bias = use_bias

        if self.use_input_decay:
            self.input_decay_kernel = nn.Parameter(torch.Tensor(input_size))
            if self.use_decay_bias:
                self.input_decay_bias = nn.Parameter(torch.Tensor(input_size))

        if self.use_hidden_decay:
            self.hidden_decay_kernel = nn.Parameter(
                torch.Tensor(input_size, hidden_size)
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = nn.Parameter(torch.Tensor(hidden_size))

        if self.feed_masking:
            self.masking_kernel_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.masking_kernel_r = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.masking_kernel_h = nn.Parameter(torch.Tensor(hidden_size, input_size))

            if self.use_masking_decay:
                self.masking_decay_kernel = nn.Parameter(torch.Tensor(input_size))
                if self.use_decay_bias:
                    self.masking_decay_bias = nn.Parameter(torch.Tensor(input_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel_z)
        nn.init.xavier_uniform_(self.kernel_h)
        nn.init.xavier_uniform_(self.kernel_r)
        nn.init.xavier_uniform_(self.recurrent_kernel_z)
        nn.init.xavier_uniform_(self.recurrent_kernel_h)
        nn.init.xavier_uniform_(self.recurrent_kernel_r)

        if self.use_input_decay:
            nn.init.zeros_(self.input_decay_kernel)
            if self.use_decay_bias:
                nn.init.zeros_(self.input_decay_bias)

        if self.use_hidden_decay:
            nn.init.zeros_(self.hidden_decay_kernel)
            if self.use_decay_bias:
                nn.init.zeros_(self.hidden_decay_bias)

        if self.feed_masking:
            nn.init.xavier_uniform_(self.masking_kernel_z)
            nn.init.xavier_uniform_(self.masking_kernel_h)
            nn.init.xavier_uniform_(self.masking_kernel_r)
            if self.masking_decay:
                nn.init.zeros_(self.masking_decay_kernel)
                if self.use_decay_bias:
                    nn.init.zeros_(self.masking_decay_bias)

    @jit.script_method
    def forward(self, input_x, input_m, input_s, h_tm1, x_keep_tm1, s_prev_tm1):

        input_1m = 1.0 - input_m.float()
        input_d = input_s - s_prev_tm1

        self.input_dropout_masks = self.get_dropout_mask_for_cell(
            input_x,
            dropout_prob=self.dropout,
            training=self.training,
            num_units=self.input_size,
            count=3,
        )
        self.recurrent_dropout_masks = self.get_rdropout_mask_for_cell(
            h_tm1,
            dropout_prob=self.recurrent_dropout,
            training=self.training,
            num_units=self.hidden_size,
            count=3,
        )
        self.feed_dropout_masks = self.get_mdropout_mask_for_cell(
            input_m.double(),
            dropout_prob=self.dropout,
            training=self.training,
            num_units=self.input_size,
            count=3,
        )

        gamma_di = torch.tensor(1)
        gamma_dh = torch.tensor(1)
        gamma_dm = torch.tensor(1)

        if self.use_input_decay:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = gamma_di + self.input_decay_bias
            gamma_di = self.input_decay(gamma_di)

        if self.use_hidden_decay:
            gamma_dh = torch.matmul(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = gamma_dh + self.hidden_decay_bias
            gamma_dh = self.hidden_decay(gamma_dh)

        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = gamma_dm + self.masking_decay_bias
            gamma_dm = self.masking_decay(gamma_dm)

        if self.use_input_decay:
            x_keep_t = torch.where(input_m, input_x, x_keep_tm1)
            x_t = torch.where(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == "forward":
            x_t = torch.where(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == "zero":
            x_t = torch.where(input_m, input_x, torch.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == "raw":
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError(f"Invalid x_imputation: {self.x_imputation}")

        if self.use_hidden_decay:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        m_t = torch.tensor(1)
        m_z = torch.tensor(1)
        m_h = torch.tensor(1)
        m_r = torch.tensor(1)

        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        if self.training:
            x_z, x_r, x_h = (
                x_t * self.input_dropout_masks[0],
                x_t * self.input_dropout_masks[1],
                x_t * self.input_dropout_masks[2],
            )
            h_tm1_z, h_tm1_r = (
                h_tm1d * self.recurrent_dropout_masks[0],
                h_tm1d * self.recurrent_dropout_masks[1],
            )
            if self.feed_masking:
                m_z, m_r, m_h = (
                    m_t * self.feed_dropout_masks[0],
                    m_t * self.feed_dropout_masks[1],
                    m_t * self.feed_dropout_masks[2],
                )
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t

        z_t = F.linear(x_z, self.kernel_z) + F.linear(h_tm1_z, self.recurrent_kernel_z)
        r_t = F.linear(x_r, self.kernel_r) + F.linear(h_tm1_r, self.recurrent_kernel_r)
        hh_t = F.linear(x_h, self.kernel_h)
        if self.feed_masking:
            z_t += F.linear(m_z, self.masking_kernel_z)
            r_t += F.linear(m_r, self.masking_kernel_r)
            hh_t += F.linear(m_h, self.masking_kernel_h)
        if self.use_bias:
            z_t = z_t + self.bias_z
            r_t = r_t + self.bias_r
            hh_t = hh_t + self.bias_h
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)

        h_hm1_t = r_t * h_tm1d * self.recurrent_dropout_masks[2]
        hh_t = self.activation(hh_t + F.linear(h_hm1_t, self.recurrent_kernel_h))

        h_t = z_t * h_tm1 + (1 - z_t) * hh_t

        s_prev_t = torch.where(input_m, input_s.expand(-1, self.input_size), s_prev_tm1)

        return h_t, x_keep_t, s_prev_t


class GRUD(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, device, **kwargs):
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size, device, **kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

    @jit.script_method
    def forward(self, values, mask, time, h_t=None, x_keep_t=None, s_prev_t=None):
        batch_size, seq_len, _ = values.size()

        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=values.device)
            x_keep_t = torch.zeros(batch_size, self.input_size, device=values.device)
            s_prev_t = time[:, 0].expand(-1, self.input_size)

        outputs = []

        for t in range(seq_len):
            # Don't manually move to device - inputs are already on correct device
            x_t = values[:, t]
            m_t = mask[:, t]
            s_t = time[:, t]

            h_t, x_keep_t, s_prev_t = self.cell(
                x_t, m_t, s_t, h_t, x_keep_t, s_prev_t
            )
            outputs.append(h_t)
        self.cell.reset_masks()

        return torch.stack(outputs, dim=1)


# Prediction head classes
@gin.configurable
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


@gin.configurable
class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# Encoder class
@gin.configurable
class GRUDEncoder(nn.Module):
    """
    GRUD-based encoder based on SEFT tensorflow implementation at:
    https://github.com/BorgwardtLab/Set_Functions_for_Time_Series/blob/master/seft/models/gru_d.py

    """

    def __init__(
        self,
        device="cpu",
        pooling="hidden",
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        recurrent_n_units=128,
        dropout=0.2,
        recurrent_dropout=0.2,
        use_static=True,
        obs_strategy="both",
        x_imputation="zero",
        input_decay="exp_relu",
        hidden_decay="exp_relu",
        activation="tanh",
        recurrent_activation="hardsigmoid",
        use_decay_bias=True,
        feed_masking=True,
        masking_decay=None,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.sensors_count = sensors_count
        self.max_timepoint_count = max_timepoint_count
        self.static_count = static_count
        self.use_static = use_static
        self.obs_strategy = obs_strategy
        self.pooling = pooling
        self.recurrent_n_units = recurrent_n_units

        self.input_dim = sensors_count

        self.rnn = GRUD(
            self.input_dim,
            recurrent_n_units,
            device=self.device,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(static_count, recurrent_n_units),
            nn.ReLU(),
            nn.Linear(recurrent_n_units, recurrent_n_units),
        ).to(self.device)

        self.output_dim = recurrent_n_units

    def forward(self, x, static, time, sensor_mask, **kwargs):
        values = torch.permute(x, (0, 2, 1))
        sensor_mask_permuted = torch.permute(sensor_mask, (0, 2, 1)).bool()

        if self.obs_strategy == "both":
            pass
        elif self.obs_strategy == "indicator_only":
            values = sensor_mask_permuted.clone()
        elif self.obs_strategy == "obs_only":
            sensor_mask_permuted = torch.ones_like(sensor_mask_permuted)
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

        if self.use_static:
            static_encoded = self.static_encoder(static)
        else:
            # Use the device of the static_encoder instead of self.device
            encoder_device = next(self.static_encoder.parameters()).device
            static_encoded = torch.zeros((static.shape[0], self.recurrent_n_units), device=encoder_device)

        time_expanded = time.unsqueeze(-1)

        h_t = static_encoded
        x_keep_t = torch.zeros(static.size(0), self.input_dim, device=static.device)
        s_prev_t = time_expanded[:, 0].repeat(1, self.input_dim).to(static.device)

        time_mask = (torch.count_nonzero(time_expanded, dim=2)) > 0
        time_mask[:, 0] = True

        grud_output = self.rnn(values, sensor_mask_permuted, time_expanded, h_t, x_keep_t, s_prev_t)

        if self.pooling == "hidden":
            last_valid_indices = time_mask.sum(dim=1).long() - 1
            pooled = grud_output[torch.arange(grud_output.shape[0]), last_valid_indices]
        elif self.pooling == "max":
            pooled = masked_max_pooling(grud_output, time_mask)
        elif self.pooling == "mean":
            pooled = masked_mean_pooling(grud_output, time_mask)
        else:
            raise NotImplementedError(f"Pooling function {self.pooling} not supported.")

        return pooled


# EncoderPrediction wrapper
@gin.configurable
class GRUDEncoderPrediction(nn.Module):
    """Combines GRU-D encoder with a prediction head."""

    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}

        self.input_dim = self.encoder_class.output_dim

        self.head = self.prediction_head(
            input_dim=self.input_dim,
            **self.prediction_head_kwargs
        )

    def forward(self, x, static, time, sensor_mask):
        features = self.encoder_class(x, static, time, sensor_mask)

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Mismatch between computed input_dim ({self.input_dim}) and actual ({features.shape[1]})"
            )

        return self.head(features)


# Main wrapper class for YAIB integration
@gin.configurable
class GRUDModel(CustomDLPredictionWrapper):
    """
    GRU-D wrapper for YAIB integration.
    Handles recurrent neural networks with decay mechanisms for irregular time series.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        recurrent_n_units=128,
        dropout=0.2,
        recurrent_dropout=0.2,
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

        encoder = GRUDEncoder(
            device=self.device,
            pooling=pooling,
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
            recurrent_n_units=recurrent_n_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

        self.model = GRUDEncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

        self.logit = nn.Linear(1, prediction_head_kwargs.get("num_classes", 2))

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)


@gin.configurable
class SSL_GRUD(SSLWrapper):
    """
    Self-supervised GRU-D for next token prediction/forecasting.
    Uses the SSLWrapper for forecasting-based pretraining.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        recurrent_n_units=128,
        dropout=0.2,
        recurrent_dropout=0.2,
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

        # Extract dimensions from dataset
        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)

        # Instantiate GRUD encoder
        encoder = GRUDEncoder(
            device=self.device,
            pooling=pooling,
            sensors_count=sensors_count,
            max_timepoint_count=max_timepoint_count,
            static_count=static_count,
            recurrent_n_units=recurrent_n_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

        # Compose full prediction model with forecasting head
        self.model = GRUDEncoderPrediction(
            encoder_class=encoder,
            prediction_head=prediction_head,
            prediction_head_kwargs=prediction_head_kwargs,
        )

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)


@gin.configurable("GRUD")
class GRUDImputation(ImputationWrapper):
    """
    Imputation model using GRU-D (Gated Recurrent Unit with Decay).

    GRU-D naturally handles missing data through:
    - Decay mechanisms for both input and hidden states
    - Masking indicators that inform the model about missingness patterns
    - Imputation strategies (forward fill, zero fill, or decay-based)
    """

    requires_backprop = True

    def __init__(
        self,
        *args,
        input_size,
        hidden_size=64,
        dropout=0.0,
        recurrent_dropout=0.0,
        x_imputation="zero",
        input_decay="exp_relu",
        hidden_decay="exp_relu",
        activation="tanh",
        recurrent_activation="hardsigmoid",
        use_decay_bias=True,
        feed_masking=True,
        masking_decay=None,
        **kwargs
    ):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
            **kwargs
        )

        self.input_size = input_size
        self.n_features = input_size[2]  # (batch, time, features)
        self.hidden_size = hidden_size

        # GRUD core
        self.rnn = GRUD(
            input_size=self.n_features,
            hidden_size=hidden_size,
            device=self.device,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
        )

        # Output layer to predict imputed values
        self.imputation_layer = nn.Linear(hidden_size, self.n_features)

    def forward(self, amputated, amputation_mask):
        """
        Forward pass for imputation.

        Args:
            amputated: Tensor of shape (batch, time, features) with missing values
            amputation_mask: Boolean tensor of shape (batch, time, features)
                           where True indicates missing values

        Returns:
            Tensor of shape (batch, time, features) with imputed values
        """
        batch_size, seq_len, n_features = amputated.shape

        # Create observation mask (inverse of amputation mask)
        # GRUD expects mask where True = observed, False = missing
        observation_mask = ~amputation_mask.bool()

        # Create time tensor (assuming uniform time steps)
        # For real data, this should come from actual timestamps
        time = torch.arange(seq_len, device=amputated.device, dtype=amputated.dtype)
        time = time.unsqueeze(0).expand(batch_size, -1)

        # Run GRUD to get hidden states
        # GRUD.forward expects: (values, mask, time)
        hidden_states = self.rnn(
            values=amputated,
            mask=observation_mask,
            time=time
        )  # Returns (batch, time, hidden_size)

        # Project hidden states to imputed values
        imputed = self.imputation_layer(hidden_states)

        # Only replace missing values, keep observed values
        output = torch.where(amputation_mask, imputed, amputated)

        return output