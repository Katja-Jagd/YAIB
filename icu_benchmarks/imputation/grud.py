from icu_benchmarks.models.wrappers import ImputationWrapper
from icu_benchmarks.models.dl_models.grud import GRUD
import torch
import torch.nn as nn
import gin


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
