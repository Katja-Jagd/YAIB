import gin
from torch import nn as nn
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import CustomDLPredictionWrapper, SSLWrapper

# From BAT
import torch
import numpy as np
from torch import nn
from x_transformers import Encoder


# Prediction head classes 
@gin.configurable
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    
@gin.configurable
class ForecastingHead(nn.Module):
    def __init__(self, input_dim, forecast_len, sensors_count):
        super().__init__()
        self.linear = nn.Linear(input_dim, sensors_count * forecast_len)
        self.sensors_count = sensors_count
        self.forecast_len = forecast_len

    def forward(self, x):
        out = self.linear(x)
        return out.reshape(-1, self.sensors_count, self.forecast_len)

# Own encoder prediction class that plugs R's model to different prediction heads 
@gin.configurable
class EncoderPrediction(nn.Module):
    def __init__(self, encoder_class, prediction_head, prediction_head_kwargs=None):
        super().__init__()
        self.encoder_class = encoder_class
        self.prediction_head = prediction_head
        self.prediction_head_kwargs = prediction_head_kwargs or {}
        self.head = None  # Instantiated lazily

    def forward(self, x, static, time, sensor_mask):
        features = self.encoder_class(x, static, time, sensor_mask)

        # Lazily instantiate the head based on input dim 
        if self.head is None:
            self.head = self.prediction_head(
                input_dim=features.shape[1], 
                **self.prediction_head_kwargs
                ).to(features.device)

        return self.head(features)


### From R's repo 
def masked_mean_pooling(datatensor, mask):
    """
    Adapted from HuggingFace's Sentence Transformers:
    https://github.com/UKPLab/sentence-transformers/
    Calculate masked average for final dimension of tensor
    TO DO: expand to multiple dimensions
    """
    if mask == None:
        averaged = torch.mean(datatensor, dim=2)
        averaged = torch.mean(averaged, dim=1)

    else:
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
    Calculate masked average for final dimension of tensor
    """

    if mask == None:
        maxed_1 = torch.max(datatensor, dim=2, keepdim=False)[0]
        maxed_2 = torch.max(maxed_1, dim=1, keepdim=False)[0]  # (N, 18)

    else:
        # eliminate all values learned from missing timepoints
        mask_expanded = mask.unsqueeze(-1).expand(datatensor.size()).float() # (N, F, T) --> (N, F, T, 18)

        datatensor[mask_expanded == 0] = (
            -1e9
        )  # Set padding tokens to large negative value
        maxed_1 = torch.max(datatensor, 2)[0]
        maxed_2 = torch.max(maxed_1, 1)[0]

    return maxed_2


class PositionalEncodingTF(nn.Module):
    """
    Based on the SEFT positional encoding implementation
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        # create a timescale of all times from 0-1
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        # make a tensor to hold the time embeddings
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        # scale the timepoints according to the 0-1 scale
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        # pe = pe.cuda()
        return pe

@gin.configurable
class EncoderClassifierCrossParallel(nn.Module):

    def __init__(
        self,
        device="cpu",
        pooling="mean",
        num_classes=2,
        sensors_count=37,
        max_timepoint_count=215,
        static_count=8,
        value_embed_size=8,
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

        if self.obs_strategy in ("indicator_only", "obs_only"):  # BINARY
            # print("binary_values_only")
            self.sensor_axis_dim = self.sensors_count  # BINARY
            self.time_axis_dim = self.max_timepoint_count  # BINARY
        else:  # BINARY
            self.sensor_axis_dim = 2 * self.sensors_count  # BINARY
            self.time_axis_dim = min(2 * self.max_timepoint_count, 100)  # BINARY

        self.static_out = self.static_count + 4
        self.embed_out = value_embed_size
        self.sensor_encoding_out = min(600, round(1.6 * sensors_count**0.50))
        if self.sensor_encoding_out % 2 != 0:
            self.sensor_encoding_out += 1
        
        self.attn_layers_1 = Encoder(
            dim=self.sensor_encoding_out + self.embed_out,  # new
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
            attn_flash=True,

        )

        self.attn_layers_2 = Encoder(
            dim=self.sensor_encoding_out + self.embed_out,
            depth=layers,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=dropout,
            attn_flash=True,
        )

        self.sensor_encoding = torch.nn.Embedding(
            sensors_count, self.sensor_encoding_out
        )
        if self.obs_strategy in ("indicator_only", "obs_only"):  # BINARY
            self.time_embedding = nn.Linear(1, self.embed_out)  # 2 because of mask only
        else:
            self.time_embedding = nn.Linear(
                2, self.embed_out
            )  # 2 because of sensor reading + mask

        self.use_static = use_static

        self.nonlinear_merger_1 = nn.Linear(
            2 * (self.sensor_encoding_out + self.embed_out),
            2 * (self.sensor_encoding_out + self.embed_out),
        )
        
        if self.use_static:
            print("use static")
            self.static_embedding = nn.Linear(self.static_count, self.static_out)

            self.nonlinear_merger_2 = nn.Linear(
                2 * (self.sensor_encoding_out + self.embed_out) + self.static_out,
                2 * (self.sensor_encoding_out + self.embed_out) + self.static_out,
            )

        self.pos_encoder = PositionalEncodingTF(
            self.sensor_encoding_out + self.embed_out
        )  

        # create layers necessary for pooling fxn
        if self.pooling == "conv":
             # taken from metnet
            self.sensor_conv_first = nn.Conv1d(
                self.max_timepoint_count, 1, kernel_size=(1,)
            )  # taken from metnet
            self.time_conv_second = nn.Conv1d(
                self.sensors_count, 1, kernel_size=(1,)
            )
        if self.pooling == "linear_unroll":
            self.sensor_linear_first = nn.Linear(
                self.sensors_count * self.time_axis_dim, self.time_axis_dim
            )
            self.time_linear_second = nn.Linear(
                self.sensor_axis_dim * self.max_timepoint_count, self.sensor_axis_dim
            )
        if self.pooling == "parameterised_mean":
            self.linear_layer_sensor = nn.Linear(self.time_axis_dim, 1)
            self.linear_layer_time = nn.Linear(self.sensor_axis_dim, 1)

    def forward(self, x, static, time, sensor_mask, **kwargs):

        # QUESTION: should my embedding output size be min(600, round(1.6 * dim ** 0.50))?
        # QUESTION: why does LR keep going down despite val loss doing well?  

        x_time = torch.clone(x).unsqueeze(3)  # (N, F, T, E) 

        """
        x = x.permute((0, 2, 1)) (N, T, F)
        sensor_mask = sensor_mask.permute((0, 2, 1))
        mask = sensor_mask.sum(-1)
        valid_time_points = mask.any(dim=0)
        x = x[:, valid_timepoints, :].permute((0, 2, 1))
        time = time[:, valid_timepoints]
        sensor_mask = sensor_mask[:, valid_time_points, :].permute((0,2,1)) 

        x_time = torch.clone(x).unsqueeze(3)  # (N, F, T, E) 

        print(f"Reduced batch size by {torch.count_nonzero(valid_time_points)} timepoints")
        """

        # add indication for missing values to data
        x_time_mask = torch.clone(sensor_mask).unsqueeze(3)  # (N, F, T, E)
        if self.obs_strategy == "indicator_only":  # Binary
            x_time = x_time_mask.float()
        elif self.obs_strategy == "obs_only":
            x_time = x_time
        elif self.obs_strategy == "both":
            x_time = torch.cat([x_time, x_time_mask], axis=3)  # (N, F, T, 2E)
        else:
            raise NotImplementedError(f"Obs strategy {self.obs_strategy} not found.")

        # make embeddings
        x_time = self.time_embedding(x_time)  # (N, F, T, 8E)

        # Free memory from unused tensors
        del x_time_mask

        # make "encodings" for each sensor
        s_list = torch.arange(0, self.sensors_count)
        sensor_encoding = x.clone()  # make new version for embedding
        sensor_encoding = torch.permute(sensor_encoding, (0, 2, 1))  # (N, T, F)
        sensor_encoding[:, :, :] = (
            s_list  # for every batch/timepoint, we have spots for embedding sensors
        )

        sensor_encoding = self.sensor_encoding(
            sensor_encoding.long()
        )  # create sensor embedding/encoding/something (N, T, F, 10)
        sensor_encoding = torch.permute(sensor_encoding, (0, 2, 1, 3))  # (N, F, T, 10)
        x_time = torch.cat(
            (x_time, sensor_encoding), dim=3
        )  # (N, F, T, 18E) where 1 holds the sensor value

        # Free memory from unused tensor
        del sensor_encoding

        with torch.no_grad():
            # add positional encodings
            #pe = self.pos_encoder(time).to(self.device)  # taken from RAINDROP (N, T, pe) # cpu
            pe = self.pos_encoder(time).to(x_time.device)  # x_time is on cuda due to pytorch ligtning 
            pe = pe.unsqueeze(2)  # CHECK IF THIS IS RIGHT
            # Repeat the 2nd dimension 36 times
            pe = pe.repeat(1, 1, self.sensors_count, 1)
            pe = torch.permute(pe, (0, 2, 1, 3))  # (N, F, T, 18) 

        x_time = torch.add(x_time, pe)  # (N, F, T, 18E)
        del pe

        # get versions/shapes ready for attention
        n, f, t, e = x_time.shape
        # make copy for other attention
        x_sensor = x_time.clone()
        x_sensor = torch.permute(x_sensor, (0, 2, 1, 3))  # (N, T, F, 18)
        x_time = x_time.view(n * f, t, e)  # (N*F, T, 18)
        x_sensor = x_sensor.reshape(n * t, f, e)  # (N*T, F, 18)

        # make mask of all empty (missing) timepoints
        timepoint_mask = torch.clone(x)  # (N, F, T)
        timepoint_mask = torch.permute(timepoint_mask, (0, 2, 1))  # (N, T, F)
        mask = (
            torch.count_nonzero(timepoint_mask, dim=2)
        ) > 0  # mask for sum of all sensors for each person/at each timepoint (N, T)
        mask = mask.repeat(self.sensors_count, 1)  # (N*F, T)
        if not self.use_mask:
            mask_attention = None
        else:
            mask_attention = mask
        
        del x # Free memory from unused tensor
        del timepoint_mask

        # run sensor attention
        if self.return_intermediates:
            x_sensor, sensor_intermediates = self.attn_layers_1(
                x_sensor, return_hiddens=True
            )
        else:
            x_sensor = self.attn_layers_1(x_sensor)
        x_sensor = x_sensor.reshape(n, t, f, e)  # (N, T, F, 18)

        # run time attention
        if self.return_intermediates:
            x_time, time_intermediates = self.attn_layers_2(
                x_time, mask=mask_attention, return_hiddens=True
            )  # attention on time
        else:
            x_time = self.attn_layers_2(x_time, mask=mask_attention)
        x_time = x_time.reshape(n, f, t, e)  # (N, F, T, 18)
        mask = mask.reshape(n, f, t)  # (N, F, T)

        cross = True

        if cross:
            # cross and perform attention again
            x_time = torch.permute(x_time, (0, 2, 1, 3))  # (N, T, F, 18)
            x_time = x_time.reshape(n * t, f, e)  # (N*T, F, 18) flatten
            if self.return_intermediates:
                x_time, time_intermediates = self.attn_layers_1(
                    x_time, return_hiddens=True
                )
            else:
                x_time = self.attn_layers_1(x_time)
            x_time = x_time.view(n, t, f, e)  # (N, T, F, 18)
            x_time = torch.permute(x_time, (0, 2, 1, 3))  # (N, F, T, 18) is this needed?

            x_sensor = torch.permute(x_sensor, (0, 2, 1, 3))  # (N, F, T, 18)
            x_sensor = x_sensor.reshape(n * f, t, e)  # (N*F, T, 18) flatten
            # mask = mask.reshape(n*f, t) # (N*F, T)
            if self.return_intermediates:
                x_sensor, sensor_intermediates = self.attn_layers_2(
                    x_sensor, mask=mask_attention, return_hiddens=True
                )  # attention on time
            else:
                x_sensor = self.attn_layers_2(x_sensor, mask=mask_attention)
            # mask = mask.reshape(n, f, t) # (N, F, T)
            x_sensor = x_sensor.view(n, f, t, e)  # (N, F, T, 18)
            #x_sensor = torch.permute(x_sensor, (0, 2, 1, 3))  # (N, T, F, 18) is this needed?

        del mask_attention

        # Pool
        if not self.use_mask:
            mask_pooling = None
        else:
            mask_pooling = mask

        if self.pooling == "mean":
            #x_sensor = torch.mean(x_sensor, dim=2)
            #x_sensor = torch.mean(x_sensor, dim=1)
            x_sensor = masked_mean_pooling(x_sensor, mask_pooling)
            x_time = masked_mean_pooling(x_time, mask_pooling)
        elif self.pooling == "median":
            x_sensor = torch.median(x_sensor, dim=1)[0]
            x_time = torch.median(x_time, dim=1)[0]
        elif self.pooling == "sum":
            x_sensor = torch.sum(x_sensor, dim=1)  # sum on sensors
            x_time = torch.sum(x_time, dim=1)  # sum on time
        elif self.pooling == "max":
            #x_sensor = torch.max(x_sensor, dim=2, keepdim=False)[0]
            #x_sensor = torch.max(x_sensor, dim=1, keepdim=False)[0]  # (N, 18)
            x_sensor = masked_max_pooling(x_sensor, mask_pooling)
            x_time = masked_max_pooling(x_time, mask_pooling)
        elif self.pooling == "linear_unroll":
            x_sensor = x_sensor.reshape(x_sensor.shape[0], -1)
            x_sensor = self.sensor_linear_first(x_sensor)
            x_time = x_time.reshape(x_time.shape[0], -1)
            x_time = self.time_linear_second(x_time)
        elif self.pooling == "parameterised_mean":
            sensor_weights = self.linear_layer_sensor(x_sensor)
            time_weight = self.linear_layer_time(x_time)
            x_sensor = (x_sensor @ sensor_weights) / sensor_weights.sum().squeeze()
            x_time = (x_time @ time_weight) / time_weight.sum().squeeze()
        elif self.pooling == "conv":
            x_sensor = self.time_conv_second(x_sensor).squeeze()
            x_time = self.sensor_conv_first(x_time).squeeze()

        # concatenate poolingated attented tensors
        x_merged = torch.cat((x_sensor, x_time), dim=1)  # (N, F+T)
        x_merged = self.nonlinear_merger_1(x_merged).relu()

        if self.use_static:
            static = self.static_embedding(static)  # .relu()
            x_merged = torch.cat((x_merged, static), axis=1)
            nonlinear_merged = self.nonlinear_merger_2(x_merged).relu()
        else:
            nonlinear_merged = x_merged

        if self.return_intermediates:
            return (
                sensor_intermediates.attn_intermediates[0].post_softmax_attn,
                time_intermediates.attn_intermediates[0].post_softmax_attn,
            )
       
        return nonlinear_merged
    
@gin.configurable
class BAT(CustomDLPredictionWrapper):
    """Wrapper to integrate EncoderPrediction with CustomDLPredictionWrapper logic."""

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
        prediction_head=BinaryClassificationHead,
        prediction_head_kwargs={"num_classes": 2},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)
        # Extract dimensions from dataset
        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)  # fallback if static shape isn't passed

        # Instantiate encoder
        encoder = EncoderClassifierCrossParallel(
            device=self.device,
            pooling="max",
            value_embed_size=value_embed_size,
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

        # For compatibility with classification output detection
        #self.logit = nn.Linear(1, prediction_head_kwargs.get("num_classes", 2))  # dummy shape

    def forward(self, data, static, time, sensor_mask):
        return self.model(data, static=static, time=time, sensor_mask=sensor_mask)
    
@gin.configurable
class SSL_BAT(SSLWrapper):
    """Wrapper to integrate EncoderPrediction with CustomDLPredictionWrapper logic."""

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
        prediction_head=ForecastingHead,
        prediction_head_kwargs={"sensors_count": 48, "forecast_len": 2},
        lr=1e-4,
        optimizer=torch.optim.Adam,
        *args,
        **kwargs
    ):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)
        # Extract dimensions from dataset
        sensors_count = input_size[1]
        max_timepoint_count = input_size[2]
        static_count = kwargs.get("static_count", 4)  # fallback if static shape isn't passed

        # Instantiate encoder
        encoder = EncoderClassifierCrossParallel(
            device=self.device,
            pooling="max",
            value_embed_size=value_embed_size,
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
