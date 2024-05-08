import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import transformer.Constants as Constants
import params as hp
from text.symbols import symbols

from model.modules import MultiHeadAttention, ConvNorm, LinearNorm, RangeParameterPredictor, Conv,FiLM,FFTBlock
import math
import utils
from collections import OrderedDict

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class SpeakerIntegrator(nn.Module):
    """ Speaker Integrator """

    def __init__(self):
        super(SpeakerIntegrator, self).__init__()

    def forward(self, x, spembs):
        """
        x      shape : (batch, 39, 256)
        spembs shape : (batch, 256)
        """
        # spembs = spembs.unsqueeze(1)
        spembs = spembs.repeat(1, x.shape[1], 1)
        x = x + spembs

        return x

class Encoder(nn.Module):
    """ Encoder with Residual Connections and Layer Normalization """

    def __init__(
        self,
        model_config,
    ):
        super(Encoder, self).__init__()
        n_src_vocab=len(symbols) + 1
        len_max_seq = model_config["max_seq_len"]
        d_word_vec= model_config["prosody_encoder"]["encoder_hidden"]
        d_model = model_config["prosody_encoder"]["encoder_hidden"]
        n_layers=model_config["prosody_encoder"]["encoder_layer"]
        n_head=model_config["prosody_encoder"]["encoder_head"]
        d_k=model_config["prosody_encoder"]["encoder_hidden"] // model_config["prosody_encoder"]["encoder_head"]
        d_v=model_config["prosody_encoder"]["encoder_hidden"]// model_config["prosody_encoder"]["encoder_head"]
        dropout= model_config["prosody_encoder"]["dropout"]

        n_position = len_max_seq + 1
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )

      
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                MultiHeadAttention(
                    n_head,d_model, d_k, d_v, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > hp.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], hp.encoder_hidden
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            # Residual connection + Layer Normalization
            residual = enc_output
            enc_output, enc_slf_attn  = enc_layer(enc_output, enc_output, enc_output, slf_attn_mask)
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
            if return_attns:
                enc_slf_attn_list += [enc_layer.attn_weights]
        return enc_output

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor()
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(hp.f0_min), np.log(hp.f0_max), hp.n_bins - 1)
            )
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(hp.energy_min, hp.energy_max, hp.n_bins - 1)
        )
        self.pitch_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)
        self.energy_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)

    # 1. predict duration
    # 2. predict f0     -> embedding
    # 3. predict energy -> embedding
    # 4. x + pitch_embedding + energy_embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        max_len=None,
        gammas=None,
        betas=None,
    ):

        ## Duration Predictor ##
        log_duration_prediction = self.duration_predictor(x, src_mask,gammas,betas)

        if duration_target is not None:
            duration_values = duration_target
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - hp.log_offset), min=0
            )
            duration_values = duration_rounded

        ## Pitch Predictor ##
        pitch_prediction = self.pitch_predictor(x, src_mask,gammas,betas)

        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_target.detach(), self.pitch_bins.detach())
            )
        else:
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_prediction.detach(), self.pitch_bins.detach())
            )

        ## Energy Predictor ##
        energy_prediction = self.energy_predictor(x, src_mask,gammas,betas)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_target.detach(), self.energy_bins.detach())
            )
        else:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_prediction.detach(), self.energy_bins.detach())
            )
        x = x + pitch_embedding + energy_embedding

        return (
            x,
            log_duration_prediction,
            pitch_prediction,
            energy_prediction,

            mel_mask,
        )

class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(VariancePredictor, self).__init__()
        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.film = FiLM()

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask,gammas,betas):
        out = self.conv_layer(encoder_output)
        out = self.film(out, gammas, betas)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class ProsodyEncoder(nn.Module):
    """ Prosody Encoder """

    def __init__(self, preprocess_config,model_config):
        super(ProsodyEncoder, self).__init__()

        self.max_seq_len = model_config["max_seq_len"] + 1
        n_conv_layers = model_config["prosody_encoder"]["conv_layer"]
        kernel_size = model_config["prosody_encoder"]["conv_kernel_size"]
        self.d_model = model_config["prosody_encoder"]["encoder_hidden"]
        n_mel_channels =preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.filter_size = model_config["prosody_encoder"]["conv_filter_size"]
        n_layers=model_config["prosody_encoder"]["encoder_layer"]
        n_head=model_config["prosody_encoder"]["encoder_head"]
        d_k=model_config["prosody_encoder"]["encoder_hidden"] // model_config["prosody_encoder"]["encoder_head"]
        d_v=model_config["prosody_encoder"]["encoder_hidden"]// model_config["prosody_encoder"]["encoder_head"]
        dropout= model_config["prosody_encoder"]["dropout"]
        
        self.p_embedding = ConvNorm(
            1,
            self.filter_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )

        self.e_embedding = ConvNorm(
            1,
            self.filter_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )

        self.layer_stack = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNorm(
                            # 1,
                            n_mel_channels if i == 0 else self.filter_size,
                            self.filter_size,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                            dilation=1,
                            transform=True,
                        ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(dropout),
                )
                for i in range(n_conv_layers)
            ]
        )
    
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(self.max_seq_len, self.filter_size).unsqueeze(0),
            requires_grad=False,
        )
        self.fftb_linear = LinearNorm(self.filter_size, self.d_model)
        self.fftb_stack = nn.ModuleList(
            [
                FFTBlock(
                    self.d_model, n_head, d_k, d_v, self.filter_size, [kernel_size, kernel_size], dropout=dropout, film=False
                )
                for _ in range(n_layers)
            ]
        )

        self.feature_wise_affine = LinearNorm(self.d_model, 2 * self.d_model)

    def forward(self, mel, max_len, mask, pitch, energy, spker_embed):

        batch_size = mel.shape[0]
        pitch = self.p_embedding(pitch.unsqueeze(-1))
        energy = self.e_embedding(energy.unsqueeze(-1))
        # -- Prepare Input
        enc_seq = mel
        for enc_layer in self.layer_stack:
            enc_seq = enc_layer(enc_seq)
        # Make sure the shape of pitch and energy matches enc_seq
        pitch = pitch.unsqueeze(1).expand(-1, enc_seq.shape[1], -1, -1)
        energy = energy.unsqueeze(1).expand(-1, enc_seq.shape[1], -1, -1)
        # enc_seq = torch.mean(enc_seq, dim=2, keepdim=True)
        pitch = torch.sum(pitch, dim=2)  # Sum along the third dimension
        energy = torch.sum(energy, dim=2)  # Sum along the third dimension
        enc_seq = enc_seq + pitch + energy

        if mask is not None:
            enc_seq = enc_seq.masked_fill(mask.unsqueeze(-1), 0.0)

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_seq = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.filter_size
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_seq = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        enc_seq = self.fftb_linear(enc_seq)
        for fftb_layer in self.fftb_stack:
            enc_seq, _ = fftb_layer(
                enc_seq, mask=mask, slf_attn_mask=slf_attn_mask
            )

        # -- Avg Pooling
        prosody_vector = enc_seq.mean(dim=1, keepdim=True) # [B, 1, H]
        # -- Feature-wise Affine
        gammas, betas = torch.split(
            self.feature_wise_affine(prosody_vector + spker_embed), self.d_model, dim=-1
        )

        return gammas, betas

class GaussianUpsampling(nn.Module):
    """ Gaussian Upsampling """

    def __init__(self,preprocess_config, model_config):
        super(GaussianUpsampling, self).__init__()
        kernel_size = model_config["prosody_predictor"]["kernel_size"]
        self.log_duration = preprocess_config["preprocessing"]["duration"]["log_duration"]
        self.prosody_hidden = model_config["transformer"]["encoder_hidden"]

        self.p_embedding = ConvNorm(
            1,
            self.prosody_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.e_embedding = ConvNorm(
            1,
            self.prosody_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.d_embedding = ConvNorm(
            1,
            self.prosody_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            transform=True,
        )
        self.range_param_predictor_paper = nn.Sequential(
            nn.Linear(self.prosody_hidden, 1),
            nn.Softplus(),
        )
        self.range_param_predictor = RangeParameterPredictor()
    def get_alignment_energies(self, gaussian, frames):
        log_prob = gaussian.log_prob(frames)
        energies = log_prob.exp()  # [B, L, T]
        return energies
    
    def forward(self,
        encoder_outputs,
        p_prediction,
        e_prediction,
        d_prediction,
        p_targets=None,
        e_targets=None,
        d_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        src_mask=None,
        mel_len=None,
        max_mel_len=None,
        mel_mask=None,
        src_len=None,
    ):
        device = p_prediction.device

        if d_target is not None:
            p_prediction = p_targets
            e_prediction = e_targets
            d_prediction = d_target.float()
        else:
            p_prediction = p_prediction * p_control
            e_prediction = e_prediction * e_control
            d_prediction = torch.clamp(
                (torch.round(torch.exp(d_prediction) - 1) * d_control),
                min=0
            ).float() if self.log_duration else (d_prediction * d_control)
            mel_len = d_prediction.sum(-1).int().to(device)
            max_mel_len = mel_len.max().item()
            mel_mask = utils.get_mask_from_lengths(mel_len, max_mel_len)
        # Prosody Projection
        p_embed = self.p_embedding(p_prediction.unsqueeze(-1))
        e_embed = self.e_embedding(e_prediction.unsqueeze(-1))
        d_embed = self.d_embedding(d_prediction.unsqueeze(-1))
        # Range Prediction

        s_input = p_embed + e_embed + d_embed + encoder_outputs
        s = self.range_param_predictor(s_input, src_len, src_mask).unsqueeze(-1) if src_len is not None \
            else self.range_param_predictor_paper(s_input)
        if src_mask is not None:
            s = s.masked_fill(src_mask.unsqueeze(-1), 1e-8)

                # Gaussian Upsampling
        t = torch.sum(d_prediction, dim=-1, keepdim=True) #[B, 1]
        e = torch.cumsum(d_prediction, dim=-1).float() #[B, L]
        c = e - 0.5 * d_prediction #[B, L]
        t = torch.arange(1, torch.max(t).item()+1, device=device) # (1, ..., T)
        t = t.unsqueeze(0).unsqueeze(1) #[1, 1, T]
        c = c.unsqueeze(2)

        g = torch.distributions.normal.Normal(loc=c, scale=s)

        w = self.get_alignment_energies(g, t)  # [B, L, T]

        if src_mask is not None:
            w = w.masked_fill(src_mask.unsqueeze(-1), 0.0)

        attn = w / (torch.sum(w, dim=1).unsqueeze(1) + 1e-8)  # [B, L, T]
        out = torch.bmm(attn.transpose(1, 2), p_embed + e_embed + encoder_outputs)

        return out, attn, mel_len, max_mel_len, mel_mask, d_prediction

class Decoder(nn.Module):
    """ Decoder """

    def __init__(
        self,config

    ):
        n_position = config["max_seq_len"] + 1
        d_word_vec=config["transformer"]["decoder_hidden"]
        n_layers=config["transformer"]["decoder_layer"]
        n_head=config["transformer"]["decoder_head"]
        d_k=config["transformer"]["decoder_hidden"]// config["transformer"]["decoder_head"]
        d_v=config["transformer"]["decoder_hidden"]// config["transformer"]["decoder_head"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        self.d_model=config["transformer"]["decoder_hidden"]
        d_inner=config["transformer"]["conv_filter_size"]
        dropout=config["transformer"]["decoder_dropout"]

        super(Decoder, self).__init__()
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(self.d_model, n_head, d_k, d_v,d_inner, [kernel_size, kernel_size],dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask,  gammas, betas,return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and enc_seq.shape[1] > hp.max_seq_len:
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], hp.decoder_hidden
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            dec_output = enc_seq + self.position_enc[:, :max_len, :].expand(
                batch_size, -1, -1
            )

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, gammas, betas, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output,mask
