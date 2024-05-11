import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformer import Encoder, Decoder, PostNet
from utils.tools import get_mask_from_lengths
from model.models import Encoder, ProsodyEncoder,SpeakerIntegrator,GaussianUpsampling, VarianceAdaptor,Decoder
from model.modules import Embedding

class VoiceCloningModel(nn.Module):
    """ VoiceCloningModel """

    def __init__(self, preprocess_config, model_config, n_spkers= 1, spker_embed_dim=128, spker_embed_std=0.01):
        super(VoiceCloningModel, self).__init__()
        self.model_config = model_config
        
        self.n_spkers = n_spkers
        self.spker_embed_dim = spker_embed_dim
        self.spker_embed_std = spker_embed_std
        self.speaker_integrator = SpeakerIntegrator()
        self.encoder = Encoder(model_config)

        self.spker_embeds = Embedding(
            n_spkers, spker_embed_dim, padding_idx=None, std=spker_embed_std
        )

        self.prosody_enc = ProsodyEncoder(preprocess_config,model_config)

        self.variance_adaptor = VarianceAdaptor()

        self.gaussian_upsample = GaussianUpsampling(preprocess_config, model_config)

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

    def forward(self,ids, speaker_id,text_seq,
        text_len,
        d_gt=None,
        p_gt=None,
        e_gt=None,
        mel_len=None,
        max_text_len=None,
        max_mel_len=None,mel_gt=None):

        spker_embed = self.spker_embeds(speaker_id).unsqueeze(1)
        text_mask = get_mask_from_lengths(text_len, max_text_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len)
        encoder_out = self.encoder(text_seq,text_mask)
        encoder_out = self.speaker_integrator(encoder_out,spker_embed)
        gammas, betas = self.prosody_enc(mel_gt,max_mel_len, mel_mask, p_gt, e_gt,spker_embed)

        # === Variance Adaptor === #
        if d_gt is not None:
            (
                variance_adaptor_output,
                d_pred,
                p_pred,
                e_pred,
                _,
            ) = self.variance_adaptor(
                encoder_out, text_mask, mel_mask, d_gt, p_gt, e_gt, max_mel_len, gammas,betas
            )
        else:
            (
                variance_adaptor_output,
                d_pred,
                p_pred,
                e_pred,
                mel_mask,
            ) = self.variance_adaptor(
                encoder_out, text_mask, mel_mask, d_gt, p_gt, e_gt, max_mel_len,gammas,betas
            )

        variance_adaptor_output = self.speaker_integrator(
            variance_adaptor_output, spker_embed
        )

        out, attn, mel_len, max_mel_len, mel_mask, d_prediction =  self.gaussian_upsample(
            variance_adaptor_output,
            p_pred,
            e_pred,
            d_pred,
            p_gt,
            e_gt,
            d_gt,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
            src_mask=text_mask,
            mel_len=mel_len,
            max_mel_len=max_mel_len,
            mel_mask=mel_mask,
            src_len=text_len,
            )

        output, mel_mask = self.decoder(out, mel_mask,gammas, betas)
        output = self.mel_linear(output)
        return output,p_pred,e_pred, d_pred,text_mask,mel_mask,mel_len,text_len