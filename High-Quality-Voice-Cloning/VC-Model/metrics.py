import soundfile as sf
from pystoi import stoi
import pesq
import librosa
import torch
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio,ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality

def calc_stoi(wav_reconstructed,wav_predict):
    stoi = ShortTimeObjectiveIntelligibility(8000)
    stois = stoi(wav_predict, wav_reconstructed)
    # print(stois,'stoi')
    return stois

def calc_pesq(original,cloned):
    nb_pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
    wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
    wb_pesq_score = wb_pesq(cloned, original)
    nb_pesq_score = nb_pesq(cloned, original)
    # print(nb_pesq_score,wb_pesq_score,'pesq')
    return nb_pesq_score,wb_pesq_score

def calc_snr(original,cloned):
    snr = SignalNoiseRatio()
    snr_value = snr(cloned, original)
    # print(snr_value,'snr')
    return snr_value

def calc_sdr(original,cloned):
    sdr = SignalDistortionRatio()
    sdr_value = sdr(cloned, original)
    # print(sdr_value,'sdr')
    return sdr_value

def metrics(wav_reconstructed,wav_predict):
    wav_reconstructed = torch.from_numpy(wav_reconstructed)
    wav_predict = torch.from_numpy(wav_predict)
    stoi = calc_stoi(wav_reconstructed,wav_predict)
    nb_pesq_score,wb_pesq_score = calc_pesq(wav_reconstructed,wav_predict)
    snr = calc_snr(wav_reconstructed.float(),wav_predict.float())
    sdr = calc_sdr(wav_reconstructed.float(),wav_predict.float())
    return stoi,nb_pesq_score,wb_pesq_score,snr,sdr