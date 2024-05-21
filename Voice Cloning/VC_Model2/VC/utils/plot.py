# -*- coding: utf-8 -*- #

import numpy as np
import librosa.display
from utils import audio
from config import config
import matplotlib.pyplot as plt
plt.switch_backend('agg')


#############
# CONSTANTS #
#############
fs = config.sample_rate
win = config.frame_length_ms
hop = config.frame_shift_ms
nfft = (config.num_freq - 1) * 2
hop_length = config.hop_length


############################
# PREPROCESS VISUALIZATION #
############################
"""
	visualize the preprocessed waveforms
"""
def preprocess_visualization(name, y, yt, sr, output_dir, visualization_dir, vis_process):
    # ---visualization---#
    plt.figure(figsize=(16, 4))

    if vis_process:
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(yt, sr=sr, color='r')  # Use librosa.display.waveshow
        plt.title('Processed Waveform')

        plt.subplot(2, 1, 2)

    librosa.display.waveshow(y, sr=sr, color='tab:orange')  # Use librosa.display.waveshow
    plt.title('Original Waveform')

    plt.tight_layout()

    # ---save---#
    plt.savefig(visualization_dir + name + '.jpeg')
    plt.close()

##################
# PLOT ALIGNMENT #
##################
def plot_alignment(alignment, path, info=None):
	plt.gcf().clear()
	fig, ax = plt.subplots()
	im = ax.imshow(
		alignment,
		aspect='auto',
		origin='lower',
		interpolation='none')
	fig.colorbar(im, ax=ax)
	xlabel = 'Decoder timestep'
	if info is not None:
		xlabel += '\n\n' + info
	plt.xlabel(xlabel)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()
	plt.savefig(path, dpi=300, format='png')
	plt.close()


####################
# PLOT SPECTROGRAM #
####################
def plot_spectrogram(linear_output, path):
	spectrogram = audio._denormalize(linear_output)
	plt.gcf().clear()
	plt.figure(figsize=(16, 10))
	plt.imshow(spectrogram.T, aspect="auto", origin="lower")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(path, dpi=300, format="png")
	plt.close()	


##################
# TEST VISUALIZE #
##################
def test_visualize(alignment, spectrogram, path):
	
	_save_alignment(alignment, path)
	_save_spectrogram(spectrogram, path)
	label_fontsize = 16
	plt.gcf().clear()
	plt.figure(figsize=(16,16))
	
	plt.subplot(2,1,1)
	plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
	plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
	plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
	plt.colorbar()

	plt.subplot(2,1,2)
	librosa.display.specshow(spectrogram.T, sr=fs, 
							 hop_length=hop_length, x_axis="time", y_axis="linear")
	plt.xlabel("Time", fontsize=label_fontsize)
	plt.ylabel("Hz", fontsize=label_fontsize)
	plt.tight_layout()
	plt.colorbar()

	plt.savefig(path + '_all.png', dpi=300, format='png')
	plt.close()


##################
# SAVE ALIGNMENT #
##################
def _save_alignment(alignment, path):
	plt.gcf().clear()
	plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
	plt.xlabel("Decoder timestamp")
	plt.ylabel("Encoder timestamp")
	plt.colorbar()
	plt.savefig(path + '_alignment.png', dpi=300, format='png')


####################
# SAVE SPECTROGRAM #
####################
def _save_spectrogram(spectrogram, path):
	plt.gcf().clear()  # Clear current previous figure
	cmap = plt.get_cmap('jet')
	t = win + np.arange(spectrogram.shape[0]) * hop
	f = np.arange(spectrogram.shape[1]) * fs / nfft
	plt.pcolormesh(t, f, spectrogram.T, cmap=cmap)
	plt.xlabel('Time (sec)')
	plt.ylabel('Frequency (Hz)')
	plt.colorbar()
	plt.savefig(path + '_spectrogram.png', dpi=300, format='png')
