# -*- coding: utf-8 -*- #



import os
import sys
import nltk
import argparse
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from pypinyin import Style, pinyin
import soundfile as sf
#--------------------------------#
import torch
from torch.autograd import Variable
#--------------------------------#
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize, plot_alignment
#--------------------------------#
from model import Model
from config import config, get_test_args


############
# CONSTANT #
############
USE_CUDA = torch.cuda.is_available()


##################
# TEXT TO SPEECH #
##################
def tts(model, text):
	
	if USE_CUDA:
		model = model.cuda()
	# TODO: Turning off dropout of decoder's prenet causes serious performance regression, not sure why.
	# model.decoder.eval()
	model.encoder.eval()
	model.postnet.eval()

	sequence = np.array(text_to_sequence(text))
	sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
	if USE_CUDA:
		sequence = sequence.cuda()

	# Greedy decoding
	mel_outputs, linear_outputs, alignments = model(sequence)

	linear_output = linear_outputs[0].cpu().data.numpy()
	spectrogram = audio._denormalize(linear_output)
	alignment = alignments[0].cpu().data.numpy()

	# Predicted audio signal
	waveform = audio.inv_spectrogram(linear_output.T)

	return waveform, alignment, spectrogram


####################
# SYNTHESIS SPEECH #
####################
def synthesis_speech(model, text, figures=True, path=None):
	waveform, alignment, spectrogram = tts(model, text)
	if figures:
		test_visualize(alignment, spectrogram, path)
	# librosa.output.write_wav(path + '.wav', waveform, config.sample_rate)
	sf.write(path + '.wav', waveform, config.sample_rate)



#############
# CH2PINYIN #
#############
def ch2pinyin(txt_ch):
	ans = pinyin(txt_ch, style=Style.TONE2, errors=lambda x: x, strict=False)
	return ' '.join([x[0] for x in ans if x[0] != 'EMPH_A'])


########
# MAIN #
########
def main():

	#---initialize---#
	args = get_test_args()

	model = Model(n_vocab=len(symbols),
					 embedding_dim=config.embedding_dim,
					 mel_dim=config.num_mels,
					 linear_dim=config.num_freq,
					 r=config.outputs_per_step,
					 padding_idx=config.padding_idx,
					 use_memory_mask=config.use_memory_mask)

	#---handle path---#
	checkpoint_path = os.path.join(args.ckpt_dir, args.checkpoint_name + '470000' + '.pth')
	os.makedirs(args.result_dir, exist_ok=True)
	
	#---load and set model---#
	print('Loading model: ', checkpoint_path)
	checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint["state_dict"])
	
	if args.long_input:
		model.decoder.max_decoder_steps = 500 # Set large max_decoder steps to handle long sentence outputs
	else:
		model.decoder.max_decoder_steps = 50
		
	if args.interactive == True:
		output_name = args.result_dir + args.model

		#---testing loop---#
		while True:
			try:
				text = str(input('< Model > Text to speech: '))
				text = ch2pinyin(text)
				print('Model input: ', text)
				synthesis_speech(model, text=text, figures=args.plot, path=output_name)
			except KeyboardInterrupt:
				print()
				print('Terminating!')
				break

	elif args.interactive == False:
		output_name = args.result_dir + args.model + '/'
		os.makedirs(output_name, exist_ok=True)

		#---testing flow---#
		with open(args.test_file_path, 'r', encoding='utf-8') as f:
			
			lines = f.readlines()
			for idx, line in enumerate(lines):
				text = ch2pinyin(line)
				print("{}: {} - {} ({} words, {} chars)".format(idx, line, text, len(line), len(text)))
				synthesis_speech(model, text=text, figures=args.plot, path=output_name+line)

		print("Finished! Check out {} for generated audio samples.".format(output_name))
	
	else:
		raise RuntimeError('Invalid mode!!!')
		
	sys.exit(0)

if __name__ == "__main__":
	main()
