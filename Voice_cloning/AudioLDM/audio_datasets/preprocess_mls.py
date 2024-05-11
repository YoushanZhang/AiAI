import csv
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
from audio_datasets.constants import ENCODEC_REDUCTION_FACTOR, ENCODEC_SAMPLING_RATE

MAX_DURATION_IN_SECONDS = 20


def round_up_to_multiple(number, multiple):
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        return number + (multiple - remainder)

def compute_max_length(multiple=128):
    max_len = MAX_DURATION_IN_SECONDS*ENCODEC_SAMPLING_RATE
    waveform_multiple = multiple*ENCODEC_REDUCTION_FACTOR

    max_len = round_up_to_multiple(max_len, waveform_multiple)
    return max_len

def is_audio_length_in_range(audio, sampling_rate):
    return len(audio) <= (MAX_DURATION_IN_SECONDS*sampling_rate)

def main():
    max_length = compute_max_length()
    # Define the header names
    headers = ['file_name', 'text']
    data_dir = 'D:\\audioldm\\tts\\simple-tts\\data\\mls\\mls_english'
    

    for split in ['train', 'dev', 'test']:
        print(f'Converting {split} split...')
        # Specify the input and output file paths
        split_dir = os.path.join(data_dir, split)
        input_file = os.path.join(split_dir, f'transcripts.txt')
        output_file = os.path.join(split_dir, f'metadata.csv')

        # Open the input file for reading
        with open(input_file, 'r') as file:
            # Create a CSV writer for the output file
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write the headers to the CSV file
                writer.writerow(headers)
                # Read each line in the input file
                for line in tqdm(file):
                    parts = line.strip().split('\t')
                    if len(parts) == 2:  # Ensure exactly two parts are present
                        audio_id, description = parts
                        audio_id_parts = audio_id.split('_')
                        if len(audio_id_parts) == 3:  # Ensure there are three parts separated by underscores
                            speaker_id, book_id, file_id = audio_id_parts
                            file_path = os.path.join('audio', speaker_id, book_id, f'{audio_id}.wav')
                            audio_file_path = os.path.join(split_dir, file_path)
                            print(f"Loading audio file: {audio_file_path}")
                            audio, samplerate = sf.read(audio_file_path)
                            if is_audio_length_in_range(audio, samplerate):
                                print(f"Audio loaded successfully with sampling rate {samplerate} Hz.")
                                # Write the file path and description as a row in the CSV file
                                writer.writerow([file_path, description])
                            else:
                                print(f"Audio {audio_file_path} exceeds maximum duration.")
                        else:
                            print(f"Issue with audio ID format: {audio_id}")
                    else:
                        print(f"Issue with line format: {line}")

        print('Conversion complete!')


if __name__ == "__main__":
    main()