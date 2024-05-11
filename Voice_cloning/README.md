# High Quality Voice Cloning

Proposed Text-to-Speech (TTS) system architecture represents a meticulously designed sequence of components aimed at synthesizing natural and expressive voice from input text. At its core are three major components: the Text Encoder, Mel Spectrogram Encoder, and Voice Cloning Model. The Text Encoder serves as the initial step, translating input text into a robust representation through a series of intricate procedures, including character embeddings, bidirectional GRU processing, and attention mechanisms. Following this, the Mel Spectrogram Encoder generates an encoded representation of the mel spectrogram, capturing crucial acoustic subtleties. Finally, the Voice Cloning Model combines these encoded representations, employing a detailed decoder architecture with RNN layers and attention mechanisms to synthesize speech. Notably, the incorporation of attention mechanisms throughout enhances the model's ability to focus on specific parts of the input sequence, resulting in more realistic and contextually relevant speech synthesis. 

## Dataset
https://drive.google.com/drive/folders/1mry8U9Oo2OgHiJFQh0AUjFAAfuAhbGrz?usp=sharing

## Model Training and Validation Metrics

| Model     | STOI  | NB PESQ | WB PESQ | SNR   | SDR   |
|-----------|-------|---------|---------|-------|-------|
| Model 1   | 0.7089| 2.4835  | 2.7810  | 5.6912| 4.9329|
| Model 2   | 0.5789| 1.9785  | 2.3781  | 4.8869| 4.1840|
| AudioLDM  | 0.6825| 2.9904  | 2.8158  | 5.4882| 5.401 |
| Tecotron  | 0.6785| 2.3229  | 2.1754  | 5.9902| 2.3560|
| Bark      | 0.7342| 2.8499  | 2.6745  | 5.3992| 5.8611|
| Tortoise  | 0.8823| 3.2788  | 3.1468  | 7.9871| 6.2952|

## Validation Metrics

| Model     | STOI  | NB PESQ | WB PESQ | SNR   | SDR   |
|-----------|-------|---------|---------|-------|-------|
| Model 1   | 0.4567| 1.8934  | 1.9645  | 5.6781| 2.5489|
| Model 2   | 0.4250| 1.8131  | 1.4977  | 4.2401| 2.4110|
| AudioLDM  | 0.5609| 2.1933  | 1.8922  | 5.2032| 2.9756|
| Tecotron  | 0.2449| 1.2219  | 1.3901  | 5.2554| 1.889 |
| Bark      | 0.6194| 2.1876  | 1.8940  | 4.1945| 4.1896|
| Tortoise  | 0.6345| 2.4611  | 2.2362  | 6.9064| 5.4530|


## Predictions
https://siddantak.github.io/High-Quality-Voice-Cloning/

## Weights
https://drive.google.com/drive/folders/1XAEC5EWQdiefKi0HrtHlr_WEbC-AQizc?usp=sharing

