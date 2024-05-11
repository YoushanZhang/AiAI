# Sabda2Baachan

This work proposes a novel neural architecture Sabda2Baachan for text-to-speech synthesis capable of producing high-quality speech with natural prosody and speaker characteristics. The model employs a multi-stream approach, where distinct components predict various low-level prosodic features, including energy, pitch, and duration. The proposed model demonstrated superior performance compared to several state-of-the-art models, achieving remarkable naturalness, intelligibility, and speaker similarity in the synthesized speech.

<a href="https://youshanzhang.github.io/AiAI/High-Quality-Voice-Cloning/" target="_blank">See the github website</a>

<div align="center">
<a><img width="720" src="final-custom-model-architecture.png" alt="soft"></a>
</div>

## Dependencies

You can install the Python dependencies with

```
pip3 install -r requirements.txt
```

# Config

Here are the config files used to train the single/multi-speaker TTS models.

- LibriTTS: suggested configuration for LibriTTS dataset.

Setup Your model.yaml, train.yaml and preprocess.yml before everything inside config

# Datset Splits

<table>
    <caption>Custom Dataset Split for Training, Validation, and Testing to fine-tune both Custom Models.</caption>
    <thead>
        <tr>
            <th>Dataset Split</th>
            <th>Audio Clips (.wav)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Total Dataset</td>
            <td>500</td>
        </tr>
        <tr>
            <td>Training Dataset</td>
            <td>350 (70%)</td>
        </tr>
        <tr>
            <td>Validation Dataset</td>
            <td>50 (10%)</td>
        </tr>
        <tr>
            <td>Test Dataset</td>
            <td>100 (20%)</td>
        </tr>
    </tbody>
</table>

# Preprocessing

```
python prepare_align.py config/{Your voice Dataset}/preprocess.yaml
```

to align the corpus and then run the preprocessing script - Followed by

```
python preprocess.py config/{Your voice Dataset}/preprocess.yaml
```

# Training

```
python3 train.py -p config/{Your voice Dataset}/preprocess.yaml -m config/{Your voice Dataset}/model.yaml -t config/{Your voice Dataset}/train.yaml
```

# Inference

You can download the [pretrained models] https://drive.google.com/drive/folders/1GCuMiniTQZHeTwHEgAo6dS9bQquvVDs5?usp=drive_link
and put them in output/ckpt/{Your voice Dataset}/

For vocoder,download pretrained from https://drive.google.com/drive/folders/1GCuMiniTQZHeTwHEgAo6dS9bQquvVDs5?usp=drive_link and put it in Hifi-Gan

```
python3 synthesize.py --text "YOUR DESIRED TEXT" --restore_step 900000 --mode single -p config/{Your voice Dataset}/preprocess.yaml -m config/{Your voice Dataset}/model.yaml -t config/{Your voice Dataset}/train.yaml
```

https://github.com/SiddantaK/High-Quality-Voice-Cloning/assets/101938746/18f762ac-a29d-44c0-86f8-bd69b0ee002d

# Results

<div class='section table-responsive-sm'>
    <table class="table">
    <h2>Evaluation Metrics  for Different Datasets Split</h2>
    <thead>
        <tr>
            <th>Splits</th>
            <th>PESQ(nb)</th>
            <th>PESQ(wb)</th>
            <th>SDR</th>
            <th>SNR</th>
            <th>STOI</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training</td>
            <td>2.960</td>
            <td>2.722</td>
            <td>5.637</td>
            <td>6.179</td>
            <td>0.823</td>
        </tr>
        <tr>
            <td>Validation</td>
            <td>2.535</td>
            <td>2.331</td>
            <td>5.527</td>
            <td>5.137</td>
            <td>0.797</td>
        </tr>
        <tr>
            <td>Testing</td>
            <td>2.377</td>
            <td>2.004</td>
            <td>5.662</td>
            <td>5.029</td>
            <td>0.635</td>
        </tr>
    </tbody>
</table>
</div>

# Cloned Outputs

Here you can listen the output of our cloned voice along with its ground truth audio.
https://drive.google.com/drive/folders/1FTqapXz9Z1kqtOPNUKaHX5GVVw0e7_8y?usp=sharing

# Download model weight here

https://drive.google.com/drive/folders/1at6L_zuqVHkL3gZeZhjF9rVYe-idgEVa?usp=sharing

<div align="center">
<a><img width="720" src="synthesized.png" alt="soft"></a>
</div>

<div align="center">
<a><img width="720" src="Loss_total_loss.svg" alt="soft"></a>
</div>

# For Evaluation

You can use metrics.py inside the "Sabda2Baachan" folder for evaluation of the generated audios with actual one

```
stoi,nb_pesq_score,wb_pesq_score,snr,sdr = metrics(wav_reconstructed,wav_predict)
```

From here you can obtain ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality(narrow band), PerceptualEvaluationSpeechQuality(Wide band), SignalNoiseRatio, SignalDistortionRatio

# References

<a href="https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner" target="_blank">Montreal Forced Aligner</a> <br/>
<a href="https://github.com/ming024/FastSpeech2" target="_blank">Fast Speech 2 </a> <br/>
<a href="https://github.com/keonlee9420/WaveGrad2" target="_blank">keonlee9420's WaveGrad2</a> for GaussianUpsampling<br/>
<a href="https://arxiv.org/abs/1709.07871" target="_blank">FiLM: Visual Reasoning with a General Conditioning Layer</a><br/>
<a href="https://github.com/keonlee9420/Daft-Exprt" target="_blank"> Daft-Exprt: Robust Prosody Transfer Across Speakers for Expressive Speech Synthesis</a><br/>
