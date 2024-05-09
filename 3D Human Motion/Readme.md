# Text Residual Motion Encoder for 3D Human Motion Generation

![teaser_image](https://github.com/YoushanZhang/AiAI/blob/main/3D%20Human%20Motion/T2M/Screenshot%202024-04-29%20190517.png)

## Abstract
In the domain of 3D human motion generation from textual descriptions, the TRME model significantly improves architectural flexibility and dataset diversity. By incorporating an additional layer of residual blocks into the Vector Quantized Variational Autoencoder (VQ-VAE) and utilizing the comprehensive "CHAD" dataset, our model captures finer motion details, improving both the diversity and quality of generated human motions. 

## Repository Contents
- `src/` - All source code for the TRME model and dataset processing.
- `data/` - Scripts to process and prepare the CHAD dataset.
- `models/` - Pre-trained models and configuration files.
- `notebooks/` - Jupyter notebooks for demonstration purposes.
- `docs/` - Additional documentation and results.
- `environment.yml` - Conda environment file to set up the Python environment.

## Installation
To set up the necessary environment:

```bash
conda env create -f environment.yml

## Data Preparation
CHAD dataset is a combination of several datasets, providing diverse motion categories suitable for robust training:


## Training
To train the model:


## Evaluation
Evaluate the trained model


## Visualization
Visualize motion predictions from textual descriptions

## Citations/References:

If you are using HumanML3D dataset, please consider citing the following papers:

@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}


<a href="https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner" target="_blank">Montreal Forced Aligner</a> <br/>
<a href="https://github.com/ming024/FastSpeech2" target="_blank">Fast Speech 2 </a> <br/>
<a href="https://github.com/keonlee9420/WaveGrad2" target="_blank">keonlee9420's WaveGrad2</a> for GaussianUpsampling<br/>
<a href="https://arxiv.org/abs/1709.07871" target="_blank">FiLM: Visual Reasoning with a General Conditioning Layer</a><br/>
<a href="https://github.com/keonlee9420/Daft-Exprt" target="_blank"> Daft-Exprt: Robust Prosody Transfer Across Speakers for Expressive Speech Synthesis</a><br/>

## Acknowledgement

We appreciate prior contributions as a source of inspiration for Text to 3D Human Motion Generation from :  

* public code like [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [TM2T](https://github.com/EricGuo5513/TM2T), [MDM](https://github.com/GuyTevet/motion-diffusion-model).







