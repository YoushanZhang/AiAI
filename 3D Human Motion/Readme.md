# Text Residual Motion Encoder for 3D Human Motion Generation

![teaser_image](https://github.com/YoushanZhang/AiAI/blob/main/3D%20Human%20Motion/motions.png)


## Table of Contents

- [Text Residual Motion Encoder for 3D Human Motion Generation](#Text-Residual-Motion-Encoder-for-3D-Human-Motion-Generation)
  - [Introduction](#introduction)
  - [Repository Contents](#repository-contents)
  - [Installation](#installation)
  - [Datasets](#datasets)
    - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
  - [Citations](#citations)
  - [Acknowledgement](#acknowledgement)


## Introduction
In the domain of 3D human motion generation from textual descriptions, the TRME model significantly improves architectural flexibility and dataset diversity. By incorporating an additional layer of residual blocks into the Vector Quantized Variational Autoencoder (VQ-VAE) and utilizing the comprehensive "CHAD" dataset, our model captures finer motion details, improving both the diversity and quality of generated human motions. 

## Repository Contents
- `MDM/` - Clone this directory to download and train the Motion Diffusion Model. For more information, refer to the original github page [MDM](https://github.com/GuyTevet/motion-diffusion-model)
- `T2M/` - Clone this directory to download and train the text to motion model. For more information, refer to the original github page [T2M](https://github.com/EricGuo5513/text-to-motion)
- `TRME/` - Clone this directory to download and train the TRME (Ours) model. For more information on the original VQVAE & GPT architecture, refer to the original github page [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)

## Installation
To set up the necessary environment:

```bash
conda env create -f environment.yml
```

## Datasets

### Data Preparation

The CHAD dataset is an essential component of our project, combining several datasets to provide a diverse array of motion categories for robust model training. The dataset preparation involves several critical steps to ensure the data is ready for use with our Text Residual Motion Encoder (TRME) model.

- Download Required Datasets
Download the additional datasets from the AMASS collection, which are integrated into the HumanML3D dataset to enrich it. The specific datasets to download are:

- `MOYO`
- `MM-FIT`
- `CNRS`
- `DanceDB`
- `GRAB`

These datasets can be found on the AMASS dataset repository [AMASS](https://amass.is.tue.mpg.de/) and should be downloaded and placed in the data directory within your project structure. For additional information, please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D)

## Training
To train the model:


## Evaluation
Evaluate the trained model


## Visualization
Visualize motion predictions from textual descriptions

## Citations:

```
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
```

```
@InProceedings{guo2020action2motion,
  title={Action2motion: Conditioned generation of 3D human motions},
  author={Guo, Chuan and Zuo, Xinxin and Wang, Sen and Zou, Shihao and Sun, Qingyao and Deng, Annan and Gong, Minglun and Cheng, Li},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2021--2029},
  year={2020}
}
```

```
@article{tevet2022human,
  title={Human motion diffusion model},
  author={Tevet, Guy and Raab, Sigal and Gordon, Brian and Shafir, Yonatan and Cohen-Or, Daniel and Bermano, Amit H},
  journal={arXiv preprint arXiv:2209.14916},
  year={2022}
}

```
- <a href="https://github.com/GuyTevet/motion-diffusion-model" target="_blank">MDM: Human Motion Diffusion Model</a> <br/>
- <a href="https://github.com/Mael-zys/T2M-GPT" target="_blank">T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations</a> <br/>
- <a href="https://github.com/EricGuo5513/text-to-motion" target="_blank">Generating Diverse and Natural 3D Human Motions from Text</a><br/>

### Datasets:
- <a href="https://github.com/EricGuo5513/HumanML3D" target="_blank">HumanML3D: 3D Human Motion-Language Dataset</a> <br/>
- <a href="https://amass.is.tue.mpg.de/index.html" target="_blank">AMASS: Archive of Motion Capture As Surface Shapes</a><br/>
- <a href="https://aistdancedb.ongaaccel.jp/" target="_blank">AIST Dance Video Database (DanceDB)</a><br/>
- <a href="https://moyo.is.tue.mpg.de/" target="_blank">MOYO 🧘🏻‍♀️: A dataset containing complex yoga poses, multi-view videos, SMPL-X meshes, pressure and body center of mass</a><br/>
- <a href="https://entrepot.recherche.data.gouv.fr/dataverse/cnrs" target="_blank">CNRS Research Data</a><br/>
- <a href="https://grab.is.tue.mpg.de/" target="_blank">GRAB: A Dataset of Whole-Body Human Grasping of Objects</a><br/>

## Acknowledgement

We appreciate prior contributions as a source of inspiration for Text to 3D Human Motion Generation from :  

* Public code repositories like [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [TM2T](https://github.com/EricGuo5513/TM2T), [MDM](https://github.com/GuyTevet/motion-diffusion-model).
* Public datasets - [AMASS](https://amass.is.tue.mpg.de/)

<a href='https://github.com/YoushanZhang/'>Dr. Youshan Zhang</a> for guidance and valuable feedback.
