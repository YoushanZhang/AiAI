# Text Residual Motion Encoder for 3D Human Motion Generation

![teaser_image](https://github.com/YoushanZhang/AiAI/blob/main/3D%20Human%20Motion/motions.png)

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
```

## Data Preparation
CHAD dataset is a combination of several datasets, providing diverse motion categories suitable for robust training:


## Training
To train the model:


## Evaluation
Evaluate the trained model


## Visualization
Visualize motion predictions from textual descriptions

## Citations/References:

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
* <a href='https://github.com/YoushanZhang/'>Dr. Youshan Zhang</a> for guidance and valuable feedback.





