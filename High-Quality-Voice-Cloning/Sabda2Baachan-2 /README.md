## This Readme is for the Custom-Model-2: High-Quality Voice Cloning

This Readme provides instructions for running the Custom-Model-2 project for high-quality voice cloning.

## Prerequisites

**Python Version 3.10 or above, recommended Python Version 3.10**

**In Colab Environment you can simply run this without any issue. (you also do not need any virtual environment in Colab)**

**To run this model/project, first start with cloning the entire High-Quality-Voice-Cloning**

### Step 1: Clone this repository

### Step 2: Navigate to the Project Directory

`cd ./Custom-Model-2/`

### Step 3: Create a new virtual environment (Optional)and run the following dependencies.

**For example**

```
python -m venv voice_clone
call voice_clone/bin/activate
```

### Step 4: Install Dependencies

`pip install `

### Step 5: You might have to install some other dependencies or libraries because the _pip install ._ might not include all the libraries needed to run this repository.

**Additional libraries to install:**

```
pip install accelerate

pip install diffusers

pip install fairseq

pip install audiolm_pytorch

import torch

import gc

import os

```

### Step 6: Download Model Checkpoints(Optional)

**For training / fine-tuning the model from the provided model checkpoints**
If you plan to train/fine-tune from scratch, download the Suno/Bark model checkpoints for the semantic, coarse, and fine models from Hugging Face Hub.

[Hugging Face Model Weights Link](https://huggingface.co/suno/bark/tree/main)

These models are also downloaded when you run the Jupyter notebooks. You can downlaod them if you want to download either the small or the large model manually.

### Step 7: Fine-tuning on a Custom Dataset

1. Open the _train_semantic.ipynb_ :

- Change _dataset_path_ and _projectdir_path_ to your desired dataset and logging directory.
- Adjust training/validation epochs and total training epochs.
- Set the output folder to save your model weights (".bin" format).

2. Repeat for _train_coarse.ipynb_ and _train_fine.ipynb_ :
   Follow similar steps as in Step 7: (1) for dataset paths, logging directories, epochs, and output folders.

### Step 10: Generating Output (Inference Step)

- After training the three models, use the _".bin"_ weights from the google drive link below and first make some changes to the _test_models.ipynb_ notebook.
- The notebook sets trained model weight after **Fine-Tuning** for all three models.

### Step 11: Using trained model Weights

1. Download Model Weights:

- Use the provided Google Drive link to download the pre-trained model weights generated from the original 500 audio clip training session:

**Important Note**

- This folder contains the output weights for all three models in the Capstone Datasets(Input and Output)
  [Google Drive Link for Dataset all Input as well as test output including tokens and raw audio as well as generated audio with text prompts.](https://drive.google.com/drive/folders/1FTqapXz9Z1kqtOPNUKaHX5GVVw0e7_8y?usp=sharing)

2. Follow the instructions in the _test_models.ipynb_ notebook. I have already loaded an example for the dataset and how to load the model weights.

- Make sure to set the proper dataset path for the _voice_name_ parameter as it loads the tokens which is inside the Input Folder in the Google Drive.

### Generating Inference Audio

- Run the _test_models.ipynb_ after making the necessary changes as entioned above.

**Additional Notes**

- Ensure you have the necessary libraries installed before running the notebooks.
- Adjust hyperparameters in the notebooks as needed for your specific dataset and desired outcomes.

**Enjoy generating high-quality voice with Custom-Model-2!**
