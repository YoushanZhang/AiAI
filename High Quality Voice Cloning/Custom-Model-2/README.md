This Readme is for the Custom-Model-2.

To run this model/project, first start with cloning the entire High-Quality-Voice-Cloning

Step 1: Clone the repository
Step 2: cd ./Custom-Model-2/
Step 3: create a new virtual environment and run the following dependencies.

To run this model you might need Python Version 3.10 or above, recommended Python Version 3.10
In Colab Environment you can simply run this without any issue. (you also do not need any virtual environment in Colab)

Step 4: pip install .
Step 5: You might have to install some other dependencies or libraries because the pip install . might not include all the libraries needed to run this repository.

# additional libraries to install:

--> pip install accelerate
--> pip install diffusers
--> pip install fairseq
--> pip install audiolm_pytorch
--> import torch
--> import gc
--> import os

Step 6: For training / fine-tuning the model from the provided model checkpoints,
link to large and small Suno/Bark Checkpoints for training the semantic, coarse, and fine models,
https://huggingface.co/suno/bark/tree/main

These models are downloaded when you run the Jupyter notebooks below, in case you need to download them separately, I have provided the link above from Hugging Face Hub.

Step 7: To fine-tune on a custom dataset first --> Open the train_semantic.ipynb notebook and change the dataset_path and projectdir_path to your desired dataset and logging directory respectively.
Furthermore, you can make adjustments to the train and validation epoch size, and training epochs in the notebook.
Also, set the output folder destination to save your model's weight which is in the ".bin" format.

Step 8: To fine-tune the train_coarse.ipynb notebook, follow similar steps as Step 7 for dataset path and logging directory path.
Make similar adjustments to epochs and output folder.

Step 9: To fine-tune the train_fine.ipynb notebook, Follow a similar step as in Step 7 or Step 8.

Step 10: To generate the output after training or fine-tuning the three different models/notebooks in this repository, on your custom dataset,
Use the model weight in ".bin" created by training the 3 notebooks in Steps 7, 8, and 9 in the test_models.ipynb notebook
There is a declaration for the output for all three models.

## Important

Step 11: To use the fine-tuned model weights generated from the original training of the 500 audio clips,
I have provided the model weights in the following Google Drive link:
please use the Model Weights provided in this link to generate the output based on this training:

https://drive.google.com/drive/folders/128vBHjuYdsO0lpya34vHpbZM4cYFsyu4?usp=sharing

This drive contains the output weight for all 3 models required to generate the inference output.

Follow the instructions in the test_models.ipynb notebook.

For the Voice_name argument, you need the tokens if you are using the above-provided weight, which can be found in the following Google Drive link:

Link to tokens for voice name:
https://drive.google.com/drive/folders/1-1mrvO-yofh8txIOFXNh0slFCxv0Y6rc?usp=sharing

Step 12: You can generate the inference audio.
Enjoy!!
