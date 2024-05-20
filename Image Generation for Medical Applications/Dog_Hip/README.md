# Image generation for the radiography of dog hip.
The aim of this study is to utilize the diffusion model to augment the original dataset, thereby improving the performance of the Norberg Angle prediction model. The Norberg Angle is illustrated in the figure below. The study involved fine-tuning the DDPM, DreamBooth, and Stable Diffusion models to produce high-quality images. Moreover, a tailored model that amalgamated the EfficientNet and ViT_Gigantic_Patch14_Clip_224 architectures achieved the lowest prediction loss.

<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/3c3fd898-7857-4f2a-88fd-723165ddfb4f" width="450" height="250">

## The workflow of the study
<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/8ce23469-dc6c-4eb3-8fa9-781e1f20cb92" width="550" height="350">


## Image generation from Diffusion models 
You can find the diffusion model weights through the [link](https://yuad-my.sharepoint.com/:f:/g/personal/syueh_mail_yu_edu/EtfXe9VM9rtIoYwemdFNoxoBma16sDeEfTBqY8VSZkXkiA?e=AdV3Jj)

Also, you can find all the generated images [here](https://drive.google.com/drive/folders/1Y_rxgAFNPX2thpiiFbtdiF52q3OT3mbK?usp=drive_link) 

In total, our dataset comprises 1047 real-world images and 1474 generated images. To prevent data leakage and ensure unbiased evaluation, we
meticulously partitioned the dataset into distinct training, testing, and validation sets. Specifically, the training set encompasses Set1, Set2,
Set3, and a subset of Set4, as delineated in the following table. The testing and
validation sets were both split from real-word image Set4. This selection
rationale ensures that the training data encapsulates a diverse
array of visual contexts, fostering robust model learning and adaptation


| Source           | # Images |
|------------------|---------:|
| Set1             |      219 |
| Set2             |       31 |
| Set3             |       84 |
| Set4             |      713 |
| DDPM             |      307 |
| DreamBooth 1     |      467 |
| DreamBooth 2     |      500 |
| Stable Diffusion 1 |     100 |
| Stable Diffusion 2 |     100 |

### Examples of generated images_1:
<img src="https://github.com/YSH-314/AiAI/assets/74528993/236c09c6-2e4d-43b7-af1f-6313e279c3b5" width="750" height="350">

### Examples of generated images_2:
<img src="https://github.com/YSH-314/AiAI/assets/74528993/1daddc41-df10-4945-ae24-269611550c6a" width="650" height="450">

## Inference from diffusion models:
#### Import the packages:
```python

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display
```
#### Load diffusion model weights:
```python
model_path = "your_path_to_model_weight"       
pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to('cuda')
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = torch.Generator(device='cuda')
seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)
```
#### Inference:
```python
prompt = "give_a_prompt" #@param {type:"string"} # ex:x-ray of doghips with black background
negative_prompt = "" #@param {type:"string"}
num_samples = 1 #@param {type:"number"}
guidance_scale = 7.5 #@param {type:"number"}
num_inference_steps = 24 #@param {type:"number"} 
height = 512 #@param {type:"number"}
width = 512 #@param {type:"number"}


# This example makes the model generate 500 images 
i=499
for seed in range(i):

    g_cuda.manual_seed(seed)


    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for img in images:
        #print(type(img))
        img = img.resize((1908,768)) # adjust based on your need
        #img.save(f'path_to_your_folder_filename.png','PNG')
        display(img)
```
## Norberg Angle prediction model:
We published the results through the Symposium on Artificial Intelligence in Veterinary Medicine 2024. And you can find the published results [here](https://www.researchgate.net/publication/380167225_Diffusion_Data_Augmentation_for_Enhancing_Norberg_Hip_Angle_Estimation). These experiments were completed by the notebook through the path `DogHip_predict\Dog_hip_training_v9.ipynb` from the [link](https://yuad-my.sharepoint.com/:f:/g/personal/syueh_mail_yu_edu/EtfXe9VM9rtIoYwemdFNoxoBma16sDeEfTBqY8VSZkXkiA?e=AdV3Jj).



For the customized model experiments, the `tool2.ipynb` provides the data preprocessing and model architectures. `Dog_hip_training_v15.ipynb` provides the whole training process and the prediction results. The best performance model v4 can be found in the `NA_prediction_model_weights` folder through the [link](https://yuad-my.sharepoint.com/:f:/g/personal/syueh_mail_yu_edu/EtfXe9VM9rtIoYwemdFNoxoBma16sDeEfTBqY8VSZkXkiA?e=AdV3Jj)


<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/9c97a166-8dd9-4e97-b086-69de2f1598e2" width="750" height="350">
<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/cc787234-52f5-4308-a889-4782e1085443" width="750" height="350">

