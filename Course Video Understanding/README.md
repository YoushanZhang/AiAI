# SparrowVQE


<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650c7fbb8ffe1f53bdbe1aec/DTjDSq2yG-5Cqnk6giPFq.jpeg" width="40%" height="auto"/>
</p>


<div class="center-div", align="center">
    <table width="100%" height="auto">
        <tr>
            <td align="center">
                <a href="https://colab.research.google.com/github/rrymn/SparrowVQE/blob/main/SparrowVQE_Demo.ipynb">[Google Colab Demo]</a>
                <a href="https://huggingface.co/spaces/rrymn/SparrowVQE">[🤗 HuggingFace Demo]</a>
            </td>
        </tr>
    </table>
</div>




<p align='center', style='font-size: 16px;' >A Custom 3B parameter Model Enhanced for Educational Contexts: This specialized model integrates slide-text pairs from machine learning classes, leveraging a unique training approach. It connects a frozen pre-trained vision encoder (SigLip) with a frozen language model (Phi-2) through an innovative projector. The model employs attention mechanisms and language modeling loss to deeply understand and generate educational content, specifically tailored to the context of machine learning education. </p>

## How to use


**Install dependencies**
```bash
pip install transformers 
pip install -q pillow accelerate einops
```


```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

torch.set_default_device("cuda")

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "rrymn/SparrowVQE", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("rrymn/SparrowVQE", trust_remote_code=True)

#function to generate the answer
def predict(question, image_path):
    #Set inputs
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question}? ASSISTANT:"
    image = Image.open(image_path)
    
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to('cuda')
    image_tensor = model.image_preprocess(image)
    
    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=25,
        images=image_tensor,
        use_cache=True)[0]
    
    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

```
## Week 3-8 Slides Summary
![Week 3 to Week 8 Machine Learning Concepts](/images/example_01.drawio.png)
*This image provides a visual summary of key machine learning concepts covered between weeks 3 to 8. It includes topics such as avoiding overfitting, understanding logistic functions in the context of probabilities, exploring the 'face space' in image recognition, analyzing the curse of dimensionality, PCA as matrix factorization, and Gaussian Mixture Models.*

## Week 10-15 Slides Summary
![Week 10 to Week 15 Machine Learning Concepts](/images/example_02.drawio.png)
*This image provides a visual summary of key machine learning concepts covered between weeks 10 to 15. It illustrates topics such as the differences in model spaces between decision trees and nearest neighbors, understanding margins in SVMs, the role of Vision Transformers and MLP heads in neural networks, the effect of bagging on model variance, and an introduction to entropy in the context of decision trees.*

