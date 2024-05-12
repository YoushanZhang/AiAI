# VetMedGPT
VetMedGPT is a specialized tool developed to assist in the initial diagnosis and first aid for animals, aiming to bridge the gap in the field of artificial intelligence (AI) by providing tailored support for veterinary medicine healthcare.

- You can access the pretrain Model [VetMedGPT 1B](https://huggingface.co/azsxscdvfb/VetMedGPT-1B-V0.2/tree/main)
- And the finetuned version [VetMedGPT 1B chat](https://huggingface.co/azsxscdvfb/VetMedGPT-1B-chat-V0.2/tree/main) 
- Explore the Website and Prototype [website](https://1b6c-2601-8c-4e00-a930-f565-9d0a-bef4-603f.ngrok-free.app)
![UI](https://github.com/YoushanZhang/AiAI/blob/main/VetMedGPT/img/Web%20page%20UI.jpeg)
## Project Overview
In the realm of AI, significant advancements have been achieved in developing large language models (LLMs) catering to various domains, including human healthcare. However, the domain of veterinary science has often been overlooked, resulting in limitations in scope, efficacy, or availability of existing AI models for animal healthcare.

To address this gap, our project focuses on:

- **Dataset Collection**: We collected a novel veterinary medicine dataset named VetMed, comprising 500GB of training data sourced from online public sources include Wikipedia, redpajama and S2ORC. This dataset includes about one million question-answer and multiple-choice-question-answer pairs.
![task](https://github.com/YoushanZhang/AiAI/blob/main/VetMedGPT/img/Dataset.jpeg)
- **Model Development**: We developed a novel generative pre-trained transformer, VetMedGPT, specifically tailored for veterinary medicine healthcare. This model is fine-tuned and tested on the VetMed dataset to enhance its effectiveness in animal health diagnosis and care.
![Workflow](https://github.com/YoushanZhang/AiAI/blob/main/VetMedGPT/img/Workflow.jpeg)
- **Training Arguments**
 
| Setting                         | Description                                                   |
|---------------------------------|----------------------------------------------------------------|
| Parameters                      | 1B                                                           |
| Attention Variant               | Grouped Query Attention                                        |
| Model Size                      | Layers: 16, Heads: 16, Query Groups: 4, Embedding Size: 2048, Intermediate Size (Swiglu): 7168 |
| Sequence Length                 | 2048                                                           |
| Batch Size                      | 2 million tokens (2048 * 1024)                                             |
| Learning Rate                   | 4e-4                                                           |
| Learning Rate Schedule          | Cosine with 2000 warmup steps|
| Combined Dataset Size           | Around 120B tokens                                              |
| Total Tokens During Training    | 120B                                          |
| Hardware                        | 6 A100-80G GPUs                                              |
- **Finetune Arguments**

| Setting                         | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| per_device_train_batch_size     | 48                                                           |
| max_new_tokens                  | 512                                                            |
| source_max_len                  | 512                                                            |
| target_max_len                  | 1024                                                           |
| learning_rate                   | 8e-5                                                           |
| adam_beta2                      | 0.999                                                          |
| max_grad_norm                   | 1.0                                                            |
| weight_decay                    | 0.0                                                            |
|epoch                            |3                                                               |
| Hardware                        | 8 A100-80G GPUs                                              |
## Results


| **model**       | **ROUGE-1 r** | **ROUGE-1 p** | **ROUGE-1 f** | **ROUGE-2 r** | **ROUGE-2 p** | **ROUGE-2 f** | **ROUGE-L r** | **ROUGE-L p** | **ROUGE-L f** |
|-----------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Llama2 7B       | 0.3520        | 0.1452        | 0.1788        | 0.1397        | 0.0586        | 0.0688        | 0.3244        | 0.1318        | 0.1630        |
| Mistral 7B      | 0.3664        | 0.0894        | 0.1370        | 0.1078        | 0.0214        | 0.0341        | 0.3360        | 0.0804        | 0.1236        |
| VetMedGPT 1B    | 0.2677        | 0.0669        | 0.1034        | 0.0342        | 0.0066        | 0.0107        | 0.2390        | 0.0588        | 0.0912        |
| Falcon 7B       | 0.3495        | 0.1097        | 0.1568        | 0.1633        | 0.0434        | 0.0646        | 0.3265        | 0.1015        | 0.1454        |
| tinyllama 1B    | 0.2088        | 0.0678        | 0.0967        | 0.0424        | 0.0097        | 0.0148        | 0.1909        | 0.0608        | 0.0870        |

Through table, by using the ROUGE metric (Recall, Precision, and F-score), we can understand the degree to which the models' answers match the reference answer across different granularities of text (ROUGE-1 for single strings, ROUGE-2 for larger strings, and ROUGE-L for the longest common subsequence). Compared to other models, TinyLlama 1B scores lower on all ROUGE metrics, indicating that its answers are less similar to the reference texts. This may reflect its general training background or smaller parameter size, which impacts its ability to capture the subtle differences in veterinary queries. VetMedGPT scores moderately on all ROUGE metrics, showing a slight improvement over TinyLlama 1B.

## Benefits
- **Enhanced Accessibility**: VetMedGPT aims to make veterinary care more accessible to pet owners, particularly in situations where immediate clinical support is unavailable.
- **Resource for Veterinary Professionals**: The tool also serves as a valuable resource for veterinary professionals, offering support and insights to address challenges posed by current limitations in AI-supported veterinary medicine
## Applications
- **Virtual Veterinary Assistant App**: A web app for pet owners to input symptoms and receive initial diagnoses and first aid recommendations for their pets.
- **Veterinary Clinic Support Tool**: Integration into veterinary clinics' systems to assist veterinarians in diagnosing and treating animal patients, providing additional insights and reference materials.
- **Educational Resource for Veterinary Students**: An online platform for veterinary students to access educational materials and practice scenarios, serving as a virtual tutor for veterinary medicine studies.
## Citations
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). Attention is all you need. arXiv preprint arXiv:1706.03762.     
Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., ... Li, S. (2023). PyTorch FSDP: Experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277.      
Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691.      
Zhang, P., Zeng, G., Wang, T., & Lu, W. (2024). TinyLlama: An open-source small language model. arXiv preprint arXiv:2401.02385.     
