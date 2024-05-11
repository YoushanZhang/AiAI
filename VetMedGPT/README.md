# VetMedGPT
VetMedGPT is a specialized tool developed to assist in the initial diagnosis and first aid for animals, aiming to bridge the gap in the field of artificial intelligence (AI) by providing tailored support for veterinary medicine healthcare.

- You can access the Model [Model](https://huggingface.co/azsxscdvfb/vetmedgpt-1B-V0.1)    
- Explore the Website and Prototype [website](https://eb99-2601-8c-4e00-a930-7918-efc0-75c9-fdab.ngrok-free.app/)
![UI](https://github.com/YoushanZhang/AiAI/blob/main/VetMedGPT/img/Web%20page%20UI.jpeg)
## Project Overview
In the realm of AI, significant advancements have been achieved in developing large language models (LLMs) catering to various domains, including human healthcare. However, the domain of veterinary science has often been overlooked, resulting in limitations in scope, efficacy, or availability of existing AI models for animal healthcare.

To address this gap, our project focuses on:

- **Dataset Collection**: We collected a novel veterinary medicine dataset named VetMed, comprising 500GB of training data sourced from reputable sources such as Wikipedia and ArXiv. This dataset includes about one million question-answer and multiple-choice-question-answer pairs.
![task](https://github.com/YoushanZhang/AiAI/blob/main/VetMedGPT/img/Dataset.jpeg)
- **Model Development**: We developed a novel generative pre-trained transformer, VetMedGPT, specifically tailored for veterinary medicine healthcare. This model is fine-tuned and tested on the VetMed dataset to enhance its effectiveness in animal health diagnosis and care.
![Workflow](https://github.com/YoushanZhang/AiAI/blob/main/VetMedGPT/img/Workflow.jpeg)
## Results
The Mistral 7B Instruct model attained the highest accuracy in answering veterinary science multiple-choice questions with a score of 0.4391, followed by Llama2 7B chat at 0.3173, highlighting their superior performance over Tinyllama-based models. Regarding text similarity, VetMedGPT showed a moderate improvement over TinyLlama 1B across ROUGE metrics, indicating its proficiency in generating responses closely aligned with reference texts.

| Model             | Rouge-1 |      |      | Rouge-2 |      |      | Rouge-3 |      |      |
|-------------------|---------|------|------|---------|------|------|---------|------|------|
|                   | r       | p    | f    | r       | p    | f    | r       | p    | f    |
| tinyllama 1B      | 0.3684  | 0.067| 0.114| 0.073   | 0.010| 0.018| 0.361   | 0.067| 0.111|
| VetMedGPT 1B      | 0.391   | 0.073| 0.122| 0.093   | 0.012| 0.021| 0.356   | 0.066| 0.110|
| llama2 7b chat    | 0.473   | 0.105| 0.170| 0.149   | 0.023| 0.040| 0.431   | 0.095| 0.154|
| Falcon 7B         | 0.360   | 0.117| 0.174| 0.106   | 0.030| 0.046| 0.325   | 0.106| 0.157|
| Mistral 7B Instruct-V0.2 | 0.491 | 0.095 | 0.158 | 0.143 | 0.020 | 0.035 | 0.451 | 0.087 | 0.145 |



 
## Benefits
- **Enhanced Accessibility**: VetMedGPT aims to make veterinary care more accessible to pet owners, particularly in situations where immediate clinical support is unavailable.
- **Resource for Veterinary Professionals**: The tool also serves as a valuable resource for veterinary professionals, offering support and insights to address challenges posed by current limitations in AI-supported veterinary medicine
## Applications
- **Virtual Veterinary Assistant App**: A web app for pet owners to input symptoms and receive initial diagnoses and first aid recommendations for their pets.
- **Veterinary Clinic Support Tool**: Integration into veterinary clinics' systems to assist veterinarians in diagnosing and treating animal patients, providing additional insights and reference materials.
- **Educational Resource for Veterinary Students**: An online platform for veterinary students to access educational materials and practice scenarios, serving as a virtual tutor for veterinary medicine studies.
## Citations
