# KatzBot - Enhancing University Community Communication

## Table of Contents

- [Introduction](#introduction)
  - [Project Description](#project-description)
- [Dataset](#dataset)
  - [Pre Processing Data](#pre-processing-data)
  - [Sentence Completion Pairs](#sentence-completion-pairs)
  - [QA Pairs](#qa-pairs)
- [Method](#method)
  - [Data Preparation](#data-preparation)
  - [Tokenization](#tokenization)
  - [Dataset Initialization](#dataset-initialization)
- [Results](#results)
- [Citations](#citations)

## Introduction

Enhancing communication within university communities is imperative to meet the diverse needs of stakeholders, including prospective and current students, alumni, and other information seekers. Current academic chatbot systems often exhibit imprecise communication, necessitating tailored solutions. To address this, we propose KatzBot, leveraging Katz generative pre-trained transformer (KatzGPT), a sophisticated custom large language model (LLM), to effectively bridge precision gaps in existing chatbot systems and academic universities. Moreover, we have curated two datasets: 6,280 sentence-completion pairs and 7,330 question-answer pairs. We train KatzGPT on the sentence-completion pairs to expand its knowledge base, followed by training on the question-answer pairs to enhance its accuracy. Through extensive experimentation, KatzGPT demonstrates superior performance compared to state-of-the-art methods on our dataset. In addition, our KatzBot system enables a concise and effective interface to enhance the communication between users and the KatzGPT model.

## Dataset

### Pre Processing Data

We preprocessed the data using regex and custom parsing techniques, standardizing sentences and tokenizing them. We removed duplicates and noise, organizing the dataset into sentence and question-answer pairs. Finally, we stored the data in CSV and JSON formats, ensuring compliance with privacy laws and creating a clean, structured dataset for chatbot development.

### Sentence Completion Pairs

We automated text segmentation into related sentence pairs, each column containing one sentence. Scripts were used to detect and remove nonsensical text and unwanted characters, followed by a manual review for accuracy. This effort yielded a dataset of over 6,400 related sentence pairs.

#### Sentence Completion Dataset Sample

| Sentence 1                                                      | Sentence 2                                                                                       |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Accommodation is allocated on a first-come, first-served basis. | Alternatively, you may consider living in New Jersey near bus routes that provide an easy commute to campus. |
| It is prohibited to work without first obtaining CPT authorization. | Please be patient during the processing time for OPT applications, as they cannot be expedited. |

### QA Pairs

We employed an automated script to process the text file, generating numerous question-and-answer pairs tailored for the chatbot. Additionally, we manually crafted additional pairs addressing common queries from new students. This effort resulted in a dataset of 7,334 question-and-answer pairs, enhancing quality and coverage.

#### QA Dataset Sample

| Question                                                                                    | Answer                                                                                                                                                                              |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| What criteria are considered for successful Honors applicants?                              | Offers of admission are contingent on a student submitting an Intent to Enroll Form by the May 1st deadline. Enrollment in another university will be considered a forfeiture of Honors admission and scholarship. |
| What is the critical deadline students must meet to secure their admission and scholarship? | Offers of admission are contingent on a student submitting an Intent to Enroll Form by the May 1st deadline. Enrollment in another university will be considered a forfeiture of Honors admission and scholarship. |
| What are the average criteria for the current Honors cohort in terms of GPA, SAT, and ACT scores? | The current Honors cohort averages a GPA of 94, SAT score of 1460, and ACT score of 32.                                                                                              |

## Method

### Data Preparation

The dataset comprises question-answer pairs from academic contexts, such as university-related inquiries and responses. Each data item consists of a question and its corresponding answer.

### Tokenization

The data is tokenized using the AutoTokenizer from the Hugging Face Transformers library. The tokenizer encodes each question and answer text into token indices, ensuring compatibility with the model architecture.

### Dataset Initialization

The tokenized data is organized into a custom PyTorch Dataset subclass, named QADataset. This dataset prepares input-output pairs for training the CLM. Each data item includes input token IDs and corresponding label token IDs, representing the next token prediction task.

## Results

In our comparative analysis detailed in the table below, we assess the performance of several Large Language Models (LLMs), highlighting our in-house developed KatzGPT model. This evaluation benchmarks the Rouge Scores, focusing particularly on Rouge-L, which measures the long-form coherence of generated texts.

### Comparison of Rouge F-Scores for LLMs (sorted by Rouge L)

| Model                | Rouge-1 | Rouge-2 | Rouge-L |
|----------------------|---------|---------|---------|
| KatzGPT (Deepa)      | 0.17    | 0.028   | 0.15    |
| KatzGPT (Haider)     | 0.05    | 0.39    | 0.10    |
| KatzGPT (Shashidhar) | 0.24    | 0.10    | 0.20    |
| Llama2 3B            | 0.23    | 0.07    | 0.20    |
| Microsoft Phi 1.5    | 0.26    | 0.10    | 0.24    |
| Llama2 7B            | 0.28    | 0.12    | 0.25    |
| KatzGPT (Sahil)      | 0.27    | 0.15    | 0.25    |
| Microsoft Phi2       | 0.34    | 0.15    | 0.31    |
| Mistral 7B Instruct  | 0.43    | 0.20    | 0.33    |
| GPT-2                | 0.45    | 0.32    | 0.43    |

### LLMs Predictions Comparison

| User Input                                                          | GPT-2                                                                                         | Mistral 7B                                                                                                                                         | Phi 2                                                                                                                       | Llama2 7B                                                                                                                  |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| What career services does the university provide to its students?   | The university offers services to its student body.                                           | The university provides career services that include personalized counseling, career planning, and access to professional networks.               | The university provides career services to help students find employment after graduation. These services include resume building, interview preparation, job search strategies, and networking opportunities. | Access to professional mentors and career coaches. Opportunities for internships, STEM OPT, and career opportunities through the Career Center. |
| Can you share information on internship opportunities for students? | Internships are primarily during the summer, but you can also consider internships in New York, the financial capital of the world, or on an individual basis. | Yes, there are numerous internship opportunities available for students, both in the U.S. and in Israel, through the Yeshiva University Israel Center. | Yes, KatzBot can provide information on internship opportunities for students. What department are you interested in?            | The program offers internship opportunities that give students industry experience and valuable career skills.                 |
| What are the basic admission requirements for this university for Data Analytics? | The basic admission rules for this college program include a minimum GPA of 3.2, an SAT score of 1460 or an ACT score of 32, and approval from their high school administration. | The basic admission requirements for this university for Data Analytics include a bachelors degree with a strong mathematical and programming background. | The basic admission requirements include a bachelors degree from an accredited college or university, a minimum GPA of 3.0, and a minimum grade of Bplus in the prerequisite courses. | Bachelors degree, one year of related work experience, minimum grade point average of 3.0 on a 4.0 scale, or minimum grade of B in prerequisite courses. |

## Citations

- Nur Syahirah Ahmad, Mohd Hamim Sanusi, Mohd Helmy Abd Wahab, Aida Mustapha, Zainal Abidin Sayadi, and Mohd Zainuri Saringat. "Conversational bot for pharmacy: a natural language approach." In 2018 IEEE conference on open systems (ICOS), pp. 76–79. IEEE, 2018.
- V. Arulmangainayaki, M. Harini, M. Keerthiga, and M. Deva Priya. "Intelligent chatbot for medical assistance in rural areas." International Journal of Innovative Technology and Exploring Engineering, 9, 2020.
- R. G. Athreya, A. C. Ngonga Ngomo, and R. Usbeck. "Enhancing community interactions with data-driven chatbots-the dbpedia chatbot." 2018.
- Ayah Atiyah, Shaidah Jusoh, and Firas Alghanim. Evaluation of the naturalness of chatbot applications,2019.
- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020b.
- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms, 2023.
- A. Jiao. An intelligent chatbot system based on entity extraction using rasa nlu and neural network.jphcs, 1487(1), 2020.


