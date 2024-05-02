import warnings
from multiprocessing import Pool, set_start_method
import json
import os
from tqdm import tqdm
from pprint import pprint
from Questgen import main
import torch
# Define the function to generate question-answer pairs
def generate_qa_pairs(paragraph, qg, answer_module):
    # Generate Questions
    payload = {'input_text': paragraph, 'max_questions': 3}
    output = qg.predict_shortq(payload)
    questions = [q.get('Question', '') for q in output.get('questions', [])]

    # Generate subjective answers
    payload['input_question'] = questions
    subjective_answers = answer_module.predict_answer(payload)

    return questions, subjective_answers

# Read data from the JSON file and generate question-answer pairs with a limit of 2000
def load_data(folder_path):
    seen_paragraphs = set()
    unique_paragraphs = []
    file_list = os.listdir(folder_path)
    repeat_count = 0
    bar = tqdm(file_list, desc=f'Loading data from {folder_path}')
    for file_name in bar:
        if file_name.endswith('.json'):
            try:
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for record in data:
                        if record['paragraph'] not in seen_paragraphs:
                            seen_paragraphs.add(record['paragraph'])
                            unique_paragraphs.append(record)
                        else:
                            repeat_count += 1
            except:
                continue
    print(f"Number of repeated paragraphs: {repeat_count}")
    return unique_paragraphs
    
def process_data(folder_path, qg, answer_module):
    data = load_data(folder_path)
    qa_pairs = []
    total_qa_pairs = 0
    for item in tqdm(data, desc='Generating question-answer pairs'):
        paragraph = item['paragraph']
        questions, subjective_answers = generate_qa_pairs(paragraph, qg, answer_module)
        num_qa_pairs = len(questions)
        qa_pairs.append({
            'paragraph': paragraph, 
            'questions': questions, 
            'subjective_answers': subjective_answers, 
            'num_qa_pairs': num_qa_pairs
        })
        total_qa_pairs += num_qa_pairs

    return qa_pairs

def process_folder(args):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", message="The `pad_to_max_length` argument is deprecated")
    warnings.filterwarnings("ignore", message="This sequence already has </s>")
    
    torch.cuda.is_available()
    qg = main.QGen(device="cuda:1")
    answer_module = main.AnswerPredictor(device="cuda:1")

    folder_path, output_path = args
    
    results = process_data(folder_path, qg, answer_module)
    output_file = os.path.join(output_path, os.path.basename(folder_path) + ".json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Saved {len(results)} question-answer pairs to {output_file}")

def process_total(target_path, output_path):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", message="The `pad_to_max_length` argument is deprecated")
    warnings.filterwarnings("ignore", message="This sequence already has </s>")
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folder_list = [(os.path.join(target_path, folder), output_path) for folder in os.listdir(target_path)]
    # print(folder_list)
    with Pool(processes=8) as pool:
        pool.map(process_folder, folder_list)

if __name__ == '__main__':
    set_start_method('spawn')  # Set the start method for multiprocessing
    # Filter out all future warnings and specific warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", message="The `pad_to_max_length` argument is deprecated")
    warnings.filterwarnings("ignore", message="This sequence already has </s>")
    target_path = "/scratch/vetgpt/data/extract_para"
    output_path = "/scratch/vetgpt/data/qa_pairs"
    process_total(target_path, output_path)
