import os
import re
import json
import pandas as pd
from multiprocessing import Pool, cpu_count
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors

def get_csv_path(folder_path):
    # 改为单个进程执行的函数
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def extract_paragraphs_with_animals(text, animals):
    if not isinstance(text, str):
        text = str(text)
    paragraphs = text.split('\n')
    animal_paragraphs = {animal: [] for animal in animals}
    patterns = {animal: re.compile(r'\b' + re.escape(animal) + r'\b', re.IGNORECASE) for animal in animals}
    for paragraph in paragraphs:
        for animal, pattern in patterns.items():
            if pattern.search(paragraph):
                animal_paragraphs[animal].append(paragraph)
    animal_paragraphs = {animal: paragraphs for animal, paragraphs in animal_paragraphs.items() if paragraphs}
    return animal_paragraphs

def judge_similar_words(text, specific_word, model):
    words = word_tokenize(text)
    words = [word for word in words if word in model.key_to_index]
    similarities = {word: model.similarity(specific_word, word) for word in words}
    return sum(1 for sim in similarities.values() if sim > 0.3) > 2

def get_csv_paragraphs(args):
    csv_file_path, animals, save_path,  animal_list, specific_word = args
    model_path = "/scratch/vetgpt/temp/word2vec/git/word2vec-google-news-300"
    model = KeyedVectors.load_word2vec_format(model_path, binary=True) 
    print(f"Processing {csv_file_path}")
    animal_dict = {animal: [] for animal in animal_list}
    df = pd.read_csv(csv_file_path)
    for i in range(len(df)):
        content = df.iloc[i]['content']
        extracted = extract_paragraphs_with_animals(content, animals)
        for animal, animal_paragraphs in extracted.items():
            for paragraph in animal_paragraphs:
                if judge_similar_words(paragraph, specific_word, model):
                    animal_dict[animal].append(paragraph)
    save_paragraphs(animal_dict, save_path, os.path.basename(csv_file_path))

def save_paragraphs(paragraphs, save_path, file_name):
    json_file_name = file_name.replace('.csv', '.json')
    for animal, animal_paragraphs in paragraphs.items():
        with open(os.path.join(save_path, animal, json_file_name), 'w') as f:
            json.dump([{'paragraph': p} for p in animal_paragraphs], f)

def create_folder(folder_path, animal_list):
    for animal in animal_list:
        os.makedirs(os.path.join(folder_path, animal), exist_ok=True)

def main(folder_path, animals, save_path, animal_list, specific_word):
    create_folder(save_path, animal_list)
    csv_files = get_csv_path(folder_path)
    pool_size = 96
    with Pool(pool_size) as pool:
        pool.map(get_csv_paragraphs, [(csv_file, animals, save_path, animal_list, specific_word) for csv_file in csv_files])

if __name__ == "__main__":
    
    animal_list = ['dog','cat','goat','bird','donkey',"amphibian","camel","cattle","chicken","duck","fish","horse","pig","rabbit","rat","sheep","turkey","turtle","llama","alpaca","ferret","gerbil","hamster","hedgehog","mouse","parrot","cow"]
    main("/scratch/vetgpt/data/wikipedia_data/csv", animal_list, "/scratch/vetgpt/data/extract_para", animal_list, 'disease')
