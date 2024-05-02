import pandas as pd
import json
import os
import re
from gensim.corpora.wikicorpus import filter_wiki
from tqdm import tqdm, trange
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm, trange
import argparse  # Import the argparse library

class Csv2Json:
    def chunk_text_by_sentences_and_size_with_overlap(self, text, max_chunk_size, overlap_sentences):
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= max_chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1  
            else:
                chunks.append(" ".join(current_chunk))
                
                if overlap_sentences > 0:
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_length = sum(len(s) + 1 for s in current_chunk) 
                else:
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    def csv2json(self, folder_path, chunk_size=4000):
        if folder_path[-1] == '/':
            folder_path = folder_path[:-1]
        # got last folder name of folder path
        if not os.path.exists('./log'):
            os.makedirs('./log')
        folder_name = os.path.basename(folder_path)
        log_file_path = f'./log/{folder_name}.log'
        #create a log file
        if not os.path.exists(log_file_path):
            open(log_file_path, 'w').close()
        with open(log_file_path, 'r') as log_file:
            processed_files = log_file.read().splitlines()
        csv_list = self.get_csv_list(folder_path)
        print(f"detect {len(csv_list)} csv files, processing dataframe content")
        all_dfs = []
        current_files = []
        file_size = 0
        file_size_threshold = 500*(1024**2)
        file_index = 0
        count = 0
        for file_path in csv_list:
            file_name = os.path.basename(file_path)
            print(f"processing {count}/{len(csv_list)-len(processed_files)}", end='\r')
            if file_name not in processed_files:
                count += 1
                df = pd.read_csv(file_path)
                file_size += os.path.getsize(file_path)
                all_dfs.append(df)
                current_files.append(file_name)
                if file_size > file_size_threshold:
                    print(f"processing {count}/{len(csv_list)}")
                    print(f"generating No.{file_index+1} df, size = {file_size/(1024**2)} MB")
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    combined_df['content'] = combined_df['content'].astype(str)
                    combined_df['content'] = combined_df['content'].apply(self._wiki_replace)
                    self.save_json(combined_df, folder_name)
                    all_dfs = []
                    file_size = 0
                    file_index += 1
                    
                    with open(log_file_path, 'a') as log_file:
                        print("writing log file")
                        for file_name_cur in current_files:
                            log_file.write(file_name_cur+'\n')
                    current_files = []
                    
        if len(all_dfs) > 0:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            self.save_json(combined_df, folder_name)
            with open(log_file_path, 'a') as log_file:
                print("writing log file")
                for file_name_cur in current_files:
                    log_file.write(file_name_cur+'\n')
    def save_json(self, df, folder_name):
        index = self.get_next_index(folder_name)
        print("writing jsonl files")
        with open(f'./jsonl/{folder_name}/{folder_name}_{index}.jsonl', 'w', encoding="utf-8") as f:
            out_list = []
            for i in trange(len(df)):
                if not isinstance(df.iloc[i]['content'], str):
                    continue
                content = self.eliminate_structure(df.iloc[i]['title'],df.iloc[i]['content'])
                content = content
                # if length < 100, skip
                if len(content) < 10:
                    continue
                title = df.iloc[i]['title']
                out_list.append({'title':title,'text': content})
            json.dump(out_list, f, ensure_ascii=False)
    def get_next_index(self, folder_name):
        directory = f'./jsonl/{folder_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
            return 1
        
        files = os.listdir(directory)
        indexes = []
        pattern = re.compile(f'^{folder_name}_([0-9]+)\\.jsonl$')
        
        for file in files:
            match = pattern.match(file)
            if match:
                indexes.append(int(match.group(1)))
        return max(indexes) + 1 if indexes else 1
    def get_csv_list(self, folder_path):
        
        print("loading csv files")
        
        res_csv_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.startswith('._') and file.split('.')[-1] == 'csv':
                    res_csv_list.append(os.path.join(root, file))
        return res_csv_list
    
    
    def _wiki_replace(self, s):
        s = re.sub(':*{\|[\s\S]*?\|}', '', s)
        s = re.sub('<gallery>[\s\S]*?</gallery>', '', s)
        s = re.sub('(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
        s = filter_wiki(s)
        s = re.sub('\* *\n|\'{2,}', '', s)
        s = re.sub('\n+', '\n', s)
        s = re.sub('\n[:;]|\n +', '\n', s)
        s = re.sub('\n==', '\n\n==', s)
        # s = u'【' + d[0] + u'】\n' + s
        return s
    def eliminate_structure(self, article_title, article):
        pattern = r'(={2,3})([^=]+)\1\n(.*?)(?=\n={2,3}|\Z)'
        matches = re.findall(pattern, article, re.DOTALL)
        forbidden_titles =  ['link', 'reference', 'author', 'image', 'see also','personnel','charts','reading','certifications','people','listing']
        res_text = ''
        intro_pattern = r'^.*?(?=\n={2,3})'
        intro_match = re.search(intro_pattern, article, re.DOTALL)
        if intro_match:
            res_text += intro_match.group(0)
        for _, title, content in matches:
            title = title.strip()
            content = content.strip()
            if not any(forbidden_word.lower() in title.lower() for forbidden_word in forbidden_titles):
                res_text += '\n'+ f"Here is information on the {title} of {article_title}."+content
        return res_text

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Convert CSV files to JSON format.')
    parser.add_argument('folder_path', type=str, help='The path to the folder containing CSV files.')
    parser.add_argument('--chunk_size', type=int, default=4000, help='Maximum chunk size for text segmentation.')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Initialize the Csv2Json instance
    csv2json = Csv2Json()
    
    # Call the csv2json method with the folder_path and chunk_size arguments
    csv2json.csv2json(args.folder_path, args.chunk_size)

if __name__ == "__main__":
    main()
