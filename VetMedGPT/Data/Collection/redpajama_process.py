import gzip
import json
import os
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from multiprocessing import Pool, set_start_method
import time
from nltk.corpus import stopwords
from tqdm import tqdm
from multiprocessing import Process
# Function to read multiple JSON objects from a single file
def read_multiple_json(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        while True:
            try:
                # Attempt to read the next JSON object
                json_obj = json.loads(next(file))
                data.append(json_obj)
            except StopIteration:
                # End of file reached
                break
            except json.JSONDecodeError:
                # Handle possible decoding errors (e.g., malformed JSON)
                continue
    return data



def cal_max_threshold(record):
    max_threshold = 0
    for item in record:
        if len(item['thresholds']) == 0:
            continue
        tmp_max_threshold = max(item['thresholds'])
        if tmp_max_threshold > max_threshold:
            max_threshold = tmp_max_threshold
    return max_threshold
# def make_output_folder(out_folder):
#     folder_names = ['15','20','25','30']
#     for folder_name in folder_names:
#         folder_path = os.path.join(out_folder,folder_name)
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
            
def process_file(file_list,output_path):
    for file_path in file_list:
        file_name = file_path.split('/')[-1].replace('.json.gz','')
        data = read_multiple_json(file_path)
        json_file_path = file_path.replace('.json.gz','.json')
        with open (json_file_path,'r') as f:
            result = json.load(f)
        if len(result) != len(data):
            print(f'error {file_path}')
            continue
        thresholds_list = []
        for idx,content in enumerate(data):
            record = result[idx]
            max_threshold = cal_max_threshold(record)
            
            thresholds_list.append({'max_threshold':max_threshold, 'idx':idx})
        _15 = []
        _20 = []
        _25 = []
        _30 = []
        for item in thresholds_list:
            if item['max_threshold'] > 0.3:
                _30.append({'idx':item['idx'],'max_threshold':item['max_threshold'],'text':data[item['idx']]['raw_content']})
            elif item['max_threshold'] > 0.25:
                _25.append({'idx':item['idx'],'max_threshold':item['max_threshold'],'text':data[item['idx']]['raw_content']})
            elif item['max_threshold'] > 0.2:
                _20.append({'idx':item['idx'],'max_threshold':item['max_threshold'],'text':data[item['idx']]['raw_content']})
            elif item['max_threshold'] > 0.15:
                _15.append({'idx':item['idx'],'max_threshold':item['max_threshold'],'text':data[item['idx']]['raw_content']})
        out_put_folder_path = os.path.join(output_path, file_path.split('/')[-2].replace('.json.gz',''))
        if not os.path.exists(out_put_folder_path):
            os.makedirs(out_put_folder_path)
        output_group = [_15,_20,_25,_30]
        output_number = ['15','20','25','30']
        for idx, number in enumerate(output_group):
            with open(os.path.join(out_put_folder_path,f'{file_name}_{output_number[idx]}.json'),'w') as f:
                json.dump(number,f)
        print(f'finish {file_path}')
        del data
        del result
# for idx,content in tqdm(enumerate(data)):
#     record = result[idx]
#     max_threshold = cal_max_threshold(record)
#     thresholds_list.append({'max_threshold':max_threshold, 'idx':idx})

        
def process_total(target_path,out_path):
    file_list = []
    
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.endswith('.json.gz'):
                file_list.append(os.path.join(root, file))
    # remove_list = []
    # for idx, file_path in enumerate(file_list):
    #     out_file_path = file_path.replace('.json.gz', '.json')
    #     if os.path.exists(out_file_path):
    #         remove_list.append(idx)
    # file_list = [file_list[i] for i in range(len(file_list)) if i not in remove_list]
    print(f'Total {len(file_list)} files')
    
    
    num_processes = 32
    chunked_filenames = np.array_split(file_list, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=process_file, args=(list(subset),out_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
if __name__ == '__main__':
    set_start_method('spawn')  # Set the start method for multiprocessing
    # Filter out all future warnings and specific warnings
    target_path = "/scratch/vetgpt/redpajama/documents/2023-14"
    out_path = "/scratch/vetgpt/redpajama/data_process/output_parent"
    process_total(target_path, out_path)