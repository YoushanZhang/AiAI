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
import re
def get_file_length(file_path):
    with gzip.open(file_path, 'r') as f:
        for i, l in enumerate(f):
            pass
    return i + 1
def cal_max_threshold(record):
    max_threshold = 0
    for item in record:
        if len(item['thresholds']) == 0:
            continue
        tmp_max_threshold = max(item['thresholds'])
        if tmp_max_threshold > max_threshold:
            max_threshold = tmp_max_threshold
    return max_threshold
def process_file(file_list,output_path):
    result_path = '/scratch/vetgpt/s2orc/out'
    for file in file_list:
        file_name = os.path.basename(file).replace('.jsonl.gz', '')
        if not os.path.exists(os.path.join(output_path, file_name)):
            os.makedirs(os.path.join(output_path, file_name))
        result_list = os.listdir(os.path.join(result_path, file_name))
        # print(result_list)
        sorted_files = sorted(result_list, key=lambda x: int(re.search(r'_(\d+)', x).group(1)))
        # print(sorted_files)
        length = re.search(r'_(\d+)', sorted_files[-1]).group(1)
        file_length = get_file_length(file)
        if int(length) != file_length:
            print('Length not equal')
        count = 0
        bar = tqdm(total=file_length)
        with gzip.open(file, 'r') as f:
            with open (os.path.join(result_path, file_name, sorted_files[0]), 'r') as f1:
                result = json.load(f1)
            _15 = []
            _20 = []
            _25 = []
            _30 = []
            for line in f:
                content = json.loads(line)
                if content is not None:
                    max_thresholds = cal_max_threshold(result[count%1000])
                    if max_thresholds >= 0.3:
                        _30.append({'idx': count, 'max_threshold': max_thresholds,'text': content['content']['text']})
                    elif max_thresholds >= 0.25:
                        _25.append({'idx': count, 'max_threshold': max_thresholds,'text': content['content']['text']})
                    elif max_thresholds >= 0.2:
                        _20.append({'idx': count, 'max_threshold': max_thresholds,'text': content['content']['text']})
                count +=1
                bar.update(1)
                if count % 1000 == 0:
                    del result
                    result = json.load(open(os.path.join(result_path, file_name, sorted_files[count//1000]), 'r'))
            bar.close()
        with open(os.path.join(output_path, file_name, '30.json'), 'w') as f:
            json.dump(_30, f)
        with open(os.path.join(output_path, file_name, '25.json'), 'w') as f:
            json.dump(_25, f)
        with open(os.path.join(output_path, file_name, '20.json'), 'w') as f:
            json.dump(_20, f)
        print('Finish', file)
        del _15
        del _20
        del _25
        del _30
        del result
        

def process_total(target_path,out_path):
    file_list = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.endswith('.jsonl.gz'):
                file_list.append(os.path.join(root, file))
    # remove_list = []
    # for idx, file_path in enumerate(file_list):
    #     out_file_path = file_path.replace('.json.gz', '.json')
    #     if os.path.exists(out_file_path):
    #         remove_list.append(idx)
    # file_list = [file_list[i] for i in range(len(file_list)) if i not in remove_list]
    print(f'Total {len(file_list)} files')
    
    
    num_processes = 68
    chunked_filenames = np.array_split(file_list, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        out_folder = os.path.join(out_path, str(i))
        p = Process(target=process_file, args=(list(subset),out))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
if __name__ == '__main__':
    set_start_method('spawn')  # Set the start method for multiprocessing
    # Filter out all future warnings and specific warnings
    target_path = "/scratch/vetgpt/s2orc/2024-04-16"
    out_path = "/scratch/vetgpt/s2orc/output"
    process_total(target_path, out_path)