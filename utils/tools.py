import logging
import torch
import random
import numpy as np
import os, re
# import soundfile as sf
import json
import yaml
import pandas as pd
from pathlib import Path
import inspect
import sys
from datetime import datetime
import time

def list_files(startpath):
    #输出树形结构
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def file_list(directory):
    # 确保给定的路径存在
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return []

    # 使用os.listdir()获取目录内容
    items = os.listdir(directory)

    # 过滤出文件，排除子目录
    files_only = [item for item in items if os.path.isfile(os.path.join(directory, item))]

    return files_only


def get_folder_names(directory):
    """
    获取指定目录下所有子文件夹的名字，并返回为一个列表。

    :param directory: 字符串，要检查的目录路径。
    :return: 包含所有子文件夹名字的列表。
    """
    # 确保给定的路径是一个目录
    if not os.path.isdir(directory):
        print(f"'{directory}' 不是一个有效的目录。")
        return []

    # 使用os.listdir()获取目录内容，然后过滤出文件夹
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return folders


def create_directory(directory_path):
    """
    如果指定的目录不存在，则创建该目录及其所有父目录。

    :param directory_path: 字符串，要创建的目录路径。
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        # print(f"目录 '{directory_path}' 已创建。")
    else:
        # print(f"目录 '{directory_path}' 已经存在。")

        import re


def remove_non_ascii(s):
    """移除字符串中的所有非ASCII字符"""
    # 使用正则表达式匹配非ASCII字符并替换为空字符串
    return re.sub(r'[^\x00-\x7F]', '', s)


# filter:map{'original':'replacement'}
def str_char_replace(s, filter):
    for original, replacement in filter.items():
        s = s.replace(original, replacement)
    return s


# 读取punc.textgrid解析intervals
def parse_textgrid(textgrid_path):
    with open(textgrid_path, 'r') as file:
        lines = file.readlines()[4:]
        intervals = []
        reading_intervals = False
        start_time = end_time = -1

        # 滤除字符
        char_filter = {'"': '', "\n": ''}

        for line in lines:
            if line.startswith('1'):
                reading_intervals = True
                time_info = line[2:].split(' ')
                start_time, end_time = map(float, time_info)
            elif reading_intervals:
                reading_intervals = False
                text = str_char_replace(line, char_filter)
                intervals.append(DotDict({'start_time': start_time, 'end_time': end_time, 'text': text}))

        return intervals


# 将字典转为对象 https: // blog.csdn.net / redrose2100 / article / details / 121266340
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __getstate__(self):
        # 这里定义了 __getstate__ 方法
        return self

    def __setstate__(self, state):
        # 这里定义了 __setstate__ 方法
        self.update(state)


# 检验第二个区间是否属于第一个区间
def is_sub_intervals(interval1, interval2):
    # interval1 = DotDict(interval1)
    # interval2 = DotDict(interval2)

    return interval1.start_time <= interval2.start_time and interval1.end_time >= interval2.end_time


# 获取音频长度
# def get_audio_duration(file_path):
#     # 读取音频文件
#     data, sample_rate = sf.read(file_path)

#     # 计算音频长度（秒）
#     duration = len(data) / sample_rate

#     return duration


def get_device(cuda_device=None):
    if cuda_device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cuda:{}'.format(cuda_device))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dict_to_json(data, file_path, indent=4):
    """
    Save a dictionary to a JSON file.

    :param data: Dictionary to be saved.
    :param file_path: Path to the output JSON file.
    :param indent: Number of spaces used for indentation (default is 4).
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=indent)
    print(f"Data has been saved to {file_path}")


def read_json_to_dict(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
        print(f"The file {file_path} does not exist.")
        return {}

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return DotDict(data)

def json_to_yaml(json_path, yaml_path):
    """
    将 JSON 文件转换为 YAML 文件。

    参数:
    - json_path: JSON 文件的路径。
    - yaml_path: YAML 文件的保存路径。

    返回:
    - 无返回值，直接保存 YAML 文件。
    """
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # 将 JSON 数据转换为 YAML 格式
    yaml_data = yaml.dump(json_data, default_flow_style=False)

    # 保存 YAML 文件
    with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
        yaml_file.write(yaml_data)

def yaml_to_json(yaml_path, json_path):
    """
    将 YAML 文件转换为 JSON 文件。

    参数:
    - yaml_path: YAML 文件的路径。
    - json_path: JSON 文件的保存路径。

    返回:
    - 无返回值，直接保存 JSON 文件。
    """
    # 读取 YAML 文件
    with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    # 将 YAML 数据转换为 JSON 格式
    json_data = json.dumps(yaml_data, indent=4, ensure_ascii=False)

    # 保存 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


def read_yaml_to_dict(file_path):
    """
    读取 YAML 文件并将其内容作为字典返回。

    参数:
        file_path (str): YAML 文件的路径。

    返回:
        dict: YAML 文件的内容作为字典。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return DotDict(data)


#默认第0行为header,None则无header
def read_data_sheet(file_path, header=0, sheet_name=None)->pd.DataFrame:
    # 获取文件扩展名
    _, file_extension = os.path.splitext(file_path)

    # 根据文件扩展名选择读取方法
    if file_extension in ('.xls', '.xlsx'):
        # 读取 Excel 文件
        df = pd.read_excel(file_path, header=header, sheet_name=sheet_name)
        # print(type(df))
    elif file_extension == '.csv':
        # 读取 CSV 文件
        df = pd.read_csv(file_path, header=header)
    elif file_extension == '.tsv':
        # 读取 TSV 文件
        df = pd.read_csv(file_path, sep='\t', header=header)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return df


def read_text_list(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行末尾的换行符
            line = line.strip()
            lines.append(line)
    return lines


def list2txt(data, filename):
    """
    将列表中的元素保存到文本文件中，每个元素占一行。

    :param data: 要保存的列表
    :param filename: 保存的文件名
    """
    with open(filename, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')


import subprocess


def choose_gpu(gpu_memory_threshold=0):
    # 获取GPU数量
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    num_gpus = len(result.stdout.split('\n')) - 1

    # 查找可用显存最大的GPU
    max_free_memory = gpu_memory_threshold
    selected_gpu_id = None

    for gpu_id in range(num_gpus):
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True)
        free_memory = int(result.stdout.split('\n')[0])

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            selected_gpu_id = gpu_id
            # 锁定GPU
    if selected_gpu_id is not None:
        print(f"choose gpu: {selected_gpu_id}")
        return str(selected_gpu_id)
    return None


def get_current_file_abs_path(cur_file=__file__):
    """
    获取当前 Python 文件的绝对路径。
    """
    return Path(cur_file).resolve()


def get_current_root(cur_file=__file__):
    """
    获取当前 Python 文件所在的根目录。
    """
    current_file_path = get_current_file_abs_path(cur_file)
    return current_file_path.parent


def get_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_to_file(log_file_path, **kwargs):
    """
    将多个变量的名称及其值保存到指定的日志文件中，并确保文件是一个完整的 JSON 列表。

    :param log_file_path: 日志文件的路径
    :param args: 位置参数
    :param kwargs: 关键字参数
    """
    # 创建目录（如果不存在）
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取当前调用栈的信息
    frame = inspect.currentframe().f_back
    # code_context = frame.f_code.co_code[frame.f_lasti + 3:frame.f_lasti + 6].decode()

    # 构建日志内容
    log_entry = {}
    # log_entry = {name: value for name, value in zip(arg_names, args)}
    log_entry.update(kwargs)

    # 读取现有日志文件内容
    existing_logs = []
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            try:
                existing_logs = json.load(log_file)
            except json.JSONDecodeError:
                pass

    # 将新的日志条目追加到现有日志列表中
    existing_logs.append(log_entry)

    # 写入更新后的日志列表
    with open(log_file_path, 'w') as log_file:
        json.dump(existing_logs, log_file, indent=4)


def move_to_device(data, device):
    """
    将输入数据迁移到指定设备上。

    如果数据已经是 torch.Tensor，则直接迁移；
    如果数据是 numpy.ndarray，则先转换为 torch.Tensor 再迁移；
    如果数据是字典，则递归地将每个张量属性移动到设备上。

    参数:
    - data: 输入数据，可以是 torch.Tensor、numpy.ndarray 或字典。
    - device: 目标设备，例如 'cuda' 或 'cpu'。

    返回:
    - 迁移到目标设备上的数据。
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, dict) or isinstance(data,DotDict):
        return DotDict({k: move_to_device(v, device) for k, v in data.items()})
    else:
        return data  # raise TypeError(f"Unsupported data type: {type(data)}")

def get_time():
    return time.time()

def calculate_elapsed_time(start_time):
    """
    计算从开始时间到当前时间的差值（秒）。
    
    参数:
    - start_time: 开始时间（float）
    
    返回:
    - elapsed_time: 经过的时间（秒）
    """
    current_time = time.time()
    elapsed_time = current_time - start_time
    return elapsed_time

def get_sheet_names(file_path):
    """
    获取Excel文件中的所有工作表名称。
    
    参数:
    file_path (Path): Excel文件路径。
    
    返回:
    list: 所有工作表的名称列表。
    """
    excel_file = pd.ExcelFile(file_path)
    return excel_file.sheet_names

def filter_df(df, target_columns):
    filtered_data = df.copy()
    
    # 获取列名和列索引的映射关系
    column_names = df.columns.tolist()
    
    for key, value in target_columns.items():
        # 判断 key 是否为整数（列索引号）
        if isinstance(key, int):
            # 根据列索引号获取列名
            if key < len(column_names):
                column = column_names[key]
            else:
                raise IndexError(f"Column index {key} is out of range.")
        else:
            # 直接使用列名
            column = key
        
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lower_bound, upper_bound = value
            filtered_data = filtered_data[(filtered_data[column] >= lower_bound) & (filtered_data[column] <= upper_bound)]
        else:
            filtered_data = filtered_data[filtered_data[column] == value]
    
    return filtered_data

def get_exec_name(__file__):
    return str(os.path.basename(__file__)).split('.')[0]

# if __name__=='__main__':
#     json_to_yaml('mdpe_bai_hbl_vit_config.json','mdpe_bai_hbl_vit_config.yaml')
#     # yaml_to_json('config.yaml','config.json')