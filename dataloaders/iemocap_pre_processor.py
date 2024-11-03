import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torchaudio
import numpy as np
import os
import random


class IEMOCAPDataset(Dataset):
    def __init__(self, feature_save_path):
        # 直接加载保存的pkl文件中的数据
        with open(feature_save_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def process_and_save_dataset(data_list_path, iemocap_aug_datapath, feature_save_path, max_duration=25):
    # 检查是否已经存在保存好的pkl文件
    if os.path.exists(feature_save_path):
        print(f"{feature_save_path} 已存在，跳过数据处理。")
        return

    augmented_audio = os.listdir(iemocap_aug_datapath)
    augmented_audio_dictionary = {}
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for item in augmented_audio:
        gt_audio = "Ses" + item.split("Ses")[-1]
        if gt_audio in augmented_audio_dictionary:
            augmented_audio_dictionary[gt_audio].append(iemocap_aug_datapath + item)
        else:
            augmented_audio_dictionary[gt_audio] = [iemocap_aug_datapath + item]

    dataset_features = []

    # 遍历整个数据集并处理特征
    for index in range(len(lines)):
        raw_audio_path, raw_text, label, augmented_text = lines[index].replace('\n', '').split('\t')
        raw_audio_name = os.path.basename(raw_audio_path)
        augmented_wav_index = random.randint(0, len(augmented_audio_dictionary[raw_audio_name]) - 1)
        augmented_wav_path = augmented_audio_dictionary[raw_audio_name][augmented_wav_index]

        # 加载原始和增强的音频
        raw_waveform, raw_sample_rate = torchaudio.load(raw_audio_path)
        aug_waveform, aug_sample_rate = torchaudio.load(augmented_wav_path)

        # 计算音频时长并过滤超过 25 秒的音频
        raw_duration = raw_waveform.shape[1] / raw_sample_rate
        aug_duration = aug_waveform.shape[1] / aug_sample_rate

        # 如果音频时长大于 max_duration，跳过该音频以及其对应的文本和标签
        if raw_duration > max_duration or aug_duration > max_duration:
            print(f"跳过样本 {raw_audio_name}，因为音频时长超过了 {max_duration} 秒。")
            continue

        # 保存音频的特征
        raw_waveform = raw_waveform.numpy()
        aug_waveform = aug_waveform.numpy()
        raw_audio_seq_input, raw_audio_cls_input, raw_audio_dimension = process_func(raw_waveform, raw_sample_rate)
        aug_audio_seq_input, aug_audio_cls_input, aug_audio_dimension = process_func(aug_waveform, aug_sample_rate)

        # 将特征存储到一个字典中
        features = {
            'raw_waveform': raw_waveform,
            'aug_waveform': aug_waveform,
            'raw_sample_rate': raw_sample_rate,
            'aug_sample_rate': aug_sample_rate,
            'raw_sentimental_density': raw_audio_dimension,
            'aug_sentimental_density': aug_audio_dimension,
            'raw_text_input': raw_text,
            'aug_text_input': augmented_text,
            'label': int(label)
        }
        # 添加到列表中
        dataset_features.append(features)

    # 保存所有处理后的特征到一个pkl文件中
    with open(feature_save_path, 'wb') as f:
        pickle.dump(dataset_features, f)
    print(f"特征已保存到 {feature_save_path}")



def collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_waveforms = [sample['raw_waveform'].squeeze(0) for sample in batch]
    aug_waveforms = [sample['aug_waveform'].squeeze(0) for sample in batch]
    raw_sample_rates = [sample['raw_sample_rate'] for sample in batch]
    aug_sample_rates = [sample['aug_sample_rate'] for sample in batch]

    # 使用processor处理音频序列，获取input_ids和attention_masks
    raw_audio_processed = processor(
        raw_waveforms, sampling_rate=raw_sample_rates[0], return_tensors="pt", padding=True, return_attention_mask=True
    )
    aug_audio_processed = processor(
        aug_waveforms, sampling_rate=aug_sample_rates[0], return_tensors="pt", padding=True, return_attention_mask=True
    )

    # 从processed数据中获取input_ids和attention_masks
    raw_audio_input_ids = raw_audio_processed['input_values'].to(device)
    raw_attention_masks = raw_audio_processed['attention_mask'].to(device)
    aug_audio_input_ids = aug_audio_processed['input_values'].to(device)
    aug_attention_masks = aug_audio_processed['attention_mask'].to(device)

    # 获取文本输入和标签
    raw_texts = [sample['raw_text_input'] for sample in batch]
    aug_texts = [sample['aug_text_input'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long).to(device)
    # 获取音频维度特征 (audio dimension)，并将其移动到 GPU
    raw_audio_dimensions = torch.stack([torch.tensor(sample['raw_sentimental_density']) for sample in batch]).to(device)
    aug_audio_dimensions = torch.stack([torch.tensor(sample['aug_sentimental_density']) for sample in batch]).to(device)

    return {
        'raw_audio_input_ids': raw_audio_input_ids,
        'raw_attention_masks': raw_attention_masks,
        'aug_audio_input_ids': aug_audio_input_ids,
        'aug_attention_masks': aug_attention_masks,
        'raw_audio_dimensions': raw_audio_dimensions,
        'aug_audio_dimensions': aug_audio_dimensions,
        'raw_text': raw_texts,
        'aug_text': aug_texts,
        'label': labels
    }


class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
###  加载预训练的Wav2vec2模型，输出的hidden_states1是最后一层隐藏层的沿着时间序列方向平均池化后的结果结果，logits是池化向量映射到三维连续情感空间的结果。
###  如果想要取出，hidden_states的非池化结果，即维度为[batch, seq_len, 1024],只需要取出 hidden_states0 = outputs[0]
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
    def forward(
            self,input_values,attention_mask=None):
        outputs = self.wav2vec2(input_values,attention_mask=attention_mask)
        hidden_states0 = outputs[0]
        hidden_states1 = torch.mean(hidden_states0, dim=1)
        logits = self.classifier(hidden_states1)
        return hidden_states0, hidden_states1, logits

### 将整个wav2vec计算过程封装，其中processor的作用是对语音进行编码，类似于BERT中的tokenizer.
def process_func(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)
    ### 这里，如果embeddings为true，则返回的y其实是池化后的hidden_states，否则是logits
    with torch.no_grad():
        y = audio_model(y)
    return y[0],y[1],y[2]

### 使用processor来获取语音的tokenizer
def audio_processor(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    inputs = processor(x, sampling_rate=sampling_rate,return_attention_mask=True, padding=True)
    input_values = inputs['input_values'][0]  # 输入的音频特征 (input_ids)
    attention_mask = inputs['attention_mask'][0]  # attention mask
    return input_values,attention_mask

# if __name__ =='__main__':
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = '../wav2vec2-large-uncased'
processor = Wav2Vec2Processor.from_pretrained(model_name)
audio_model = EmotionModel.from_pretrained(model_name).to(device)
# 处理并保存训练数据和验证数据
session_id=1
train_data_list_path = f'../dataset_session{session_id}/train_data.txt'
val_data_list_path = f'../dataset_session{session_id}/test_data.txt'
iemocap_aug_datapath = '/home/wangchai/SpeechEmotion/FUXIAN_MMER2023/MMER-main/data/iemocap_aug/out/'
# 设置最大音频长度 25 秒
# process_and_save_dataset(train_data_list_path, iemocap_aug_datapath, f'train_features_session{session_id}.pkl', max_duration=25)
# process_and_save_dataset(val_data_list_path, iemocap_aug_datapath, f'val_features_session{session_id}.pkl', max_duration=25)
print("pre_processor finished")
# # # 加载保存好的特征
# train_dataset = IEMOCAPDataset('train_features_session2.pkl')
# val_dataset = IEMOCAPDataset('val_features_session2.pkl')
# # # 创建dataloader
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
#
# for batch in train_dataloader:
#     raw_audio_input_ids, raw_attention_masks, raw_text, labels = batch['raw_audio_input_ids'], batch['raw_attention_masks'], \
#                                  batch['raw_text'],  batch['label']
#     print("raw_audio_input_ids:",raw_audio_input_ids.shape)
#     print("raw_attention_masks:", raw_attention_masks.shape)
#     print("raw_text:", raw_text)
#     print("labels:", labels)