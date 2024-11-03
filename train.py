import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from transformers import BertConfig
import torchaudio
import numpy as np
from pre_processor import *
from multi_modality_fusion import *
from tqdm import tqdm
import random
import os
import copy
from logger import Logger
from utils import *
from networks.autoencoder import ResidualAE
from torch.cuda.amp import GradScaler, autocast
# 初始化GradScaler 用于混合精度
scaler = GradScaler()

current_root=get_current_root(__file__)

set_seed(42)

# Set device
device = get_device()

log_name=get_exec_name(__file__)

print(log_name)
print('-'*50)
#loss weight
text_contrastive_loss_w=0.2
audio_contrastive_loss_w=0.2
classification_loss_w=1
binary_classification_loss_w=0.2
modal_contrastive_loss_w=0.2
aug_classification_loss_w=1
aug_text_loss_w=0.2
aug_audio_loss_w=0.2

# Define single modality projection heads
class AudioProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AudioProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class TextProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Define temperature model
class TemperatureModel(nn.Module):
    def __init__(self, initial_temp=1.0):
        super(TemperatureModel, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temp))

    def forward(self):
        return torch.sigmoid(self.temperature)  # Limit temperature between 0 and 1

temperature_model_init = TemperatureModel().to(device)

# RoBERTa configuration
roberta_tokenizer = RobertaTokenizer.from_pretrained('../roberta-large-uncased')
roberta_model_init = RobertaModel.from_pretrained('../roberta-large-uncased').to(device)

# Get text embeddings
class GetTextEmbedding(nn.Module):
    def __init__(self, tokenizer, text_model):
        super(GetTextEmbedding, self).__init__()
        self.tokenizer = tokenizer
        self.text_model = text_model

    def forward(self, text_inputs):
        text_embeddings = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True,
                                         max_length=80).to(device)
        text_output = self.text_model(**text_embeddings)
        text_seq_embedding, text_cls_embeddings = text_output[0], text_output[1]
        return text_seq_embedding, text_cls_embeddings

# Instantiate models
num_labels = 4
batch_size=4
TextEmbedding_model_init = GetTextEmbedding(roberta_tokenizer, roberta_model_init).to(device)

# Initialize multimodal fusion model
bert_config = BertConfig(
    hidden_size=1024,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=4096,
    max_position_embeddings=2800,
    num_labels=num_labels
)
Bert_adapter_multimodel_fusion_init = MultimodalBertModel(bert_config).to(device)

# Initialize projection heads
audio_projection_head_init = AudioProjectionHead(input_dim=1024, output_dim=1024).to(device)
text_projection_head_init = TextProjectionHead(input_dim=roberta_model_init.config.hidden_size, output_dim=1024).to(device)

# Initialize loss function
classification_criterion = nn.CrossEntropyLoss()    #分类损失
augmentation_constr_crit=nn.MSELoss()               #比较与原始样本特征的差距

# Define contrastive loss function
def nt_xent_loss(embeddings1, embeddings2, temperature):
    batch_size = embeddings1.size(0)
    device = embeddings1.device
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)
    # Concatenate embeddings
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # [2*batch_size, dim]
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.t())  # [2*batch_size, 2*batch_size]
    # Remove self-similarity
    mask = torch.eye(2 * batch_size).bool().to(device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    # Create labels for positive pairs
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]).to(device)
    # Similarity divided by temperature
    similarity_matrix = similarity_matrix / temperature
    # Cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# 模态间对比学习损失（NCE）
def modal_nt_xent_loss(audio_embeddings, text_embeddings, temperature):
    batch_size = audio_embeddings.size(0)
    device = audio_embeddings.device

    # Normalize embeddings
    audio_embeddings = F.normalize(audio_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    # Similarity matrix between audio and text
    similarity_matrix = torch.matmul(audio_embeddings, text_embeddings.t())  # [batch_size, batch_size]

    # Similarity divided by temperature
    similarity_matrix = similarity_matrix / temperature

    # Create labels (positive pairs are diagonal elements)
    labels = torch.arange(batch_size).to(device)

    # Cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def create_binary_classification_pairs(audio_embeddings, text_embeddings, labels):
    pairs = []
    pair_labels = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                pairs.append((audio_embeddings[i], text_embeddings[j]))
                pair_labels.append(1 if labels[i] == labels[j] else 0)
    return pairs, pair_labels

def get_cls_from_seq_max_pooling(input_tensor):
    """
    使用最大池化获取分类向量。
    
    :param input_tensor: 形状为 [batch_size, seq_len, feat_dim] 的张量
    :return: 形状为 [batch_size, feat_dim] 的张量
    """
    max_pooled, _ = torch.max(input_tensor, dim=1)  # 沿着 seq_len 维度求最大值
    return max_pooled

def get_cls_from_seq_avg_pooling(input_tensor):
    """
    使用平均池化获取分类向量。
    
    :param input_tensor: 形状为 [batch_size, seq_len, feat_dim] 的张量
    :return: 形状为 [batch_size, feat_dim] 的张量
    """
    avg_pooled = torch.mean(input_tensor, dim=1)  # 沿着 seq_len 维度求平均
    return avg_pooled

# Evaluate model function
def evaluate_model(text_model, audio_model, fusion_model, dataloader):
    text_model.eval()
    audio_model.eval()
    fusion_model.eval()
    audio_projection_head_init.eval()
    text_projection_head_init.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0
    class_correct = np.zeros(num_labels)
    class_total = np.zeros(num_labels)
    with torch.no_grad():
        for batch in dataloader:
            raw_audio_input_ids, raw_attention_masks, aug_audio_input_ids, aug_attention_masks, \
            raw_audio_dimensions, aug_audio_dimensions, raw_text, aug_text, labels = batch['raw_audio_input_ids'], \
                                                                                     batch['raw_attention_masks'], \
                                                                                     batch['aug_audio_input_ids'], \
                                                                                     batch['aug_attention_masks'], \
                                                                                     batch['raw_audio_dimensions'], \
                                                                                     batch['aug_audio_dimensions'], \
                                                                                     batch['raw_text'], batch['aug_text'], batch['label']

            # Get embeddings
            raw_text_seq_embedding, raw_text_cls_embeddings = text_model(raw_text)
            raw_text_cls_embeddings = text_projection_head_init(raw_text_cls_embeddings)

            raw_audio_seq_embedding, raw_audio_cls_embeddings, _ = audio_model(raw_audio_input_ids, raw_attention_masks)
            raw_audio_cls_embeddings = audio_projection_head_init(raw_audio_cls_embeddings)

            # Fuse modalities and compute logits
            logits, _,fusion_Cls = fusion_model(raw_audio_seq_embedding, raw_text_seq_embedding)
            predictions = torch.argmax(logits, dim=-1)

            # Compute loss
            loss = classification_criterion(logits, labels.to(device))
            total_loss += loss.item()

            # Compute accuracy
            correct_predictions += (predictions == labels.to(device)).sum().item()
            total_predictions += labels.size(0)

            # Class-wise accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predictions[i] == labels[i]).item()
                class_total[label] += 1

    avg_loss = total_loss / total_predictions
    class_accuracy = class_correct / class_total
    accuracy = class_accuracy.mean()
    class_weights = class_total / total_predictions
    weighted_accuracy = (class_accuracy * class_weights).sum()

    return accuracy, avg_loss, weighted_accuracy, class_accuracy, class_weights

# Training loop
num_epochs = 100
best_model_path = 'best_emotion_prediction_model.pth'
best_accuracy = 0.0
# Initialize the feature augmentation networks
AE_layers, AE_blk_num, AE_input_dim=[256,128,64],5,1024
audio_feature_augmentation = ResidualAE(AE_layers, AE_blk_num, AE_input_dim, dropout=0, use_bn=False).to(device)
text_feature_augmentation = ResidualAE(AE_layers, AE_blk_num, AE_input_dim, dropout=0, use_bn=False).to(device)
# Deep copy initial models
base_temperature_model = temperature_model_init.to(device)
base_text_model = TextEmbedding_model_init.to(device)
base_audio_model = audio_model.to(device)
base_text_projection_head = text_projection_head_init.to(device)
base_audio_projection_head = audio_projection_head_init.to(device)
base_fusion_model = Bert_adapter_multimodel_fusion_init.to(device)


session_id=1
# logger=Logger(current_root/'log'/f'{get_datetime()}_{log_name}_session{session_id}.log')


# Loop over sessions
# for session_id in range(1, 6):
if session_id:
    print(f"Training on Session {session_id}")
    logger=Logger(current_root/'log'/f'{get_datetime()}_{log_name}_session{session_id}.log')
    # Initialize models
    temperature_model_init.load_state_dict(base_temperature_model.state_dict())
    TextEmbedding_model_init.load_state_dict(base_text_model.state_dict())
    audio_model.load_state_dict(base_audio_model.state_dict())
    text_projection_head_init.load_state_dict(base_text_projection_head.state_dict())
    audio_projection_head_init.load_state_dict(base_audio_projection_head.state_dict())
    Bert_adapter_multimodel_fusion_init.load_state_dict(base_fusion_model.state_dict())

    # Update the optimizer to include the augmentation networks
    optimizer = torch.optim.Adam(
    list(temperature_model_init.parameters()) + list(TextEmbedding_model_init.parameters()) + list(audio_model.parameters()) +
    list(text_projection_head_init.parameters()) + list(audio_projection_head_init.parameters()) +
    list(Bert_adapter_multimodel_fusion_init.parameters()) +
    list(audio_feature_augmentation.parameters()) + list(text_feature_augmentation.parameters()), lr=1e-5)
    
    train_dataset = IEMOCAPDataset(f'train_features_session{session_id}.pkl')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataset = IEMOCAPDataset(f'val_features_session{session_id}.pkl')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    for epoch in range(num_epochs):

        total_loss = 0
        temperature_model_init.train()
        TextEmbedding_model_init.train()
        audio_model.train()
        text_projection_head_init.train()
        audio_projection_head_init.train()
        Bert_adapter_multimodel_fusion_init.train()
        audio_feature_augmentation.train()
        text_feature_augmentation.train()

        for batch in tqdm(train_dataloader, desc=f"Session {session_id} Train Epoch {epoch + 1}/{num_epochs}", ncols=75):
            raw_audio_input_ids, raw_attention_masks, aug_audio_input_ids, aug_attention_masks, \
            raw_audio_dimensions, aug_audio_dimensions, raw_text, aug_text, labels = batch['raw_audio_input_ids'], batch['raw_attention_masks'], \
                                                                                 batch['aug_audio_input_ids'], batch['aug_attention_masks'], \
                                                                                 batch['raw_audio_dimensions'], batch['aug_audio_dimensions'], \
                                                                                 batch['raw_text'], batch['aug_text'], batch['label']

            optimizer.zero_grad()

         # Get text embeddings and apply augmentation
            raw_text_seq_embedding, raw_text_cls_embeddings = TextEmbedding_model_init(raw_text)
            # print(f'text_shape: seq:{raw_text_seq_embedding.shape}; cls:{raw_text_cls_embeddings.shape}')
            raw_text_cls_embeddings = text_projection_head_init(raw_text_cls_embeddings)
            
            aug_text_seq_embedding,_ = text_feature_augmentation(raw_text_seq_embedding)
            aug_text_loss=augmentation_constr_crit(aug_text_seq_embedding,raw_text_seq_embedding)
            aug_text_cls_embeddings = get_cls_from_seq_avg_pooling(aug_text_seq_embedding)  # Apply text augmentation

        # Get audio embeddings and apply augmentation
            raw_audio_seq_embedding, raw_audio_cls_embeddings, _ = audio_model(raw_audio_input_ids, raw_attention_masks)
            # print(f'audio_shape: seq:{raw_audio_seq_embedding.shape}; cls:{raw_audio_cls_embeddings.shape}')
            raw_audio_cls_embeddings = audio_projection_head_init(raw_audio_cls_embeddings)

            aug_audio_seq_embedding,_ = audio_feature_augmentation(raw_audio_seq_embedding)
            aug_audio_loss=augmentation_constr_crit(raw_audio_seq_embedding,aug_audio_seq_embedding)
            aug_audio_cls_embeddings = get_cls_from_seq_avg_pooling(aug_audio_seq_embedding) # Apply audio augmentation

        # Compute contrastive losses with augmented features
            temperature = temperature_model_init()
            text_contrastive_loss = nt_xent_loss(raw_text_cls_embeddings, aug_text_cls_embeddings, temperature)
            audio_contrastive_loss = nt_xent_loss(raw_audio_cls_embeddings, aug_audio_cls_embeddings, temperature)
            modal_contrastive_loss = modal_nt_xent_loss(raw_audio_cls_embeddings, raw_text_cls_embeddings, temperature)

        # Fuse modalities and compute classification logits
            logits, binary_logits, fusion_Cls = Bert_adapter_multimodel_fusion_init(raw_audio_seq_embedding, raw_text_seq_embedding)

        # Compute classification loss
            classification_loss = classification_criterion(logits, labels.to(device))

            aug_logits, aug_binary_logits, aug_fusion_Cls = Bert_adapter_multimodel_fusion_init(aug_audio_seq_embedding, aug_text_seq_embedding)

        # Compute classification loss of augmented_data
            aug_classification_loss = classification_criterion(logits, labels.to(device))

        # Binary classification loss
            binary_pairs, binary_pair_labels = create_binary_classification_pairs(raw_audio_seq_embedding, raw_text_seq_embedding, labels)
            binary_pairs_audio = torch.stack([pair[0] for pair in binary_pairs]).to(device)
            binary_pairs_text = torch.stack([pair[1] for pair in binary_pairs]).to(device)
            binary_pair_labels = torch.tensor(binary_pair_labels).to(device)
            _, binary_logits, _ = Bert_adapter_multimodel_fusion_init(binary_pairs_audio, binary_pairs_text)
            binary_classification_loss = classification_criterion(binary_logits, binary_pair_labels)

            with autocast():
            # Total loss
                loss = text_contrastive_loss_w*text_contrastive_loss\
                    + audio_contrastive_loss_w*audio_contrastive_loss\
                    + classification_loss_w*classification_loss\
                    + binary_classification_loss_w*binary_classification_loss\
                    + modal_contrastive_loss_w*modal_contrastive_loss\
                    +aug_classification_loss_w*aug_classification_loss\
                    +aug_text_loss_w*aug_text_loss\
                    +aug_audio_loss_w*aug_audio_loss
                # loss.backward()
                # optimizer.step()
                total_loss += loss.item()
            
            # 反向传播
            scaler.scale(loss).backward()

            # 更新权重
            scaler.step(optimizer)

            # 更新缩放器
            scaler.update()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

        # Evaluate model
        accuracy, avg_loss, weighted_accuracy, class_accuracy, class_weights = evaluate_model(
            TextEmbedding_model_init, audio_model, Bert_adapter_multimodel_fusion_init,
            tqdm(val_dataloader, desc=f"Session {session_id} Test Epoch {epoch + 1}/{num_epochs}" ,ncols=75)
        )
        print(f"Validation Results - Epoch {epoch + 1}")
        print(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}, Weighted Accuracy: {weighted_accuracy:.4f}")
        print(f"Class-wise Accuracy: {class_accuracy}")
        print('-'*50)
