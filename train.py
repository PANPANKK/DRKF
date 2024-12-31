import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from transformers import BertConfig
import torchaudio
import numpy as np
from dataloaders.iemocap_pre_processor import *
from tqdm import tqdm
import random
import os
import copy
from utils.logger import Logger
from utils.tools import *
from models.subnets.auto_encoder import ResidualAE
from models.subnets.pooling import *
from models.subnets.proj_head import *
from models.subnets.temperature_model import *
from models.subnets.text_embedding import *
from torch.cuda.amp import GradScaler, autocast
from models.fusion_classifier import MultimodalBertModel
from models.contrastive import *

from config.hyper_params_config import *

# 初始化GradScaler 用于混合精度
scaler = GradScaler()

current_root=get_current_root(__file__)

set_seed(42)

# Set device
device = get_device()

log_name=get_exec_name(__file__)

print(log_name)
print('-'*50)


# Initialize multimodal fusion model
bert_config = BertConfig(
    hidden_size=1024,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=4096,
    max_position_embeddings=2800,
    num_labels=num_labels
)
AE_layers, AE_blk_num, AE_input_dim=[512,256,256],5,1024

roberta_path='../roberta-large-uncased'

# Define temperature model

temperature_model_init = TemperatureModel().to(device)

# RoBERTa configuration
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
roberta_model_init = RobertaModel.from_pretrained(roberta_path).to(device)



TextEmbedding_model_init = GetTextEmbedding(roberta_tokenizer, roberta_model_init).to(device)

Bert_adapter_multimodel_fusion_init = MultimodalBertModel(bert_config).to(device)

# Initialize projection heads
audio_projection_head_init = AudioProjectionHead(input_dim=1024, output_dim=1024).to(device)
text_projection_head_init = TextProjectionHead(input_dim=roberta_model_init.config.hidden_size, output_dim=1024).to(device)

# Initialize loss function
classification_criterion = nn.CrossEntropyLoss()    #分类损失
augmentation_constr_crit=nn.MSELoss()               #比较与原始样本特征的差距
kl_div=F.kl_div                                     #重建样本概率分布损失

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

# unimodality aug_sample logits representation
text_classifier=FcClassifier(1024,[],num_labels).to(device)
audio_classifier=FcClassifier(1024,[],num_labels).to(device)

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
    list(audio_feature_augmentation.parameters()) + list(text_feature_augmentation.parameters())+
    list(audio_classifier.parameters())+list(text_classifier.parameters()), lr=1e-5)
    
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

        # Compute classification loss for aug_data
            classification_loss = classification_criterion(logits, labels.to(device))



        # Compute classification loss of augmented_data
            # aug_logits, aug_binary_logits, aug_fusion_Cls = Bert_adapter_multimodel_fusion_init(aug_audio_seq_embedding, aug_text_seq_embedding)
            # aug_classification_loss = classification_criterion(aug_logits, labels.to(device))

        #compute aug sample kl div to original sample on meaning
            origin_text_probs=torch.softmax(text_classifier(raw_text_cls_embeddings)[0],-1)
            aug_text_probs=torch.softmax(text_classifier(aug_text_cls_embeddings)[0],-1)
            text_kl_loss=kl_div(origin_text_probs,aug_text_probs)*text_kl_w

            origin_audio_probs=torch.softmax(text_classifier(raw_text_cls_embeddings)[0],-1)
            aug_audio_probs=torch.softmax(text_classifier(aug_text_cls_embeddings)[0],-1)
            audio_kl_loss=kl_div(origin_audio_probs,aug_audio_probs)*audio_kl_w

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
                    +aug_text_recon_loss_w*aug_text_loss\
                    +aug_audio_recon_loss_w*aug_audio_loss+\
                    +text_kl_w*text_kl_loss\
                    +audio_kl_w*audio_kl_loss # +aug_classification_loss_w*aug_classification_loss\            
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
        if best_accuracy<=accuracy and best_wa<=weighted_accuracy and accuracy>0.79 and weighted_accuracy>0.77:
            best_accuracy=accuracy
            best_wa=weighted_accuracy
            torch.save({
                'adapter_multimodel_fusion_dict': Bert_adapter_multimodel_fusion_init.state_dict(),
                'audio_projection_head_dict': audio_projection_head_init.state_dict(),
                'text_projection_head_dict': text_projection_head_init.state_dict(),
                # 'audio_embedding_model':audio_model.state_dict(),
                'text_embedding_model_dict':TextEmbedding_model_init.state_dict(),
                'audio_feature_aug_dict':audio_feature_augmentation.state_dict(),
                'text_feature_aug_dict':text_feature_augmentation.state_dict(),
                'temperature_model_dict':temperature_model_init.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'next_epoch':epoch+1
            }, current_root/f'model_checkpoint_session{session_id}.pth')
    logger.__del__()
