import torch
import torch.nn as nn
import torch.nn.functional as F

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
