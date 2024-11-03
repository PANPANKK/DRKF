import torch
import torch.nn as nn
import torch.nn.functional as F

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
