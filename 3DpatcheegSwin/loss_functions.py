import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM

def get_prefrontal_mask(grid_size=(3, 3)):
    mask = np.zeros(grid_size, dtype=np.float32)
    mask[0,1] = 0.6
    mask[1,0] = 0.6 
    mask[1,1] = 1  
    return torch.tensor(mask, dtype=torch.float32)


# 假设目标层输出为 (batch, 36, 1536)，即 6×6 patch，1536 维特征
# def reshape_transform(tensor, time=4, height=3, width=3):
#     # 输入形状: (B, 36, 1536)
#     B, seq_len, C = tensor.size()

#     result = tensor.reshape(B, time, height, width, C)
#     # result = torch.mean(result, dim=1)
#     result = result.permute(0, 4, 1, 2, 3)  # (B, C, H, W)
#     return result
def reshape_transform(tensor, time=16, height=3, width=3):
    return tensor


def compute_attention_map(model, input_tensor, layer_idx=-1, block_idx=-1):
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)  # Forward pass to populate attn_scores

    target_layers = [model.patch_embed.proj]  # 替换为你的实际目标层
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # shape: (batch, H, W)
    grayscale_cam = torch.tensor(grayscale_cam, device=input_tensor.device)
    # Normalize
    grayscale_cam = F.relu(grayscale_cam)
    grayscale_cam /= torch.amax(grayscale_cam, dim=(1, 2), keepdim=True) + 1e-8
    
    return grayscale_cam

def region_wise_loss(outputs, labels, activation_maps, mask, device):
    ce_loss = F.cross_entropy(outputs, labels, reduction='mean')
    
    mask = mask.to(device)
    batch_size = activation_maps.shape[0]
    P = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        H_i = activation_maps[i]  # (3, 3)
        masked_importance = (H_i * mask).sum()
        P[i] = torch.sigmoid(masked_importance)
    
    region_loss = ce_loss - P.mean()
    return region_loss

def sample_wise_loss(outputs, labels, activation_maps, mask, device):
    batch_size = outputs.shape[0]
    mask = mask.to(device)
    W = torch.zeros(batch_size, device=device)
    Z = mask.sum()
    
    for i in range(batch_size):
        H_i = activation_maps[i]
        masked_importance = (H_i * mask).sum()
        W[i] = torch.sigmoid(-masked_importance / (Z + 1e-8))
    
    log_probs = F.log_softmax(outputs, dim=1)
    losses = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        losses[i] = -log_probs[i, labels[i]] * W[i]
    
    sample_loss = losses.mean()
    return sample_loss

def combined_loss(outputs, labels, activation_maps, mask, device, region_weight=0.5):
    region_loss = region_wise_loss(outputs, labels, activation_maps, mask, device)
    sample_loss = sample_wise_loss(outputs, labels, activation_maps, mask, device)
    loss = region_weight * region_loss + (1 - region_weight) * sample_loss
    # 确保损失非负
    # loss = = torch.log(loss)
    return loss