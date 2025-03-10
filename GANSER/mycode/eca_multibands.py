import math
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class ECA(nn.Module):
    def __init__(self, num_attention_heads, seq_len, hidden_dropout_prob):
        """
        计算 128 维度（时间步）的注意力，适用于多频带 EEG 信号。
        
        Args:
            num_attention_heads: 头数
            seq_len: 需要进行注意力计算的维度大小（这里是 128）
            hidden_dropout_prob: dropout 概率
        """
        super(ECA, self).__init__()
        if seq_len % num_attention_heads != 0:
            raise ValueError(
                "The sequence length (%d) must be a multiple of attention heads (%d)"
                % (seq_len, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = seq_len // num_attention_heads
        self.all_head_size = seq_len

        self.query = nn.Linear(seq_len, seq_len)
        self.key = nn.Linear(seq_len, seq_len)
        self.value = nn.Linear(seq_len, seq_len)

        self.attn_dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(seq_len, seq_len)
        self.LayerNorm = LayerNorm(seq_len, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        # 变换形状为 (batch, num_bands, num_heads, seq_len / num_heads, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)  # (batch, num_bands, seq_len / num_heads, num_heads, head_size)

    def forward(self, input_tensor):
        B, C, T, H, W = input_tensor.shape  # (batch_size, num_bands, 128, 9, 9)

        # 1. **调整形状，使其适应注意力计算**
        input_tensor = input_tensor.view(B, C, T, -1)  # (batch_size, num_bands, 128, 81)
        input_tensor = input_tensor.permute(0, 1, 3, 2)  # (batch_size, num_bands, 81, 128)

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 2. **计算注意力，仅作用于 `128` 维度**
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch_size, num_bands, 81, 128)

        # 3. **应用残差连接和归一化**
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # 4. **恢复原始形状 (batch_size, num_bands, 128, 9, 9)**
        hidden_states = hidden_states.permute(0, 1, 3, 2)  # (batch_size, num_bands, 128, 81)
        hidden_states = hidden_states.view(B, C, T, H, W)  # 恢复 (batch_size, num_bands, 128, 9, 9)

        return hidden_states
