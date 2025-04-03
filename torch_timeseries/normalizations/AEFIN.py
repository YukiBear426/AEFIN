import time
import torch
import torch.nn as nn

def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
        
    k_values = torch.topk(xf.abs(), k, dim = 1)  
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered


class AEFIN(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, num_heads=4, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        self.rfft = rfft
        self.num_heads = num_heads
        
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))
        
    def _build_model(self):
        self.model_freq = FANLayer(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
        self.num_heads = min([i for i in range(1, self.enc_in+1) if self.enc_in % i == 0], default=1)
        self.attention = nn.MultiheadAttention(embed_dim=self.enc_in, num_heads=self.num_heads)

    def loss(self, true):
        B, O, N = true.shape
        residual, pred_main = main_freq_part(true, self.freq_topk, self.rfft)

        stable_loss = nn.functional.mse_loss(self.pred_residual, residual)  
        unstable_loss = nn.functional.l1_loss(self.pred_main_freq_signal, pred_main)  

        freq_true = torch.fft.rfft(residual, dim=1)
        freq_pred = torch.fft.rfft(self.pred_residual, dim=1)

        freq_loss = nn.functional.mse_loss(freq_pred.abs(), freq_true.abs())

        alpha, beta, gamma = 0.5, 0.2, 0.3
        total_loss = alpha * stable_loss + beta * unstable_loss + gamma * freq_loss
        
        return total_loss
        
    
    def normalize(self, input):
        bs, len, dim = input.shape
        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)

        attn_output, _ = self.attention(x_filtered.transpose(0, 1), norm_input.transpose(0, 1), norm_input.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)

        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1, 2), input.transpose(1, 2)).transpose(1, 2)
        
        return attn_output.reshape(bs, len, dim)



    def denormalize(self, input_norm):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = input_norm.shape
        
        # freq denormalize
        self.pred_residual = input_norm

        output = self.pred_residual + self.pred_main_freq_signal
        
        return output.reshape(bs, len, dim)

    
    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)

import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 3*input_dim)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(3*input_dim,output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)

        return x

class FANLayer(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, bias=True, with_gate=True):
        super(FANLayer, self).__init__()
        self.seq_len = seq_len
        self.input_dim = seq_len
        self.output_dim = pred_len

        self.input_linear_p = nn.Linear(seq_len, pred_len // 4, bias=bias)
        self.input_linear_g = nn.Linear(seq_len, (pred_len - pred_len // 2))
        self.activation = nn.GELU()        

        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
                
        self.mlp = MLP(input_dim=seq_len + pred_len, output_dim=pred_len)
    
    def forward(self, main_freq,x):
        batch_size, enc_in, input_dim = main_freq.shape
        outputs = []

        for i in range(enc_in): 

            channel_data = main_freq[:, i, :].squeeze(1) 

            g = self.activation(self.input_linear_g(channel_data))
            p = self.input_linear_p(channel_data)
            
            if not hasattr(self, 'gate'):
                output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
            else:
                gate = torch.sigmoid(self.gate)
                output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)

            output = output.unsqueeze(1)
            outputs.append(output)

        final_output = torch.cat(outputs, dim=1)
        
        output_with_x = torch.cat((final_output, x), dim=-1)  # (batch_size, enc_in, seq_len + pred_len)

        final_output = self.mlp(output_with_x)
 
        return final_output
