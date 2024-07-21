# ## 3) MODELS

import torch
import torch.nn as nn


class Patchify(nn.Module): 
    """Takes an image and converts to patches represented with 
    embedded vectors of length d: [B,Cin,W,H] --> [B, num_patches, d]"""

    def __init__(self, H, P, Cin, d):
        super().__init__()
        
        """
        H: image height = width
        P: patch height = width
        Cin: num input channels
        d: embedding vector to represent each patch
            = num_output channels = number of filters
        """
        
        # Calculate the number of patches 
        self.num_patches = (H//P)**2 
        
        # Projection layer to convert the image into patches
        self.proj = nn.Conv2d(Cin, d, kernel_size=P, stride=P)
        
    def forward(self, x): # x: image, [B,Cin,W,H]

        x = self.proj(x) #[B, d, num_patches_per_row, num_patches_per_column]
        x = x.flatten(2) #[B, d, num_patches]
        x = x.transpose(1, 2) #[B, num_patches, d]
        
        return x


# ### 3.2) Transformer blocks

class MHA(nn.Module): #for encoder, no masking
    def __init__(self, d, nh, dpo): #No Nmax
        
        """
        nh: number of heads
        """
        
        super().__init__()
        
        assert d % nh == 0
        self.nh = nh
        self.dh = d // nh  

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d)  
        
        self.dropout = nn.Dropout(dpo)
        #no register_buffer

    def forward(self, x):
        B, N, d = x.shape

        # Form QKV before splitting data into heads
        Q = self.Wq(x) # (B, N, d)
        K = self.Wk(x)  
        V = self.Wv(x)

        # Split QKV into heads: d -> nh x dh
        Q = Q.view(B, N, self.nh, self.dh) #(B, N, nh, dh)
        K = K.view(B, N, self.nh, self.dh)
        V = V.view(B, N, self.nh, self.dh)
        
        # Transpose
        Q = Q.transpose(1, 2) #(B, nh, N, dh)
        K = K.transpose(1, 2) 
        V = V.transpose(1, 2)

        # Calculate QKT = attention scores for each head
        S = Q @ K.transpose(2, 3) #(B, nh, N, N) 

        # Softmax along the rows, no mask applied to S
        P = torch.softmax(S / self.dh**0.5, dim=-1) #(B, nh, N, N)
        P = self.dropout(P)

        # Calculate the output of each head
        PV = (P @ V) #(B, nh, N, dh)
        
        # Concat along columns = transpose & reshape --> col dim = d = nh * dh
        PV = PV.transpose(1, 2).reshape(B, N, d) #(B, N, d)
        
        # Wo projection
        self_attn_out = self.Wo(PV) #(B, N, d)

        return self_attn_out


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        
        self.linear1 = nn.Linear(d, 4*d) 
        self.gelu = GELU()
        self.linear2 = nn.Linear(4*d, d) # not followed by an activation function 
        
    def forward(self, x):
        x = self.linear1(x) #(B, N, 4d)
        x = self.gelu(x)
        x = self.linear2(x) #(B, N, d)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d):
        
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) #row-wise
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        #each column of norm_x scales and shifted by a dedicated parameter
        #(d)*(N,d)+(d)
        norm_x = self.scale * norm_x + self.shift
        
        return norm_x
