
import torch
import torch.nn as nn
import math

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [L, N, E]
        L, N, E = x.shape
        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        # qkv: [L, N, 3*E]
        
        qkv = qkv.reshape(L, N, 3, self.num_heads, self.head_dim)
        # qkv: [L, N, 3, H, D]
        qkv = qkv.permute(2, 1, 3, 0, 4) # [3, N, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2] # [N, H, L, D]
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # attn_weights: [N, H, L, L]
        
        output = torch.matmul(attn_weights, v) # [N, H, L, D]
        output = output.transpose(1, 2).reshape(N, L, E) # [N, L, E]
        output = output.transpose(0, 1) # [L, N, E]
        
        return self.out_proj(output), None # Return tuple to match nn.MultiheadAttention output signature

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(), 
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # self.attn_mask is ignored in simple MHA for now as it's None in FaRL
        return self.attn(x)[0]

    def forward(self, x: torch.Tensor):
        # x is (L, N, D) for MultiheadAttention
        # ln_1 expects (..., D)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        features = []
        for block in self.resblocks:
            x = block(x)
            features.append(x)
        return features

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        features = self.transformer(x)
        return features

class FaRLBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # ViT-Base config
        width = 768
        layers = 12
        heads = 12
        input_resolution = 448 # From inspection
        patch_size = 16
        output_dim = 512 
        
        self.visual = VisualTransformer(input_resolution, patch_size, width, layers, heads, output_dim)
        
        # Normalization params
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        
        # FPNs
        self.fpns = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(768, 768, 2, 2),
                nn.BatchNorm2d(768),
                nn.GELU(),
                nn.ConvTranspose2d(768, 768, 2, 2)
            ),
            nn.ConvTranspose2d(768, 768, 2, 2),
            nn.Identity(),
            nn.MaxPool2d(2, 2)
        ])
        
        self.output_indices = [3, 5, 7, 11]

    def forward(self, x: torch.Tensor):
        x = (x - self.image_mean) / self.image_std
        features_list = self.visual(x)
        
        N = x.shape[0]
        # Reshape logic matches FaRL JIT trace logic
        
        selected_features = []
        for idx in self.output_indices:
            feat = features_list[idx] # LND
            feat = feat[1:, :, :] # Skip class token
            feat = feat.permute(1, 2, 0) # N C L
            feat = feat.reshape(N, 768, 28, 28)
            selected_features.append(feat)
            
        f0 = self.fpns[0](selected_features[0])
        f1 = self.fpns[1](selected_features[1])
        f2 = self.fpns[2](selected_features[2])
        f3 = self.fpns[3](selected_features[3])
        
        return f0, f1, f2, f3

def load_weights(model, jit_path):
    print(f"Loading weights from {jit_path}...")
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    jit_state = jit_model.backbone.state_dict()
    
    my_state = model.state_dict()
    new_state = {}
    
    for k, v in jit_state.items():
        # Mapping
        if "mlp.c_fc" in k:
            k = k.replace("mlp.c_fc", "mlp.0")
        elif "mlp.c_proj" in k:
            k = k.replace("mlp.c_proj", "mlp.2")
            
        if k in my_state:
            if my_state[k].shape == v.shape:
                new_state[k] = v
            else:
                print(f"Shape mismatch for {k}: {my_state[k].shape} vs {v.shape}")
        else:
            print(f"Unmapped key: {k}")
                
    # Load
    keys = model.load_state_dict(new_state, strict=False)
    print(f"Missing keys: {keys.missing_keys}")
    print(f"Unexpected keys: {keys.unexpected_keys}")

if __name__ == "__main__":
    model = FaRLBackbone()
    load_weights(model, r"D:\AI\temp\_internal\ExtractTool\temp\3DDFA-V3\data\torch\hub\checkpoints\face_parsing.farl.celebm.main_ema_181500_jit.pt")
