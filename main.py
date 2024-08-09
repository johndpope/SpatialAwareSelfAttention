import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAwareSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.qkv_h = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_l = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Window partitioning
        x_windows = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        
        # High frequency path
        qkv_h = self.qkv_h(x_windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        
        # Low frequency path
        x_avg = F.avg_pool2d(x.permute(0, 3, 1, 2), self.window_size).permute(0, 2, 3, 1)
        x_avg = x_avg.view(-1, 1, C)
        qkv_l = self.qkv_l(x_avg).reshape(-1, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_l, k_l, v_l = qkv_l[0], qkv_l[1], qkv_l[2]
        
        # Attention
        attn_h = (q_h @ k_h.transpose(-2, -1)) * (self.dim ** -0.5)
        attn_h = attn_h.softmax(dim=-1)
        x_h = (attn_h @ v_h).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        
        attn_l = (q_l @ k_l.transpose(-2, -1)) * (self.dim ** -0.5)
        attn_l = attn_l.softmax(dim=-1)
        x_l = (attn_l @ v_l).transpose(1, 2).reshape(-1, 1, C)
        
        # Concatenate and project
        x_out = torch.cat([x_h, x_l.expand(-1, self.window_size * self.window_size, -1)], dim=-1)
        x_out = self.proj(x_out)
        
        # Reshape back
        x_out = x_out.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x_out = x_out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        
        return x_out

class ChannelAwareSelfAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, H, W, C = x.shape
        y = torch.mean(x, dim=(1, 2))  # Global average pooling
        y = self.fc(y).view(B, 1, 1, C)
        return x * y.expand_as(x)

class MLGFFN(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv3x3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1, groups=hidden_dim // 2)
        self.dwconv5x5 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2, groups=hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.act(self.dwconv3x3(x1))
        x2 = self.act(self.dwconv5x5(x2))
        
        x_local = torch.cat([x1, x2], dim=1)
        x_global = F.adaptive_avg_pool2d(x, (1, 1)).expand_as(x)
        
        x = torch.cat([x_local, x_global], dim=1)
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.fc2(x)
        return x

class HSCATB(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAwareSelfAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.casa = ChannelAwareSelfAttention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlgffn = MLGFFN(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.casa(self.norm2(x))
        x = x + self.mlgffn(self.norm3(x))
        return x

# Example usage
if __name__ == "__main__":
    B, H, W, C = 1, 64, 64, 256
    x = torch.randn(B, H, W, C)
    model = HSCATB(dim=C)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
