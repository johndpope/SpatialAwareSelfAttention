import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SpatialAwareSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = (window_size, window_size)
        
        self.window_attn = WindowAttention(dim, self.window_size, num_heads)
        self.global_attn = WindowAttention(dim, (1, 1), num_heads)
        
    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        
        # Window partitioning
        x_windows = self.window_partition(x)
        
        # Window attention
        x_windows = self.window_attn(x_windows)
        
        # Merge windows
        x = self.window_reverse(x_windows, H, W)
        
        # Global attention
        x_global = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (1, 1)).permute(0, 2, 3, 1)
        x_global = self.global_attn(x_global.view(B, 1, C)).view(B, 1, 1, C)
        
        # Combine local and global attention
        x = x + x_global.expand_as(x)
        
        return x
    
    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size[0] * self.window_size[1], C)
        return windows

    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / self.window_size[0] / self.window_size[1]))
        x = windows.view(B, H // self.window_size[0], W // self.window_size[1], self.window_size[0], self.window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

class ChannelAwareSelfAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        y = torch.mean(x, dim=(1, 2))  # Global average pooling
        y = self.fc(y).view(B, 1, 1, C)
        return x * y.expand_as(x)

class MLGFFN(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv3x3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1, groups=hidden_dim // 2)
        self.dwconv5x5 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2, groups=hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAwareSelfAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.casa = ChannelAwareSelfAttention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlgffn = MLGFFN(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.casa(self.norm2(x))
        x = x + self.mlgffn(self.norm3(x))
        return x

class ImageCompressor(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 256, num_blocks: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([HSCATB(dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).permute(0, 2, 3, 1)  # B, H, W, C
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = self.head(x)
        return x

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W, C = 1, 256, 256, 3
    x = torch.randn(B, C, H, W).to(device)
    model = ImageCompressor().to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
