import torch
from torch import nn, Tensor
from .layers import DropPath, trunc_normal_


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, drop=0.) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class CMLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, drop=0.) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv3d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, drop=0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, dpr=0., drop=0.):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, 1, 2, groups=dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = CMLP(dim, int(dim*4), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x))))) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, dpr=0., drop=0.) -> None:
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, drop)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, embed_dim=768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        if in_ch == 3:
            self.proj = nn.Conv3d(in_ch, embed_dim, (3, patch_size, patch_size), (2, patch_size, patch_size), (1, 0, 0))
        else:
            self.proj = nn.Conv3d(in_ch, embed_dim, (1, patch_size, patch_size), (1, patch_size, patch_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


uniformer_settings = {
    'S': [[3, 4, 8, 3], 0.1, 0.0],       # [depth]
    'B': [[5, 8, 20, 7], 0.3, 0.1]
}


class UniFormer(nn.Module):     
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 400) -> None:
        super().__init__()
        assert model_name in uniformer_settings.keys(), f"UniFormer model name should be in {list(uniformer_settings.keys())}"
        depth, dropout_rate, drop_path_rate = uniformer_settings[model_name]

        head_dim = 64
        embed_dims = [64, 128, 320, 512]
    
        for i in range(4):
            self.add_module(f"patch_embed{i+1}", PatchEmbed(4 if i == 0 else 2, 3 if i == 0 else embed_dims[i-1], embed_dims[i]))

        self.pos_drop = nn.Dropout(dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dims]

        self.blocks1 = nn.ModuleList([
            CBlock(embed_dims[0], dpr[i], dropout_rate)
        for i in range(depth[0])])

        self.blocks2 = nn.ModuleList([
            CBlock(embed_dims[1], dpr[i+depth[0]], dropout_rate)
        for i in range(depth[1])])

        self.blocks3 = nn.ModuleList([
            SABlock(embed_dims[2], num_heads[2], dpr[i+depth[0]+depth[1]], dropout_rate)
        for i in range(depth[2])])

        self.blocks4 = nn.ModuleList([
            SABlock(embed_dims[3], num_heads[3], dpr[i+depth[0]+depth[1]+depth[2]], dropout_rate)
        for i in range(depth[3])])

        self.norm = nn.BatchNorm3d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights(pretrained)

    def _inflate_weight(self, weight_2d, time_dim):
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
        return weight_3d

    def _init_weights(self, pretrained: str = None) -> None:
        # Load ImageNet1k weights
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')['model']
            state_dict_3d = self.state_dict()

            del checkpoint['head.weight']
            del checkpoint['head.bias']

            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    checkpoint[k] = self._inflate_weight(checkpoint[k], state_dict_3d[k].shape[2])

            self.load_state_dict(checkpoint, strict=False)
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                
        
    def forward(self, x: torch.Tensor):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)

        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x.flatten(2).mean(-1))
        return x

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model = UniFormer('S', 'C:\\Users\\sithu\\Documents\\weights\\backbones\\uniformer\\uniformer_small_in1k.pth')
    model.load_state_dict(torch.load('C:\\Users\\sithu\\Documents\\weights\\videocls\\uniformer\\uniformer_small_k400_8x8.pth', map_location='cpu'))
    model.eval()
    x = torch.randn(1, 3, 8, 224, 224)
    y = model(x)
    print(y.shape)