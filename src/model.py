import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block



class SiamMAE(nn.Module):
    """ 
        Siamese Masked Autoencoder with VisionTransformer backbone.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 norm_layer=nn.LayerNorm) -> None:
        super().__init__()

        # ----------------------------------- Encoder -----------------------------------
        # patch embeddings
        # input: batch of images (n_natch x C x H x W)
        # output: batch of patch embeddings (n_batch x num_patches x embed_dim)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # cls token will be appended to patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # position embeddings will be added to the patch embeddings (we'll use sin-cos-distance)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)

        self.encoder_blocks  = nn.ModuleList([
            Encoder(embed_dim, num_heads) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # ----------------------------------- Decoder -----------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            CrossSelfDecoder(decoder_embed_dim, decoder_num_heads) for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)



    def mask(self, x, mask_ratio):
        # TODO
        raise NotImplementedError

    def forward_encoder(self, f1, f2, mask_ratio):
        """
            Forward pass through encoder.
            Expected dimensions for f1 and f2: n_batch x C x H x W
        """
        # patch embeddings
        f1 = self.patch_embed(f1) # n_batch x N x embed_dim
        f2 = self.patch_embed(f2)

        f1 = f1 + self.pos_embed[:, 1:, :]
        f2 = f2 + self.pos_embed[:, 1:, :]

        # mask second frame
        f2, mask, ids = self.mask(f2, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[0, :1, :]
        cls_token = cls_token.expand(f1.shape[0], -1, -1)
        f1 = torch.cat((cls_token, f1), dim=1)
        f2 = torch.cat((cls_token, f2), dim=1)

        # now inputs are ready:
        # apply encoder blocks
        for block in self.encoder_blocks:
            f1 = block(f1)
            f2 = block(f2)

        f1 = self.norm(f1)
        f2 = self.norm(f2)

        return f1, f2, mask, ids
    

    def forward_decoder(self, x1, x2):
        """
            Forward pass through decoder.
            Expected dimensions for x1 and x2: n_batch x N x D
        """
        # embed encoder outputs (just linear layer)
        x1 = self.decoder_embed(x1) # should the decoder embeddings be different?
        x2 = self.decoder_embed(x2)

        # add mask tokens to x2
        # TODO

        # add position embeddings (just to x2? if not, should they be different?)
        x1 = x1 + self.decoder_pos_embed
        x2 = x2 + self.decoder_pos_embed
        
        # apply decoder
        for block in self.decoder_blocks:
            x2 = block(x1, x2)

        x = self.decoder_norm(x2)
        pred = self.decoder_pred(x)

        # remove cls token
        pred = pred[:, 1:, :]

        return pred
    

    def patchify(self, x):
        pass
    

    def forward(self, frames1, frames2, mask_ratio):
        frames1_enc, frames2_enc, mask, ids = self.forward_encoder(frames1, frames2, mask_ratio)
        pred = self.forward_decoder(frames1_enc, frames2_enc)

        return pred, mask
    
    def loss(self, frames, pred, mask):
        """
            Calculate the loss.
        """
        target = self.patchify(frames)

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum() # calculate loss only of masked patches
        
        return loss


class Attention(nn.Module):
    """
        (Multi-head) self-attention layer.
    """
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim*3)

        self.num_heads = num_heads

    def forward(self, x):
        B, N, D = x.shape # B: #batches, N: #patches + 1 (cls token), D: embed_dim
        proj_dim = D // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # shape: B x num_heads x N x proj_dim

        att_scores = q @ k.transpose(2,3) # B x num_heads x N x N
        att_scores_sm = torch.tensor(nn.functional.softmax(att_scores, -1))
        weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x N x N x proj_dim
        sum = weighted_vals.sum(dim=2) # B x num_heads x N x proj_dim
        
        out  = sum.reshape(B, N, D)
        return out

class CrossAttention(nn.Module):
    """
        (Multi-head) cross attention layer.
    """
    def __init__(self, dim, num_heads) -> None:
        super().__init__(dim, num_heads)

        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)

    def forward(self, x1, x2):
        B, N, D = x1.shape # B: #batches, N: #patches + 1 (cls token), D: embed_dim
        proj_dim = D // self.num_heads
        # get queries from x2 (embedded, encoded and masked future frame)
        q = self.q(x2).reshape(B, N, self.num_heads, proj_dim).permute(0, 2, 1, 3)
        # get keys and values from unmasked frame
        kv = self.kv(x1).reshape(B, N, 2, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # from here it's the same as the self-attention
        att_scores = q @ k.transpose(2,3) # B x num_heads x N x N
        att_scores_sm = torch.tensor(nn.functional.softmax(att_scores, -1))
        weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x N x N x proj_dim
        sum = weighted_vals.sum(dim=2) # B x num_heads x N x proj_dim
        
        out  = sum.reshape(B, N, D)
        return out


class MLP(nn.Module):
    """
        Multi-layer perceptron with one hidden layer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Encoder(nn.Module):
    """
        Transformer encoder block.
    """
    def __init__(self, dim, num_heads, hidden_dim, act_layer=nn.GELU) -> None:
        super().__init__()

        self.attention = Attention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim, act_layer)

    def forward(self, x):
        x = x + self.attention(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x


class CrossSelfDecoder(nn.Module):
    """
        Cross-self decoder block.
    """
    def __init__(self, dim, num_heads, hidden_dim, act_layer=nn.GELU) -> None:
        super().__init__()

        self.cross_attention = CrossAttention(dim, num_heads)
        self.attention = Attention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim, act_layer)

    def forward(self, x1, x2):
        x = x2 + self.cross_attention(x1, x2)
        x = self.norm(x) + self.attention(self.norm(x))
        x = self.norm(x) + self.mlp(self.norm(x))
        return x

