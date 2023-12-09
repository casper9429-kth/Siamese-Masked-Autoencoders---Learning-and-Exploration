from jax import random
import flax.linen as nn
import jax.numpy as jnp
import omegaconf
from omegaconf import OmegaConf
import jax

import numpy as np
import matplotlib.pyplot as plt

from util.pos_embedding import get_2d_sincos_pos_embed
from util.patchify import patchify, unpatchify

# from timm.models.vision_transformer import Block


class SiamMAE(nn.Module): # For pre training
    """ 
        Siamese Masked Autoencoder with VisionTransformer backbone.
    """
    img_size : int = 48
    patch_size : int = 16
    in_chans : int = 3
    embed_dim : int = 768
    depth : int = 12
    encoder_hidden_dim : int = int(4*768)
    num_heads : int = 16
    decoder_embed_dim : int = 512
    decoder_depth : int = 8
    decoder_hidden_dim : int = int(4*512)
    decoder_num_heads : int = 8
    mask_ratio : float = 0.95
    hparams : OmegaConf = None
    def setup(self):

        # ----------------------------------- Encoder -----------------------------------
        # patch embeddings
        # input: batch of images (n_batch x C x H x W)
        # output: batch of patch embeddings (n_batch x num_patches x embed_dim)
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.get_num_patches()
        self.num_keep = int(num_patches * (1-self.mask_ratio))

        # cls token will be appended to patch embeddings
        self.cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.embed_dim))
        # self.cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.embed_dim))
        # position embeddings will be added to the patch embeddings (we'll use sin-cos-distance)
        batch_size = int(num_patches**.5)
        #self.pos_embed = self.param("pos_embed", self.sincos_pos_embed, (1, num_patches+1, self.embed_dim)) # TODO: no grad!
        self.pos_embed = self.param("frozen_pos_embed", self.sincos_pos_embed, (1, batch_size, self.embed_dim)) # TODO: no grad!

        # plot position embeddings
        pos_embed = self.pos_embed[0, :, :]
        
        


        self.encoder_blocks  = [
            Encoder(self.embed_dim, self.num_heads, self.encoder_hidden_dim) for _ in range(self.depth)
        ]

        self.norm = nn.LayerNorm()

        # ----------------------------------- Decoder -----------------------------------
        # embedding for decoder is just a linear layer applied to the output of the encoder
        self.decoder_embed = nn.Dense(self.decoder_embed_dim)

        self.mask_token = self.param("mask_token", nn.initializers.normal(stddev=0.02), (1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = self.param("frozen_decoder_pos_embed", self.sincos_pos_embed, (1, batch_size, self.decoder_embed_dim))

        self.decoder_blocks = [
            CrossSelfDecoder(self.decoder_embed_dim, self.decoder_num_heads, self.decoder_hidden_dim) for _ in range(self.decoder_depth)
        ]

        self.decoder_norm = nn.LayerNorm()
        self.decoder_pred = nn.Dense(self.patch_size**2 * self.in_chans, kernel_init=nn.initializers.xavier_uniform())

    def sincos_pos_embed(self, key, shape):
        #_, N, embed_dim = shape[0], shape[1], shape[2]
        _, grid_size, embed_dim = shape[0], shape[1], shape[2]
        
        # make static grid_size 
        #grid_size = int((N-1)**.5)
        
        pos_embed = get_2d_sincos_pos_embed(embed_dim,grid_size , cls_token=True)

        return pos_embed[None, :, :]


    def random_mask(self, key, x):
        """
            Mask out patches of the input image given a mask ratio.
        """
        B, N, D = x.shape
        #num_keep = jnp.int32(N * (1-mask_ratio))

        noise = random.uniform(key, shape=(B, N))

        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :self.num_keep]
        x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], axis=1)

        mask = jnp.ones((B, N))
        mask = mask.at[:, :self.num_keep].set(0)

        mask = jnp.take_along_axis(mask, ids_restore, axis=1)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, f1, f2):
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
        key = random.key(12)
        f2, mask, restore_ids = self.random_mask(key, f2)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[0, :1, :]
        cls_token = jnp.tile(cls_token, (f1.shape[0], 1, 1))
        f1 = jnp.concatenate((cls_token, f1), axis=1)
        f2 = jnp.concatenate((cls_token, f2), axis=1)

        # now inputs are ready:
        # apply encoder blocks
        for block in self.encoder_blocks:
            f1 = block(f1)
            f2 = block(f2)   

        f1 = self.norm(f1)
        f2 = self.norm(f2)

        return f1, f2, mask, restore_ids
    

    def forward_decoder(self, x1, x2, ids_restore):
        """
            Forward pass through decoder.
            Expected dimensions for x1 and x2: n_batch x N x D_ec
        """
        # embed encoder outputs (just linear layer)
        x1 = self.decoder_embed(x1) # should the decoder embeddings be different for f1 and f2?
        x2 = self.decoder_embed(x2) # B x N x D_dc

        # add mask tokens to x2
        mask_tokens = jnp.tile(self.mask_token,(x2.shape[0], ids_restore.shape[1] + 1 - x2.shape[1], 1))
        x_ = jnp.concatenate((x2[:, 1:, :], mask_tokens), axis=1)
        x_ = jnp.take_along_axis(x_, jnp.tile(ids_restore[:, :, None], (1, 1, x2.shape[2])), axis=1)
        x2 = jnp.concatenate((x2[:, :1, :], x_), axis=1)

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

    def __call__(self, frames1, frames2):
        """
            Forward pass through the whole network.
        """
        frames1_enc, frames2_enc, mask, ids = self.forward_encoder(frames1, frames2)
        pred = self.forward_decoder(frames1_enc, frames2_enc, ids)

        return pred, mask
    
    def loss(self, frames, pred, mask):
        """
            Calculate the loss.
        """
        target = patchify(frames, self.patch_size)


        # 10x196x768
        
        loss = (pred - target)**2
        loss = jnp.mean(loss, axis=-1)
        # loss = jnp.mean(loss, axis=(0,1))

        loss = (loss * mask).sum() / mask.sum() # calculate loss only of masked patches
        
        return loss


class fine_SiamMAE(nn.Module):
    """ 
        Siamese Masked Autoencoder with VisionTransformer backbone.
    """
    img_size : int = 224
    patch_size : int = 16
    in_chans : int = 3
    embed_dim : int = 1024
    depth : int = 24
    encoder_hidden_dim : int = 1
    num_heads : int = 16
    decoder_embed_dim : int = 512
    decoder_depth : int = 8
    decoder_hidden_dim : int = 1
    decoder_num_heads : int = 16
    mask_ratio : float = 0.95
    hparams : OmegaConf = None
    def setup(self):
        # ----------------------------------- Encoder -----------------------------------
        # patch embeddings
        # input: batch of images (n_batch x C x H x W)
        # output: batch of patch embeddings (n_batch x num_patches x embed_dim)
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.get_num_patches()
        self.num_keep = int(num_patches * (1-self.mask_ratio))

        # cls token will be appended to patch embeddings
        self.cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.embed_dim))
        # self.cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.embed_dim))
        # position embeddings will be added to the patch embeddings (we'll use sin-cos-distance)
        batch_size = int(num_patches**.5)
        #self.pos_embed = self.param("pos_embed", self.sincos_pos_embed, (1, num_patches+1, self.embed_dim)) # TODO: no grad!
        self.pos_embed = self.param("pos_embed", self.sincos_pos_embed, (1, batch_size, self.embed_dim)) # TODO: no grad!


        self.encoder_blocks  = [
            Encoder(self.embed_dim, self.num_heads, self.encoder_hidden_dim) for _ in range(self.depth)
        ]

        self.norm = nn.LayerNorm()

        # ----------------------------------- Decoder -----------------------------------
        # embedding for decoder is just a linear layer applied to the output of the encoder
        self.decoder_embed = nn.Dense(self.decoder_embed_dim)

        self.mask_token = self.param("mask_token", nn.initializers.normal(stddev=0.02), (1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = self.param("decoder_pos_embed", self.sincos_pos_embed, (1, batch_size, self.decoder_embed_dim))

        self.decoder_blocks = [
            CrossSelfDecoder(self.decoder_embed_dim, self.decoder_num_heads, self.decoder_hidden_dim) for _ in range(self.decoder_depth)
        ]

        self.decoder_norm = nn.LayerNorm()
        self.decoder_pred = nn.Dense(self.patch_size**2 * self.in_chans, kernel_init=nn.initializers.xavier_uniform())

    def sincos_pos_embed(self, key, shape):
        #_, N, embed_dim = shape[0], shape[1], shape[2]
        _, grid_size, embed_dim = shape[0], shape[1], shape[2]
        
        # make static grid_size 
        #grid_size = int((N-1)**.5)
        
        pos_embed = get_2d_sincos_pos_embed(embed_dim,grid_size , cls_token=True)
        return pos_embed[None, :, :]


    def random_mask(self, key, x):
        """
            Mask out patches of the input image given a mask ratio.
        """
        B, N, D = x.shape
        #num_keep = jnp.int32(N * (1-mask_ratio))

        noise = random.uniform(key, shape=(B, N))

        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :self.num_keep]
        x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], axis=1)

        mask = jnp.ones((B, N))
        mask = mask.at[:, :self.num_keep].set(0)

        mask = jnp.take_along_axis(mask, ids_restore, axis=1)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, f1, f2):
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
        key = random.key(12)
        f2, mask, restore_ids = self.random_mask(key, f2)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[0, :1, :]
        cls_token = jnp.tile(cls_token, (f1.shape[0], 1, 1))
        f1 = jnp.concatenate((cls_token, f1), axis=1)
        f2 = jnp.concatenate((cls_token, f2), axis=1)

        # now inputs are ready:
        # apply encoder blocks
        for block in self.encoder_blocks:
            f1 = block(f1)
            f2 = block(f2)
        
        

        f1 = self.norm(f1)
        f2 = self.norm(f2)

        return f1, f2, mask, restore_ids
    

    def forward_decoder(self, x1, x2, ids_restore):
        """
            Forward pass through decoder.
            Expected dimensions for x1 and x2: n_batch x N x D_ec
        """
        # embed encoder outputs (just linear layer)
        x1 = self.decoder_embed(x1) # should the decoder embeddings be different for f1 and f2?
        x2 = self.decoder_embed(x2) # B x N x D_dc

        # add mask tokens to x2
        mask_tokens = jnp.tile(self.mask_token,(x2.shape[0], ids_restore.shape[1] + 1 - x2.shape[1], 1))
        x_ = jnp.concatenate((x2[:, 1:, :], mask_tokens), axis=1)
        x_ = jnp.take_along_axis(x_, jnp.tile(ids_restore[:, :, None], (1, 1, x2.shape[2])), axis=1)
        x2 = jnp.concatenate((x2[:, :1, :], x_), axis=1)

        # add position embeddings (just to x2? if not, should they be different?)
        # x1 = x1 + self.decoder_pos_embed
        x2 = x2 + self.decoder_pos_embed
        
        # apply decoder
        for block in self.decoder_blocks:
            x2 = block(x1, x2)

        x = self.decoder_norm(x2)
        pred = self.decoder_pred(x)

        # remove cls token
        pred = pred[:, 1:, :]

        return pred    

    def __call__(self, frames1, frames2, mask_ratio):
        """
            Forward pass through the whole network.
        """
        frames1_enc, frames2_enc, mask, ids = self.forward_encoder(frames1, frames2)
        pred = self.forward_decoder(frames1_enc, frames2_enc, ids)

        return pred, mask
    
    def loss(self, frames, pred, mask):
        """
            Calculate the loss.
        """
        target = patchify(frames, self.patch_size)

        loss = (pred - target)**2
        loss = jnp.mean(loss, axis=-1)

        loss = (loss * mask).sum() / mask.sum() # calculate loss only of masked patches
        
        return loss


class PatchEmbed(nn.Module):
    """
        Image to Patch Embedding.
    """
    img_size : int = 224
    patch_size : int = 16
    in_chans : int = 3
    embed_dim: int = 768

    def setup(self):
        # num_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size)
        # self.num_patches = num_patches

        self.proj = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform())

    def get_num_patches(self):
        return (self.img_size // self.patch_size) * (self.img_size // self.patch_size)

    def __call__(self, x, train=True):
        B, C, H, W = x.shape

        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # patchify the image
        x = patchify(x, self.patch_size)

        # apply linear layer for embedding the image
        x = self.proj(x)

        return x


class Encoder(nn.Module):
    """
        Transformer encoder block.
    """
    dim : int
    num_heads : int
    hidden_dim : float
    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,use_bias=True, kernel_init=nn.initializers.xavier_uniform()) # Attention(self.dim, self.num_heads)
        self.norm_1 = nn.LayerNorm()
        self.norm_2 = nn.LayerNorm()
        self.linear = [
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform()),
            nn.gelu,
            nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())
        ]

    def __call__(self, x, train=True):
        x = x + self.attention(inputs_q=self.norm_1(x), inputs_kv=self.norm_1(x))
        linear_out = self.norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out)
        x = x + linear_out
        return x


class CrossSelfDecoder(nn.Module):
    """
        Cross-self decoder block.
    """
    
    dim : int
    num_heads : int
    hidden_dim : int
    def setup(self):
        self.cross_attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, use_bias=True, kernel_init=nn.initializers.xavier_uniform())
        self.attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, use_bias=True, kernel_init=nn.initializers.xavier_uniform())
        self.norm_1 = nn.LayerNorm()
        self.norm_2 = nn.LayerNorm()
        self.linear = [
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform()),
            nn.gelu,
            nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())
        ]

    def __call__(self, x1, x2):
        x = x2 + self.cross_attention(inputs_q=x2, inputs_kv=x1)
        norm_x = self.norm_1(x)
        x = x + self.attention(inputs_q=norm_x, inputs_kv=norm_x)
        norm_x = self.norm_2(x)
        linear_out = norm_x
        for l in self.linear:
            linear_out = l(linear_out)
        x = x + linear_out
        return x

def save_pred_img(pred, name, do_unpatch=True):
    if pred.shape[0] > 1:
        pred = jnp.array([pred[0]])
    out_img = pred
    if do_unpatch:
        out_img = unpatchify(pred)
    out_img = jnp.einsum('ijkl->klj', out_img)
    # Minmax normalize to range 0-255
    out_img = (out_img - out_img.min()) * (255/(out_img.max() - out_img.min()))
    # Convert to uint8
    out_img = out_img.astype(np.uint8)
    out_img = np.array(out_img)
    # Save output image
    plt.imsave('./reproduction/{}'.format(name), out_img)
    print("Saved {}!".format(name))

def main():
    model = SiamMAE(embed_dim=8, encoder_hidden_dim=3072)
    example_batch =  jnp.ones((2,3,48,48))

    # name = "example_0.png"
    # out_img = example_batch[0].astype(np.uint8)
    # out_img = jnp.einsum('jkl->klj', out_img)
    # out_img = np.array(out_img)
    # # Save output image
    # plt.imsave('./reproduction/{}'.format(name), out_img)
    # print("Saved {}!".format(name))

    rng = jax.random.PRNGKey(42)
    params = model.init(rng, example_batch, example_batch)

    pred, mask = model.apply(params, example_batch, example_batch)

    name = "example_pred0.png"
    save_pred_img(pred, name)




if __name__ == "__main__":
    main()
