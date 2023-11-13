from jax import random
import flax.linen as nn
import jax
import jax.numpy as jnp
from utils import get_2d_sincos_pos_embed



class Test(nn.Module):
    patch_size : int = 16
    def setup(self) -> None:
        super().setup()
    
        self.input_layer = nn.Dense(1096)

    def __call__(self, x, train=True):
        B, C, H, W = x.shape

        # patchify the image
        x = x.reshape(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        x = x.transpose(0, 2, 4, 3, 5, 1)    # [B, H', W', p_H, p_W, C]
        x = x.reshape(B, -1, *x.shape[3:])   # [B, H'*W', p_H, p_W, C]
        x = x.reshape(B, x.shape[1], -1) # [B, H'*W', p_H*p_W*C]

        # apply linear layer for embedding the image
        x = self.input_layer(x)
        return x
    
class PatchEmbed(nn.Module):
    """
         Image to Patch Embedding
    """
    img_size : int = 224
    patch_size : int = 16
    in_chans : int = 3
    embed_dim: int = 768

    num_patches : int = (img_size // patch_size) * (img_size // patch_size)
    def setup(self):
        # num_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size)
        # self.num_patches = num_patches

        self.proj = nn.Dense(self.embed_dim)

    def __call__(self, x, train=True):
        B, C, H, W = x.shape

        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # patchify the image
        x = x.reshape(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        x = x.transpose(0, 2, 4, 3, 5, 1)    # [B, H', W', p_H, p_W, C]
        x = x.reshape(B, -1, *x.shape[3:])   # [B, H'*W', p_H, p_W, C]
        x = x.reshape(B, x.shape[1], -1) # [B, H'*W', p_H*p_W*C]

        # apply linear layer for embedding the image
        x = self.proj(x)

        return x
    
class SiamMAE(nn.Module):
    """ 
        Siamese Masked Autoencoder with VisionTransformer backbone.
    """
    img_size : int = 224
    patch_size : int = 16
    in_chans : int = 3
    embed_dim : int = 1024
    depth : int = 24
    num_heads : int = 16
    decoder_embed_dim : int = 512
    decocder_depth : int = 8
    decoder_num_heads : int = 16
    def setup(self):
        # ----------------------------------- Encoder -----------------------------------
        # patch embeddings
        # input: batch of images (n_batch x C x H x W)
        # output: batch of patch embeddings (n_batch x num_patches x embed_dim)
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        # cls token will be appended to patch embeddings
        self.cls.token = self.param("cls_token", nn.initializers.normal(stddev=1.0), (1, 1, self.embed_dim))
        # position embeddings will be added to the patch embeddings (we'll use sin-cos-distance)
        self.pos_embed = self.param("pos_embed", self.sincos_pos_embed, (1, num_patches+1, self.embed_dim)) # TODO: no grad!
        print(self.pos_embed)

        self.norm = nn.LayerNorm()
    def test(self):

        # cls token will be appended to patch embeddings
        self.cls.token = self.param("cls_token", nn.initializers.normal(stddev=1.0), (1, 1, self.embed_dim))
        # position embeddings will be added to the patch embeddings (we'll use sin-cos-distance)
        self.pos_embed = self.param("pos_embed", self.sincos_pos_embed, (1, 196+1, self.embed_dim)) # TODO: no grad!
        print(self.pos_embed)


    def sincos_pos_embed(self, shape):
        _, N, embed_dim = shape
        return get_2d_sincos_pos_embed(embed_dim, int((N-1)**.5), cls_token=True)
    
def random_mask(key, x, mask_ratio):
        """
            Mask out patches of the input image given a mask ratio.
        """
        B, N, D = x.shape
        num_keep = int(N * (1-mask_ratio))

        noise = random.uniform(key, shape=(B, N))

        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], axis=1)

        mask = jnp.ones((B, N))
        mask = mask.at[:, :num_keep].set(0)

        mask = jnp.take_along_axis(mask, ids_restore, axis=1)
        
        return x_masked, mask, ids_restore
    
if __name__=="__main__":
    x = jnp.ones((2,3,224,224))
    x = 3*jnp.ones((3, 20, 10))
    key = random.key(23)
    x_mask, mask, ids = random_mask(key, x, mask_ratio=0.9)
    print(x_mask)

    