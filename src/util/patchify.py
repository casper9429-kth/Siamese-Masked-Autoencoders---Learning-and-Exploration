import jax.numpy as jnp

def patchify(x, patch_size):
        """
            Patchify a batch of images.
        """
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.transpose(0, 2, 4, 3, 5, 1)    # [B, H', W', p_H, p_W, C]
        x = x.reshape(B, -1, *x.shape[3:])   # [B, H'*W', p_H, p_W, C]
        x = x.reshape(B, x.shape[1], -1) # [B, H'*W', p_H*p_W*C]

        return x