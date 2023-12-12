# from pretraining import TrainerSiamMAE
from test_loader import TestLoader
from model import SiamMAE
from util.patchify import unpatchify, patchify

from torch.utils.data import DataLoader

from omegaconf import OmegaConf

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# from jax.sharding import PositionalSharding
import orbax.checkpoint



class ReproduceF2():
    def __init__(self, dataloader, model, checkpoint_path):
        self.dataloader = dataloader
        self.model = model
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = orbax_checkpointer.restore(checkpoint_path)
        self.model_params = restored['model']['params']
        self.patch_size = 16

    def prepare_for_plotting(self, x):
        # prepare for plotting
        x_out = jnp.einsum('ijkl->iklj', x)
        # Minmax normalize to range 0-255
        x_out = (x_out - x_out.min()) * (255/(x_out.max() - x_out.min()))
        # Convert to uint8
        x_out = x_out.astype(np.uint8)
        x_out = np.array(x_out)

        return x_out

    def plot_test_video(self, preds, orig_frames, masks,  name):
        """
        Reproduce figure 2 from the paper

        Args:
            preds: [7, N, D]
            orig_frames:  [8, C, H, W]
            masks: [7, N]
            name: name of the output image
        """
        # prepare original frames
        orig_frames = self.prepare_for_plotting(orig_frames)

        # prepare predictions
        pred_out = unpatchify(preds)
        pred_out = self.prepare_for_plotting(pred_out)
        pred_out = jnp.concatenate((orig_frames[0:1,:,:,:], pred_out), axis=0)

        # prepare masked frames
        masked_frames = orig_frames[1:,:,:,:]
        masked_frames_patch = patchify(masked_frames, self.patch_size)
        masked_frames_patch = masked_frames_patch * masks[:,:,None]
        masked_frames = unpatchify(masked_frames_patch)
        masked_frames = self.prepare_for_plotting(masked_frames)
        masked_frames = jnp.concatenate((orig_frames[0:1,:,:,:], masked_frames), axis=0)

        # plot original frames, predictions and masked frames
        self.plot(orig_frames, pred_out, masked_frames, name)


    def plot(self, gt, pred, mask, name):
        """
        Plot a single video.

        Args:
            gt: np.array of shape [8, H, W, C]
            pred: np.array of shape [8, H, W, C]
            mask: np.array of shape [8, H, W, C]
            name: name of the output image
        """
        fig, ax = plt.subplots(3, 8, figsize=(16, 4))
        for i in range(8):
            ax[0, i].imshow(gt[i])
            ax[1, i].imshow(pred[i])
            ax[2, i].imshow(mask[i])
        plt.tight_layout()
        plt.savefig('./reproduction/{}'.format(name))
        print("Saved {}!".format(name))


    def run(self):
        for batch in self.dataloader:
            batch_x = batch[:,:7,:,:,:] # (batch_size, 7, 3, 224, 224)
            batch_y = batch[:,7:,:,:,:]
            B, numsamples_vid, C, H, W = batch_x.shape

            batch_x = jnp.array(batch_x) # shape: [B, numsamples_vid, C, H, W]
            batch_y = jnp.array(batch_y)
            # Reshape to [B*numsamples_vid, C, H, W]
            batch_x_all = jnp.reshape(batch_x,(B*numsamples_vid, C, H, W))
            batch_y_all = jnp.reshape(batch_y,(B*numsamples_vid, C, H, W))
            
            # make prediction
            preds, masks = self.model.apply(self.model_params, batch_x_all, batch_y_all)
            Bn, N, D = preds.shape
            # pred: [B*numsamples_vid,N, D]
            # mask: [B*numsamples_vid,N]
            # Reshape to [B, numsamples_vid,N, D]
            preds = jnp.reshape(preds,(B,numsamples_vid, N, D))
            masks = jnp.reshape(masks,(B,numsamples_vid, N))
            # Get first frame of each video
            first = batch_x[:,0,:,:,:]
            first = first[:,None,:,:,:]
            orig_frames = jnp.concatenate((first, batch_y), axis=1)

            # plot outputs
            for i in range(B):
                name = 'test_video_{}.png'.format(i)
                self.plot_test_video(preds[i], orig_frames[i], masks[i], name)


            

def main():
    model = SiamMAE()
    data = TestLoader()
    dataloader = DataLoader(data, batch_size=2,shuffle=True, num_workers=0)

    checkpoint_path = './checkpoints/_epoch_100/'

    reproduce = ReproduceF2(dataloader, model, checkpoint_path)
    reproduce.run()


if __name__ == '__main__':
    main()