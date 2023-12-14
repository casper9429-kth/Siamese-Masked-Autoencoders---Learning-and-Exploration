# from pretraining import TrainerSiamMAE
from test_loader import TestLoader
from model import SiamMAE
from util.patchify import unpatchify, patchify

from tqdm import tqdm

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
        masked_frames = jnp.einsum('iklj->ijkl', masked_frames)
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
        fig, ax = plt.subplots(3, 8, figsize=(16, 4), gridspec_kw={'hspace': 0.2, 'wspace': 0})
        for i in range(8):
            ax[0, i].imshow(gt[i])
            ax[0, i].axis('off')
            ax[1, i].imshow(pred[i])
            ax[1, i].axis('off')
            ax[2, i].imshow(mask[i])
            ax[2, i].axis('off')

        # arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='-', linestyle='solid', linewidth=1.5)
        ax[0, 0].annotate("frame 1", xy=(0.5, 1.15), xytext=(0.5, 1.3), textcoords='axes fraction',ha='center', va='center', fontsize=12)
        title = "<------------------------------------------------------------------------------ frame 2 ------------------------------------------------------------------------------>"
        ax[0, 4].annotate(title, xy=(0.5, 1.15), xytext=(0.5, 1.3), textcoords='axes fraction',ha='center', va='center', fontsize=12)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('./reproduction/{}'.format(name), bbox_inches='tight', pad_inches=0)
        print("Saved {}!".format(name))
        
        



    def run(self):
        for batch in tqdm(self.dataloader):
        # while True:
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
            masks = 1 - masks # invert mask
            # Get first frame of each video
            first = batch_x[:,0,:,:,:]
            first = first[:,None,:,:,:]
            orig_frames = jnp.concatenate((first, batch_y), axis=1)

            # plot outputs
            for i in range(B):
                name = 'test_video_{}.png'.format(i)
                self.plot_test_video(preds[i], orig_frames[i], masks[i], name)

            break



def main():
    model = SiamMAE()
    data = TestLoader()
    dataloader = DataLoader(data, batch_size=2,shuffle=True, num_workers=0)

    checkpoint_path = './checkpoints/_epoch_100/'
    checkpoint_path = "checkpoint_latest/_epoch_400_multiframe"

    reproduce = ReproduceF2(dataloader=dataloader, model=model, checkpoint_path=checkpoint_path)
    reproduce.run()


if __name__ == '__main__':
    main()