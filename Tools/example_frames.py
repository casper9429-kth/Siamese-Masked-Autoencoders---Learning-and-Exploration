import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def display_frame_patches(frame_folder, mask_ratio, patch_size=32):
    # Read the frames
    frame1 = cv2.imread(frame_folder + 'frame_70.jpg')
    frame2 = cv2.imread(frame_folder + 'frame_100.jpg')

    # Resize the frames to (224, 224)
    frame1 = cv2.resize(frame1, (224, 224))
    frame2 = cv2.resize(frame2, (224, 224))

    # Convert to RGB
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # save the frames
    cv2.imwrite('./reproduction/frame1.jpg', frame1)
    cv2.imwrite('./reproduction/frame2.jpg', frame2)

    # Divide the frames into 16x16 patches
    patches1 = np.array([frame1[x:x+patch_size, y:y+patch_size] for x in range(0, 224, patch_size) for y in range(0, 224, patch_size)])
    patches2 = np.array([frame2[x:x+patch_size, y:y+patch_size] for x in range(0, 224, patch_size) for y in range(0, 224, patch_size)])

    frames = [patches1, patches2]

    N = 224//patch_size

    for _ in range(2):
        for n, patches in enumerate(frames):
            # Display all patches in a grid so that they are in order of the image
            fig, ax = plt.subplots(N, N, figsize=(10, 10))
            for i in range(N):
                for j in range(N):
                    if random.uniform(0, 1) < mask_ratio:
                        ax[i, j].imshow(np.ones((patch_size, patch_size, 3))*0.75, cmap='gray')
                    else:
                        ax[i, j].imshow(patches[i*N+j])
                    ax[i, j].axis('off')
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.savefig('./reproduction/frame{}_{}'.format(n+1,str(mask_ratio).split(".")[1]), bbox_inches='tight', pad_inches=0)

        mask_ratio = 0.0

if __name__ == '__main__':
    frame_folder = "/Users/magnusrubentibbe/Dropbox/Magnus_Ruben_TIBBE/Uni/Master_KTH/Semester3/DeepLearningAdv/Project/Siamese-Masked-Autoencoders---Learning-and-Exploration/data/Kinetics/train_jpg/-L-cr_aSSSE_000002_000012/"
    mask_ratio = 0.9
    display_frame_patches(frame_folder, mask_ratio)
