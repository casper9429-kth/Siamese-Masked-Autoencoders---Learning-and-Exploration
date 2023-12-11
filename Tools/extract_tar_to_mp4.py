import os
import tarfile

# Define the paths
tar_folder = "data/Kinetics/tars/train/"
mp4_folder = "data/Kinetics/train_mp4"

# Create the mp4 folder if it doesn't exist
os.makedirs(mp4_folder, exist_ok=True)

# Iterate over the tar files
for file_name in os.listdir(tar_folder):
    if file_name.endswith(".tar.gz"):
        tar_path = os.path.join(tar_folder, file_name)
        mp4_path = mp4_folder#os.path.join(mp4_folder, file_name[:-7])
        
        # Extract the tar file
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(mp4_path)
        
        # Remove the tar file
        os.remove(tar_path)
