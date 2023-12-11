import glob
import os

PATH = "./data/Kinetics/train_jpg_test/*"

# Get all folders
folders = glob.glob(PATH)


# Check subfolders of each folder
for folder in folders:
    subfolders = glob.glob(folder + "/*")
    nr_of_subfolders = len(subfolders)
    # if nr_of_subfolders != 300: remove folder
    if nr_of_subfolders != 300:
        print(folder)
        # Remove folder
        os.system("rm -rf " + folder)
        print("Removed folder: " + folder)
    else:
        print("Folder is good: " + folder)