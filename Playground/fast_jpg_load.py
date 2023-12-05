import cv2
import glob
import concurrent.futures
import numpy as np
import time

def load_image(file_path):
    # Read and decode the image using OpenCV
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

def load_images_parallel(file_paths, num_workers=8):
    # Using multithreading to parallelize image loading
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_image, file_paths))

    return images

def main():
    # Replace 'your_image_directory/*.jpg' with the path to your directory containing the JPEG images
    image_directory = './data/Kinetics/train_jpg/*/*'
    file_paths = glob.glob(image_directory)[:2000]  # Adjust the number of images to load

    # Load images in parallel
    t1 = time.time()
    images = load_images_parallel(file_paths)
    
    # Transform the list into a numpy array
    print(f'Loaded {len(images)} images in {time.time() - t1} seconds')
    # 

    # Now 'images' is a list containing the loaded images

if __name__ == '__main__':
    main()
