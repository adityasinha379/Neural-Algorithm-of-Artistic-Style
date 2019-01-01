import imageio
import numpy as np

# Define some simple helper functions

# Load image and add dimension so CNN can accept it
def load_image(path):
    img = imageio.imread(path)
    img = img.astype('float64')
    img = img.reshape((1,)+img.shape)
    return img

# Change the image to uint8 and save
def imsave(path, img):
    img = np.squeeze(img)
    img = np.clip(img, 0, 255).astype('uint8')
    imageio.imwrite(path, img)