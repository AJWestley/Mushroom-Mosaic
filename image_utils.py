from PIL import Image
import numpy as np
import imageio.v3 as iio
import os

THUMBNAIL_SIZE = 128

def read_image_flat(filepath: str) -> tuple:
    '''Reads an image to a flattened (n x 3) numpy array'''
    
    im = iio.imread(filepath)
    x, y, _ = im.shape
    return im.reshape((x * y, -1)), x, y

def create_thumbnails(filepaths: list[str], size = (THUMBNAIL_SIZE, THUMBNAIL_SIZE)) -> None:
    '''Creates a set of image thumbnails'''
    
    for infile in filepaths:
        pth = os.path.basename(infile)
        file, _ = os.path.splitext(pth)
        with Image.open(infile) as img:
            img = square_image(img)
            img.thumbnail(size)
            img.save(f"images/{file}.thumbnail", "PNG")

def create_thumbnail(infile: str, outfile: str, size = (256, 256)) -> None:
    '''Creates a single image thumbnail'''
    
    with Image.open(infile) as img:
        img = square_image(img)
        img.thumbnail(size)
        img.save(outfile, "PNG")

def average_colour(filepath: str) -> np.ndarray:
    '''Gets the average colour of an image'''
    
    im, _, _ = read_image_flat(filepath)
    return np.mean(im, axis=0).astype('uint8')

def square_image(img: Image.Image) -> Image.Image:
    '''Crops an image into a square'''
    
    width, height = img.size
    radius = min(width, height) // 2
    center = (width // 2, height // 2)
    box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    img = img.crop(box)
    return img