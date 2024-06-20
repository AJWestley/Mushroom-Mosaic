import json
from multiprocessing import Pool
from functools import partial
from numpy.random import choice
from sklearn.cluster import MiniBatchKMeans
from tkinter import filedialog
from PIL import Image
from utilities import *
from image_utils import *

NUM_CLUSTERS = 50
TEMP_FILE = 'temp.png'

def create_mosaic(disable_logs = True) -> None:
    '''Generates a mosaic'''
    
    source_image, x, y = load_image()
    clustered_image, kmeans = run_pixel_clustering(source_image, x, y, disable_logs)
    image_map = create_image_map('./images', kmeans, disable_logs)
    final_image = construct_image(clustered_image, image_map, disable_logs)
    save_image(final_image, disable_logs)

def find_nearest(pixel: tuple, img_map: dict) -> tuple:
    '''Finds the nearest colour in a map to the given pixel colour'''
    
    nearest_dist = 1000000
    nearest = (0, 0, 0)
    for colour in img_map:
        colour = tuple(map(int, colour))
        pixel = tuple(map(int, pixel))
        dist = np.sqrt((pixel[0] - colour[0])**2 + (pixel[1] - colour[1])**2 + (pixel[2] - colour[2])**2)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = colour
    return nearest

def create_image_map(folder: str, kmeans: MiniBatchKMeans, disable_logs: bool = True) -> dict:
    '''Uses a trained k-means model to cluster the average colour of a set of images, then puts them in a dictionary'''
    
    with open('image_mappings.json', 'r') as json_file:
        json_data = json.load(json_file)
        img_map: dict = {}
        for avg_colour in loading_bar(json_data, 'Finding Suitable Images...', disable_logs):
            colour = np.array(tuple(map(int, avg_colour.split('|'))))
            predicted = kmeans.cluster_centers_[kmeans.predict(colour.reshape((1, len(colour))))].astype('uint8')
            predicted = tuple(map(tuple, predicted))[0]
            img_map.setdefault(predicted, [])
            img_map[predicted].extend(json_data[avg_colour])
    return img_map

def construct_image(source_image: np.ndarray, image_map: dict, disable_logs: bool = True):
    '''Creates a mosaic of a given image from a set of given image pieces'''
    
    # Build image rows
    x, y, _ = source_image.shape
    
    # Create partial function to generate each row for application in pool.map
    gen_row = partial(create_row, source_image=source_image, y=y, image_map=image_map)
    
    # Multithread the image generation
    with Pool() as pool:
        rows = pool.map(gen_row, loading_bar(range(x), 'Building Image...', disable_logs))
        
        println('Finishing Up...')
        final_image = np.vstack(rows)
    
    return final_image

def create_row(r: int, source_image: np.ndarray, y: int, image_map: dict) -> np.ndarray:
    ''' Generates a single row of the image '''
    pixel = tuple(source_image[r, 0])
    if (pixel not in image_map):
        pixel = find_nearest(pixel, image_map)
    selected_path = choice(image_map[pixel])
    row = iio.imread(selected_path)
    for c in range(1, y):
        pixel = tuple(source_image[r, c])
        if (pixel not in image_map):
            pixel = find_nearest(pixel, image_map)
        selected_path = choice(image_map[pixel])
        selected_image = iio.imread(selected_path)
        row = np.hstack((row, selected_image))
    return row

def load_image() -> tuple:
    '''Loads an image from a file dialog and returns a flattened (n x 3) numpy array'''
    
    path = filedialog.askopenfilename(filetypes=[('Image files', '.png .jpg .jpeg .PNG .JPG .JPEG .thumbnail')])
    create_thumbnail(path, TEMP_FILE)
    image, x, y = read_image_flat(TEMP_FILE)
    os.remove(TEMP_FILE)
    
    return image, x, y

def save_image(image: np.ndarray, disable_logs: bool = True) -> None:
    '''Saves a given numpy array as a PNG with the a file dialog'''
    
    println('Saving Image...', disable_logs)
    path = filedialog.asksaveasfilename(filetypes=[('Image files', '.png .PNG')], defaultextension='.png')
    output_image = Image.fromarray(image)
    output_image.save(path)
    
    if not disable_logs:
        print(f'{'Done':<100}')

def run_pixel_clustering(image: np.ndarray, width: int, height: int, disable_logs: bool = True) -> tuple:
    '''Runs k-means clustering on a flattened (n x 3) image array, then returns the resulting image and trained model'''
    
    println('\nProcessing Image...', disable_logs)
    kmeans = MiniBatchKMeans(NUM_CLUSTERS)
    kmeans.fit(image)
    clustered_image = kmeans.cluster_centers_[kmeans.predict(image)]
    clustered_image = clustered_image.reshape((width, height, 3)).astype('uint8')
    return clustered_image, kmeans

if __name__ == '__main__':
    create_mosaic(disable_logs=False)