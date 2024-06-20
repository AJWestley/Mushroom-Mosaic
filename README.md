# Mushroom Mosaic

A fun little program that I made as a gift to my girlfriend who loves mushrooms <3

### Usage

#### Text Mode

To use the app in text mode, simply run the `mosaic.py` file as you would any Python script.
You will then be shown a file open window, from there you select the file you want to "Mushroom-ify".
The program will do its thing, then when the "Saving..." prompt appears, you will see another file dialog.
Now just choose where to save the image and presto!

### Algorithms and Techniques

- **K-Means** - For clustering the pixels to represent the image with fewer colours and assign the mushroom images to their nearest colours.
- **Multiprocessing** - To parallelise the final image generation, which offered a 3-4 times speed-up.

Overall, I reduce the image to a 256x256 image, then I train a K-Means model on the resulting image. After this, I assign the average colours of each muchroom image to a cluster. I use these new average colours to assign each pixel in the 256x256 image a corresponding mushroom. Lastly, I generate the final image in parallel using multithreaded pools.