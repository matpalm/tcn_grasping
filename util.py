import random
from PIL import Image, ImageDraw

# TODO: pull in filename formatting cNN/rNN/RNNN.png util here

def random_in(a, b):
    if a > b:
        a, b = b, a        
    return (random.random()*(b-a))+a

def load_image_with_caption(fname):
    img = Image.open(fname)
    canvas = ImageDraw.Draw(img)
    canvas.rectangle((0,0,160,10), fill='black')
    canvas.text((0,0), fname)
    return img
    
