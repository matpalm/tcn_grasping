import random
from PIL import Image, ImageDraw
import os

# TODO: pull in filename formatting cNN/rNN/RNNN.png util here

def random_in(a, b):
    if a > b:
        a, b = b, a        
    return (random.random()*(b-a))+a

def load_image_with_caption(fname):
    img = Image.open(fname)
    canvas = ImageDraw.Draw(img)
    canvas.rectangle((0,0,300,10), fill='black')
    canvas.text((0,0), fname)
    return img

def slurp_manifest(manifest):
    return map(str.strip, open(manifest, "r").readlines())

def slurp_manifest_as_idx_to_name_dict(manifest):
    return {i: f for i, f in enumerate(slurp_manifest(manifest))}

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
