import random
from PIL import Image, ImageDraw
import os

# TODO: pull in filename formatting cNN/rNN/RNNN.png util here

def camera_img_fname(camera_id, run_id, frame_id):
    return "c%03d/r%03d/f%04d.png" % (camera_id, run_id, frame_id)

def random_in(a, b):
    if a > b:
        a, b = b, a
    return (random.random()*(b-a))+a

def load_image_with_caption(fname):
    img = Image.open(fname)
    canvas = ImageDraw.Draw(img)
    canvas.rectangle((0,0,300,10), fill='black')
    canvas.text((0,0), fname[-20:])
    return img

def slurp_manifest(manifest):
    return map(str.strip, open(manifest, "r").readlines())

def slurp_manifest_as_idx_to_name_dict(manifest):
    return {i: f for i, f in enumerate(slurp_manifest(manifest))}

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
