import random
from PIL import Image, ImageDraw
import os

class MaxFramesRenderedException(Exception):
    pass

def run_dir_format(run_id):
    return "r%03d" % run_id

def camera_dir_format(camera_id):
    return "c%03d" % camera_id

def frame_filename_format(frame_id):
    return "f%04d.png" % frame_id

def run_camera_frame_filename(run_id, camera_id, frame_id):
    return "/".join([run_dir_format(run_id),
                     camera_dir_format(camera_id),
                     frame_filename_format(frame_id)])

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
