# fixed camera setup

import pybullet as p
import numpy as np
from PIL import Image
import random
import os
import util as u

class Camera(object):
    
    def __init__(self, camera_id, seed, img_dir, width=320, height=240):
        self.id = camera_id
        self.img_dir = img_dir
        self.width = width
        self.height = height
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=u.random_in(50, 70),
                                                        aspect=float(self.width) / self.height,
                                                        nearVal=0.1,
                                                        farVal=100.0)

        # focus a bit above center of tray
        # (+ bit of noise)
        self.camera_target = np.array([0.6, 0.1, 0.1])
        self.camera_target += (np.random.random((3,))*0.2)-0.1
        self.camera_target = list(self.camera_target)
                               
        self.distance = 1.0 + (random.random()*0.3)

        # yaw=0 => left hand side, =90 towards arm, =180 from right hand side
        self.yaw = u.random_in(45, 135)   # (-40, 220)  # very wide

        # pitch=0 looking horizontal, we pick a value looking slightly down
        self.pitch = u.random_in(-50, -10)

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.camera_target,
                                                               distance=self.distance,
                                                               yaw=self.yaw, 
                                                               pitch=self.pitch,
                                                               roll=0,  # varying this does nothing (?)
                                                               upAxisIndex=2)

        self.light_color = [u.random_in(0.1, 1.0),
                            u.random_in(0.1, 1.0),
                            u.random_in(0.1, 1.0)]        
        self.light_direction = random.choice([[1,1,1], [0,1,1], [1,0,1], [1,1,0]])

        # reseed RNG
        random.seed()
        np.random.seed()

    def render(self, run_id, frame_num):
        
        # call bullet to render
        rendering = p.getCameraImage(width=self.width, height=self.height,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     lightColor=self.light_color,
                                     lightDirection=self.light_direction,
                                     shadow=1,
                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # convert RGB to PIL image
        rgb_array = np.array(rendering[2], dtype=np.uint8)
        rgb_array = rgb_array.reshape((self.height, self.width, 4))
        rgb_array = rgb_array[:, :, :3]
        img = Image.fromarray(rgb_array)

        # save image
        img_output_dir = "%s/c%02d/r%02d" % (self.img_dir, self.id, run_id)
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
        output_fname = "%s/f%03d.png" % (img_output_dir, frame_num)
        print("output_fname", output_fname)
        img.save(output_fname)




