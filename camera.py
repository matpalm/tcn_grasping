# fixed camera setup

import pybullet as p
import numpy as np
from PIL import Image
import random
import os
import util as u
from data import H, W

class CameraConfig(object):

    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)

        # TODO: push config of these up
        self.width = W
        self.height = H

        self.fov = u.random_in(50, 70)

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

        self.light_color = [u.random_in(0.1, 1.0),
                            u.random_in(0.1, 1.0),
                            u.random_in(0.1, 1.0)]

        self.light_direction = random.choice([[1,1,1], [0,1,1], [1,0,1], [1,1,0]])

        # reseed RNG
        random.seed()
        np.random.seed()

class Camera(object):

    def __init__(self, camera_id, config, img_dir, kuka_uid):
        self.id = camera_id
        self.img_dir = img_dir
        self.kuka_uid = kuka_uid
        self.config = config

        self.proj_matrix = p.computeProjectionMatrixFOV(fov=config.fov,
                                                        aspect=float(config.width) / config.height,
                                                        nearVal=0.1,
                                                        farVal=100.0)

        self.update_view_matrix(config.camera_target, config.distance,
                                config.yaw, config.pitch)


    def update_view_matrix(self, camera_target, distance, yaw, pitch):
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_target,
                                                               distance=distance,
                                                               yaw=yaw,
                                                               pitch=pitch,
                                                               roll=0,  # varying this does nothing (?)
                                                               upAxisIndex=2)

    def render(self, run_id, frame_num):

        # call bullet to render
        rendering = p.getCameraImage(width=self.config.width, height=self.config.height,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     lightColor=self.config.light_color,
                                     lightDirection=self.config.light_direction,
                                     shadow=1,
                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # convert RGB to PIL image
        rgb_array = np.array(rendering[2], dtype=np.uint8)
        rgb_array = rgb_array.reshape((self.config.height, self.config.width, 4))
        rgb_array = rgb_array[:, :, :3]
        img = Image.fromarray(rgb_array)

        # save image
        output_fname = u.camera_img_fname(self.id, run_id, frame_num)
        full_output_fname = "%s/%s" % (self.img_dir, output_fname)
        dir_name = os.path.dirname(full_output_fname)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        print("full_output_fname", full_output_fname)
        img.save(full_output_fname)

        # capture joint states
        # TODO: write this to a more sensible place!
        joint_states = []
        for j in range(p.getNumJoints(self.kuka_uid)):
            joint_states.append(p.getJointState(self.kuka_uid, j)[0])  # just position
        print("\t".join(map(str, ["J", self.id, run_id, frame_num] + joint_states)))
