# fixed camera setup

import pybullet as p
import numpy as np
from PIL import Image
import random
import os
import util as u
from data import H, W

class RandomCameraConfig(object):

    def __init__(self, seed=None):
        if seed is not None:
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
        if seed is not None:
            random.seed()
            np.random.seed()

class Camera(object):

    def __init__(self, camera_id, img_dir, joint_info_file, kuka_uid, fixed_config=None):
        self.id = camera_id
        self.img_dir = img_dir
        if joint_info_file == None:
            self.joint_info_file = None
        else:
            self.joint_info_file = open(joint_info_file, "w")
        self.kuka_uid = kuka_uid
        self.fixed_config = fixed_config
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    def render(self, frame_num):
        # use fixed config (if supplied) otherwise generate
        # a new random one for this render
        if self.fixed_config is None:
            config = RandomCameraConfig()
        else:
            config = self.fixed_config

        proj_matrix = p.computeProjectionMatrixFOV(fov=config.fov,
                                                   aspect=float(config.width) / config.height,
                                                   nearVal=0.1,
                                                   farVal=100.0)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=config.camera_target,
                                                          distance=config.distance,
                                                          yaw=config.yaw,
                                                          pitch=config.pitch,
                                                          roll=0,  # varying this does nothing (?)
                                                          upAxisIndex=2)

        # call bullet to render
        rendering = p.getCameraImage(width=config.width, height=config.height,
                                     viewMatrix=view_matrix,
                                     projectionMatrix=proj_matrix,
                                     lightColor=config.light_color,
                                     lightDirection=config.light_direction,
                                     shadow=1,
                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # convert RGB to PIL image
        rgb_array = np.array(rendering[2], dtype=np.uint8)
        rgb_array = rgb_array.reshape((config.height, config.width, 4))
        rgb_array = rgb_array[:, :, :3]
        img = Image.fromarray(rgb_array)

        # save image
        output_fname = "%s/%s" % (self.img_dir, u.frame_filename_format(frame_num))
        print("output_fname", output_fname)
        img.save(output_fname)

        # capture joint states
        if self.joint_info_file is not None:
            joint_info_output = [frame_num]
            for j in range(p.getNumJoints(self.kuka_uid)):
                # output just position (0th) element, ignore velocity, torques etc
                joint_info_output.append(p.getJointState(self.kuka_uid, j)[0])
            print("\t".join(map(str, joint_info_output)), file=self.joint_info_file)
