#!/usr/bin/env python3

import argparse
import camera
import math
import numpy as np
import pybullet as p
import random
import time
import util as u
import kuka_env

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img-dir', type=str, default='imgs', help='based dir for output')
parser.add_argument('--run', type=int, default=1, help='run_id for img saving')
parser.add_argument('--run-offset', type=int, default=0, help='offset to add to --run')
parser.add_argument('--num-cameras', type=int, default=20, help='number of cameras')
parser.add_argument('--num-objects', type=int, default=10, help='number of objects in tray')
parser.add_argument('--obj-urdf-dir', type=str, default='./objs', help='base dir for procedural objs')
parser.add_argument('--num-grasps', type=int, default=10, help='number of random grasps to execute')
parser.add_argument('--render-freq', type=int, default=50, help='how often (sim steps) to render a frame')
parser.add_argument('--gui', action='store_true', help='if set, run with bullet explorer gui')
opts = parser.parse_args()
print("opts", opts)

# TODO: capture joint positions at render time too (pull in old grasping prj protobuf?)

kuka_env = kuka_env.KukaEnv(gui=opts.gui,
                            render_freq=opts.render_freq,
                            run_id=opts.run+opts.run_offset,
                            num_objects=opts.num_objects,
                            obj_urdf_dir=opts.obj_urdf_dir)


if False: # PERTURBED_CAMERA_HACKTASTIC_HACK
    import copy
    base_camera_config = camera.CameraConfig(seed=0)
    configs = [base_camera_config]
    with open("perturbed_cameras.ssv", "w") as f:
        for i in range(90):
            extra_camera_config = copy.deepcopy(base_camera_config)
            camera_target_offset = (np.random.random((3,))*0.4)-0.2
            print(i+1, " ".join(map(str, list(camera_target_offset))),
                  np.linalg.norm(camera_target_offset), file=f)
            extra_camera_config.camera_target += camera_target_offset
            configs.append(extra_camera_config)
else:
    configs = [camera.CameraConfig(seed=i) for i in range(opts.num_cameras)]

kuka_env.cameras = [camera.Camera(camera_id=i, config=c, img_dir=opts.img_dir)
                    for i, c in enumerate(configs)]



# do some grasps
for _ in range(opts.num_grasps):
    
    # pick random position above tray
    x = u.random_in(0.5, 0.7)
    y = u.random_in(-0.15, 0.25)
    z = 0.3
    pos_gripper = [x, y, z]
    
    # pick random orientation of gripper (pointing down, random yaw)
    random_yaw = u.random_in(0, math.pi/2)
    orient_gripper = p.getQuaternionFromEuler([0, -math.pi, random_yaw])

    # move arm to this random starting position, with gripper open
    success = kuka_env.move_arm_to_pose(desired_pos_gripper=pos_gripper,
                                        desired_orient_gripper=orient_gripper,
                                        desired_finger_angle=0.3)

    # move down into tray
    pos_gripper[2] = 0.1
    success = kuka_env.move_arm_to_pose(desired_pos_gripper=pos_gripper,
                                        desired_orient_gripper=orient_gripper,
                                        desired_finger_angle=0.3)

    # grasp!
    success = kuka_env.move_arm_to_pose(desired_pos_gripper=pos_gripper,
                                        desired_orient_gripper=orient_gripper,
                                        desired_finger_angle=0, max_steps=100)

    # lift!
    pos_gripper[2] = 0.3
    success = kuka_env.move_arm_to_pose(desired_pos_gripper=pos_gripper,
                                        desired_orient_gripper=orient_gripper,
                                        desired_finger_angle=0)
    
p.disconnect()
