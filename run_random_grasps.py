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
parser.add_argument('--img-dir', type=str, default='imgs',
                    help='base dir for output. images are save to {img_dir}/rNNN/cNNN/fNNNN.png')
parser.add_argument('--joint-info-dir', type=str, default='joint_infos',
                    help="base dir for joint info. joint info is saved to {joint_info_dir}/rNNN/cNNN.tsv"
                         " None => don't save joint info")
parser.add_argument('--run', type=int, default=1, help='run_id for img saving')
parser.add_argument('--num-cameras', type=int, default=20, help='number of cameras')
parser.add_argument('--fixed-camera-configs', action='store_true', help='if set, have fixed camera configs')
parser.add_argument('--fixed-camera-seed-offset', type=int, default=0,
                    help='offset to add to seed when generating fixed cameras')
parser.add_argument('--num-objects', type=int, default=10, help='number of objects in tray')
parser.add_argument('--obj-urdf-dir', type=str, default='./objs', help='base dir for procedural objs')
parser.add_argument('--max-frames-to-render', type=int, default=100, help='render this many frames before stopping')
parser.add_argument('--render-freq', type=int, default=50, help='how often (sim steps) to render a frame')
parser.add_argument('--gui', action='store_true', help='if set, run with bullet explorer gui')
opts = parser.parse_args()
print("opts", opts)

kuka_env = kuka_env.KukaEnv(gui=opts.gui,
                            render_freq=opts.render_freq,
                            max_frames_to_render=opts.max_frames_to_render,
                            num_objects=opts.num_objects,
                            obj_urdf_dir=opts.obj_urdf_dir)

kuka_env.cameras = []
for i in range(opts.num_cameras):
    camera_img_dir = "/".join([opts.img_dir, u.run_dir_format(opts.run), u.camera_dir_format(i)])

    if opts.fixed_camera_configs:
        fixed_config = camera.RandomCameraConfig(seed=i+opts.fixed_camera_seed_offset)
    else:
        fixed_config = None

    if opts.joint_info_dir is None:
        camera_joint_info_file = None
    else:
        camera_joint_info_file = "/".join([opts.joint_info_dir, u.run_dir_format(opts.run), u.camera_dir_format(i)]) + ".tsv"
        u.ensure_dir_exists_for_file(camera_joint_info_file)

    kuka_env.cameras.append(camera.Camera(camera_id=i,
                                          img_dir=camera_img_dir,
                                          joint_info_file=camera_joint_info_file,
                                          kuka_uid=kuka_env.kuka.kukaUid,
                                          fixed_config=fixed_config))

# do some grasps. stop when we've reached --max-frames-to-render
try:
    while True:

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

except u.MaxFramesRenderedException:
    pass

p.disconnect()
