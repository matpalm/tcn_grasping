#!/usr/bin/env python3

from pybullet_envs.bullet import kuka
import argparse
import camera
import math
import numpy as np
import pybullet as p
import pybullet_data
import random
import time
import util as u

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

# shared variables across all rendering
cameras = [camera.Camera(camera_id=i, seed=i,
                         img_dir=opts.img_dir) for i in range(opts.num_cameras)]

if opts.gui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setGravity(0, 0, -9.8)

p.loadURDF(pybullet_data.getDataPath()+"/table/table.urdf", 0.5,0.,-0.82, 0,0,0,1)

kuka = kuka.Kuka(urdfRootPath=pybullet_data.getDataPath()) #, timeStep=1./240)

# record calls to sim step, will take images every 100th
SIM_STEPS = 0
FRAME_NUM = 0

# hand rolled IK move of arm without constraints, dx/dy/dz limits in kuka class
def move_arm_to_pose(desired_pos_gripper, desired_orient_gripper,
                     desired_finger_angle, max_steps=300):
    global SIM_STEPS
    global FRAME_NUM
    
    steps = 0
    
    while True:

        steps += 1
        if steps > max_steps:
            return False
        
        gripper_state = p.getLinkState(kuka.kukaUid, kuka.kukaEndEffectorIndex)
        actual_pos_gripper, actual_orient_gripper = gripper_state[0], gripper_state[1]

        # calculate euclidean distance between desired and actual
        # gripper positions
#        print("desired_pos_gripper", desired_pos_gripper)
#        print("actual_pos_gripper", actual_pos_gripper)
        pos_diff = np.linalg.norm(np.array(desired_pos_gripper)-np.array(actual_pos_gripper))
        
        # qaternions align when their w component is near 1.0
#        print("desired_orient_gripper", desired_orient_gripper)
#        print("actual_orient_gripper", actual_orient_gripper)
        diff_quant = p.getDifferenceQuaternion(desired_orient_gripper, actual_orient_gripper)
        diff_quant_w = diff_quant[3]

        # finger?
        left_side_finger_angle = -p.getJointState(kuka.kukaUid, 8)[0]
        right_side_finger_angle = p.getJointState(kuka.kukaUid, 11)[0]
#        print("left_side_finger_angle", left_side_finger_angle,
#              "right_side_finger_angle", right_side_finger_angle)
        left_side_diff = left_side_finger_angle - desired_finger_angle
        right_side_diff = right_side_finger_angle - desired_finger_angle
        
        # if pos, orient and fingers look good, we are done!
        # NOTE! this will always fail when grasping. TODO better way to represent finger conditions
        if (pos_diff < 0.05 and diff_quant_w > 0.9999 and
            left_side_diff < 0.01 and right_side_diff < 0.01):
            return True
        
        # use IK to calculate target joint positions
        # (ignore null space)
        joint_poses = p.calculateInverseKinematics(kuka.kukaUid, kuka.kukaEndEffectorIndex,
                                                   desired_pos_gripper,
                                                   desired_orient_gripper,
                                                   kuka.ll, kuka.ul, kuka.jr, kuka.rp)

        # set motor control for them
        for i in range(kuka.kukaEndEffectorIndex + 1):
            p.setJointMotorControl2(bodyUniqueId=kuka.kukaUid,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_poses[i],
                                    targetVelocity=0,
                                    force=kuka.maxForce,
                                    maxVelocity=kuka.maxVelocity,
                                    positionGain=0.3,
                                    velocityGain=1)

        # gripper fingers
        p.setJointMotorControl2(kuka.kukaUid, 8, p.POSITION_CONTROL,
                                targetPosition=-desired_finger_angle, force=kuka.fingerAForce)
        p.setJointMotorControl2(kuka.kukaUid, 11, p.POSITION_CONTROL,
                                targetPosition=desired_finger_angle, force=kuka.fingerBForce)
        p.setJointMotorControl2(kuka.kukaUid, 10, p.POSITION_CONTROL,
                                targetPosition=0, force=kuka.fingerTipForce)
        p.setJointMotorControl2(kuka.kukaUid, 13, p.POSITION_CONTROL,
                                targetPosition=0, force=kuka.fingerTipForce)
        
        # step sim!
        p.stepSimulation()

        # inc sim steps and take pics if required
        SIM_STEPS += 1
        if SIM_STEPS % opts.render_freq == 0:
            for c in cameras:
                c.render(opts.run+opts.run_offset, FRAME_NUM)
            FRAME_NUM += 1
            
# x range (0.5, 0.7)
# y range (-0.15, 0.25)
# z range (0.15, 0.4)    # grasp at 0.1 to 0.15 (0.125)?
                                               
# drop random objects into tray

obj_uids = []
for _ in range(opts.num_objects):
    random_obj_id = random.randint(0, 9)
    urdf_filename = "%s/%04d/%04d.urdf" % (opts.obj_urdf_dir, random_obj_id, random_obj_id)

    # drop block fixed x distance from base of arm (0.51)
    # across width of tray (-0.1, 0.3) and from fixed height (0.2)
    block_pos = [u.random_in(0.51, 0.7), u.random_in(-0.1, 0.3), 0.2]
    
    # drop with random yaw rotation
    block_angle = random.random() * math.pi
    block_orient = p.getQuaternionFromEuler([0, 0, block_angle])

    # load into scene
    obj_uids.append(p.loadURDF(urdf_filename, *block_pos, *block_orient))

    # let object fall and settle to be clear of next
    for _ in range(100):
        p.stepSimulation()

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
    success = move_arm_to_pose(desired_pos_gripper=pos_gripper,
                               desired_orient_gripper=orient_gripper,
                               desired_finger_angle=0.3)

    # move down into tray
    pos_gripper[2] = 0.1
    success = move_arm_to_pose(desired_pos_gripper=pos_gripper,
                               desired_orient_gripper=orient_gripper,
                               desired_finger_angle=0.3)

    # grasp!
    success = move_arm_to_pose(desired_pos_gripper=pos_gripper,
                               desired_orient_gripper=orient_gripper,
                               desired_finger_angle=0, max_steps=100)

    # lift!
    pos_gripper[2] = 0.3
    success = move_arm_to_pose(desired_pos_gripper=pos_gripper,
                               desired_orient_gripper=orient_gripper,
                               desired_finger_angle=0)
    
p.disconnect()
