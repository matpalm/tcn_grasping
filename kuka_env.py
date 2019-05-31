import pybullet as p
import numpy as np
from pybullet_envs.bullet import kuka
import pybullet_data
import math
import util as u
import random

class KukaEnv(object):

    def __init__(self, gui, render_freq, run_id, num_objects, obj_urdf_dir):
        self.sim_steps = 0
        self.frame_num = 0
        self.render_freq = render_freq
        self.run_id = run_id             # for rendering filename
        self.cameras = []

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # load in table and arm
        p.loadURDF(pybullet_data.getDataPath()+"/table/table.urdf", 0.5,0.,-0.82, 0,0,0,1)
        self.kuka = kuka.Kuka(urdfRootPath=pybullet_data.getDataPath())

        # drop random objects into tray
        p.setGravity(0, 0, -9.8)
        for _ in range(num_objects):
            random_obj_id = random.randint(0, 9)
            urdf_filename = "%s/%04d/%04d.urdf" % (obj_urdf_dir, random_obj_id, random_obj_id)

            # drop block fixed x distance from base of arm (0.51)
            # across width of tray (-0.1, 0.3) and from fixed height (0.2)
            block_pos = [u.random_in(0.51, 0.7), u.random_in(-0.1, 0.3), 0.2]

            # drop with random yaw rotation
            block_angle = random.random() * math.pi
            block_orient = p.getQuaternionFromEuler([0, 0, block_angle])

            # load into scene
            _obj_uid = p.loadURDF(urdf_filename, *block_pos, *block_orient)

            # let object fall and settle to be clear of next
            for _ in range(100):
                p.stepSimulation()


    # hand rolled IK move of arm without constraints, dx/dy/dz limits in kuka class
    def move_arm_to_pose(self, desired_pos_gripper, desired_orient_gripper,
                         desired_finger_angle, max_steps=200):

        # x range (0.5, 0.7)
        # y range (-0.15, 0.25)
        # z range (0.15, 0.4)    # grasp at 0.1 to 0.15 (0.125)?

        steps = 0
        while True:

            steps += 1
            if steps > max_steps:
                return False

            gripper_state = p.getLinkState(self.kuka.kukaUid, self.kuka.kukaEndEffectorIndex)
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
            left_side_finger_angle = -p.getJointState(self.kuka.kukaUid, 8)[0]
            right_side_finger_angle = p.getJointState(self.kuka.kukaUid, 11)[0]
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
            joint_poses = p.calculateInverseKinematics(self.kuka.kukaUid, self.kuka.kukaEndEffectorIndex,
                                                       desired_pos_gripper,
                                                       desired_orient_gripper,
                                                       self.kuka.ll, self.kuka.ul, self.kuka.jr, self.kuka.rp)

            # set motor control for them
            for i in range(self.kuka.kukaEndEffectorIndex + 1):
                p.setJointMotorControl2(bodyUniqueId=self.kuka.kukaUid,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_poses[i],
                                        targetVelocity=0,
                                        force=self.kuka.maxForce,
                                        maxVelocity=self.kuka.maxVelocity,
                                        positionGain=0.3,
                                        velocityGain=1)

                # gripper fingers
                p.setJointMotorControl2(self.kuka.kukaUid, 8, p.POSITION_CONTROL,
                                        targetPosition=-desired_finger_angle, force=self.kuka.fingerAForce)
                p.setJointMotorControl2(self.kuka.kukaUid, 11, p.POSITION_CONTROL,
                                        targetPosition=desired_finger_angle,force=self.kuka.fingerBForce)
                p.setJointMotorControl2(self.kuka.kukaUid, 10, p.POSITION_CONTROL,
                                        targetPosition=0, force=self.kuka.fingerTipForce)
                p.setJointMotorControl2(self.kuka.kukaUid, 13, p.POSITION_CONTROL,
                                        targetPosition=0, force=self.kuka.fingerTipForce)

                # step sim!
                p.stepSimulation()

                # inc sim steps and take pics if required
                self.sim_steps += 1
                if self.sim_steps % self.render_freq == 0:
                    for c in self.cameras:
                        c.render(self.run_id, self.frame_num)
                    self.frame_num += 1
