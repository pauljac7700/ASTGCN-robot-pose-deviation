"""Modified Denavit-Hartenberg model of the UR5, and pose conversion helpers.

Defines the MDH kinematic chain used for the geometric calibration stage, plus
conversions between rotation vectors, rotation matrices, roll-pitch-yaw angles and
homogeneous transformation matrices. Needs roboticstoolbox and spatialmath.
"""

from roboticstoolbox import DHRobot, RevoluteMDH
import numpy as np
import math
from spatialmath import SE3, SO3


def UR5_robot_model():
    # define the robot
    MDH = [[0, 0, 0, 0.08916],
           [math.pi / 2, 0, -math.pi / 2, 0],
           [0, -0.425, 0, 0],
           [0, -0.39225, -math.pi / 2, 0.10915],
           [math.pi / 2, 0, math.pi, 0.09456],
           [-math.pi / 2, 0, 0, 0.0823]]

    alpha = [MDH[0][0], MDH[1][0], MDH[2][0], MDH[3][0], MDH[4][0], MDH[5][0]]
    a = [MDH[0][1], MDH[1][1], MDH[2][1], MDH[3][1], MDH[4][1], MDH[5][1]]
    theta_offset = [MDH[0][2], MDH[1][2], MDH[2][2], MDH[3][2], MDH[4][2], MDH[5][2]]
    d = [MDH[0][3], MDH[1][3], MDH[2][3], MDH[3][3], MDH[4][3], MDH[5][3]]

    robot = DHRobot([
        RevoluteMDH(d=d[0], a=a[0], alpha=alpha[0]),  # joint 1
        RevoluteMDH(d=d[1], a=a[1], alpha=alpha[1]),  # joint 2
        RevoluteMDH(d=d[2], a=a[2], alpha=alpha[2]),  # joint 3
        RevoluteMDH(d=d[3], a=a[3], alpha=alpha[3]),  # joint 4
        RevoluteMDH(d=d[4], a=a[4], alpha=alpha[4]),  # joint 5
        RevoluteMDH(d=d[5], a=a[5], alpha=alpha[5]),  # joint 6
    ], name='UR5 Robot')

    return robot


def rotation_vector_to_matrix(r):
    theta = np.linalg.norm(r)
    if np.isclose(theta, 0):
        return np.eye(3)
    n = r / theta
    n_x, n_y, n_z = n
    K = np.array([
        [0, -n_z, n_y],
        [n_z, 0, -n_x],
        [-n_y, n_x, 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


def rotation_vector_to_rpy(r):
    # 1. 旋转矢量转旋转矩阵
    R = rotation_vector_to_matrix(r)

    # 2. 提取 RPY 角 (ZYX 顺序)
    yaw = np.arctan2(R[1, 0], R[0, 0])        # (Yaw)
    pitch = np.arcsin(-R[2, 0])               # (Pitch)
    roll = np.arctan2(R[2, 1], R[2, 2])       # (Roll)

    return roll, pitch, yaw


def build_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


if __name__ == "__main__":
    # Example
    robot = UR5_robot_model()

    # check the robot
    print(robot)

    # forward kinematics
    q = [np.pi/4, np.pi/6, -np.pi/3, np.pi/3, np.pi/4, np.pi/6]
    # Calculate the pose of the end-effector
    T_fk = robot.fkine(q)
    print("The results of forward kinematics: ")
    print(T_fk)

    # Given target pose (compared with forward kinematics results)
    target_pose = T_fk.t
    # target_pose = np.array([-0.723035973263365, -0.384445046414668, 0.3])
    print("target_pose: ", target_pose)

    # Converts a rotation vector to a rotation matrix
    r = np.array([4.041, 1.516, -1.466])  # Rotation vector
    R = rotation_vector_to_matrix(r)
    print("Rotation matrix is:", R)

    # Construct a transformation matrix
    T = build_transformation_matrix(R, target_pose)
    Tep = SE3(T)

    # Calculating inverse kinematics (return joint Angle)
    q0 = np.array([0.247089858721723, -0.432687973390118, 0.583264451244517, 3.06642392925511, -2.66944092414946, 0.0674611825254398])
    q_ik = robot.ikine_LM(Tep, q0=q0)  # inverse kinematics - Levenberg-Marquardt method
    print("The results of inverse kinematics: ")
    print(q_ik.q)

    # 验证逆运动学
    T_check = robot.fkine(q_ik.q)
    print("The verification result is: ")
    print(T_check)




