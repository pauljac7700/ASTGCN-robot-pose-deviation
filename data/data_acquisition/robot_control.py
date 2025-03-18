import rtde_control
import time
import pandas as pd


if __name__ == "__main__":
    rtde_c = rtde_control.RTDEControlInterface("169.254.60.1")
    # set the speed and acceleration of the motion
    speed = 0.1
    acceleration = 0.2

    # get the joint angles
    joint_angles_set = pd.read_csv("grid_joint_angles.csv", header=0, index_col=0)

    point_num = 1
    for point_num in range(1000):
        print(f"point {point_num}")
        joint_angles = joint_angles_set.iloc[point_num, :].to_numpy().astype(float)

        # Move to initial joint position with a regular moveJ
        rtde_c.moveJ(joint_angles, speed, acceleration)
        time.sleep(1)

    rtde_c.disconnect()
