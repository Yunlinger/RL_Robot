"""A small foot-trajectory/IK prior; SAC learns bounded residual corrections."""
import numpy as np

from robot_config import JOINT_LIMITS, ROBOT


def gait_reference(phase, speed, *, walking=True):
    """Angles and desired contacts for one cycle, in JOINT_NAMES order.

    +x is forward, +y left. A positive knee angle bends the shin backwards.
    This is a kinematic starting point, not evidence of a stable walking policy.
    """
    angles, contacts = [], []
    sway = ROBOT.lateral_sway * np.cos(2 * np.pi * (phase - ROBOT.stance_fraction / 2)) if walking else 0.0
    stride = (speed * ROBOT.gait_period * ROBOT.stance_fraction * ROBOT.stride_gain
              if walking else 0.0)
    for leg_phase in (phase % 1, (phase + 0.5) % 1):
        stance = leg_phase < ROBOT.stance_fraction or not walking
        if stance:
            fraction = leg_phase / ROBOT.stance_fraction
            x = stride * (0.5 - fraction) if walking else 0.0
            lift = 0.0
        else:
            fraction = (leg_phase - ROBOT.stance_fraction) / (1 - ROBOT.stance_fraction)
            smooth = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
            x = stride * (smooth - 0.5)
            lift = ROBOT.foot_clearance * np.sin(np.pi * fraction)**2
        down = ROBOT.stance_height - lift
        hip_roll = np.arctan2(-sway, down)
        sagittal_down = np.hypot(down, sway)
        length2 = x*x + sagittal_down*sagittal_down
        a, b = ROBOT.thigh_length, ROBOT.shin_length
        knee = np.arccos(np.clip((length2 - a*a - b*b) / (2*a*b), -1, 1))
        hip_pitch = -np.arctan2(x, sagittal_down) - np.arctan2(b * np.sin(knee), a + b * np.cos(knee))
        angles.extend((hip_roll, hip_pitch, knee, -hip_pitch - knee, -hip_roll))
        contacts.append(float(stance))
    limits = np.asarray(JOINT_LIMITS)
    return np.clip(angles, limits[:, 0], limits[:, 1]), np.asarray(contacts)
