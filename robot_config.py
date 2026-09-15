"""Shared dimensions and actuator assumptions for the unbuilt SG90 prototype.

SI units throughout. Torque/speed are design assumptions, not a manufacturer's
continuous-duty rating; measure the assembled robot before sim-to-real use.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RobotConfig:
    thigh_length: float = 0.055
    shin_length: float = 0.055
    hip_spacing: float = 0.060
    hip_drop: float = 0.0275
    hip_pitch_drop: float = 0.009
    foot_drop: float = 0.012
    torso_mass: float = 0.100
    hip_mass: float = 0.012
    thigh_mass: float = 0.020
    shin_mass: float = 0.018
    ankle_mass: float = 0.010
    foot_mass: float = 0.015
    torque_limit: float = 0.080
    servo_speed: float = 6.0
    servo_kp: float = 0.12
    servo_kd: float = 1.0
    control_hz: int = 50
    physics_hz: int = 500
    gait_period: float = 0.8
    stance_fraction: float = 0.62
    foot_clearance: float = 0.014
    lateral_sway: float = 0.016
    stride_gain: float = 1.05
    stance_height: float = 0.107

    @property
    def total_mass(self):
        return self.torso_mass + 2 * (
            self.hip_mass + self.thigh_mass + self.shin_mass + self.ankle_mass + self.foot_mass
        )


ROBOT = RobotConfig()
JOINT_TYPES = ("hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
JOINT_NAMES = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in JOINT_TYPES)
JOINT_LIMITS = ((-0.45, 0.45), (-0.85, 0.65), (0.02, 1.5), (-0.95, 0.65), (-0.45, 0.45)) * 2
# The reference trajectory supplies the large motion.  SAC only needs a small
# correction for balance; limiting residuals keeps exploration from destroying
# the stable two-foot gait.
ACTION_SCALE = (0.03, 0.05, 0.08, 0.05, 0.03) * 2
