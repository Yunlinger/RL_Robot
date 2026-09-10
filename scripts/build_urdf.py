"""Regenerate biped.urdf from robot_config.py; no external mesh files needed."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_config import JOINT_LIMITS, ROBOT, ROOT


def vec(values):
    return " ".join(f"{v:.10g}" for v in values)


def box_link(robot, name, mass, size, centre=(0, 0, 0), colour=(0.2, 0.5, 0.8, 1)):
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=vec(centre))
    ET.SubElement(inertial, "mass", value=str(mass))
    x, y, z = size
    ET.SubElement(inertial, "inertia", ixx=str(mass * (y*y + z*z) / 12),
                  iyy=str(mass * (x*x + z*z) / 12), izz=str(mass * (x*x + y*y) / 12),
                  ixy="0", ixz="0", iyz="0")
    for kind in ("visual", "collision"):
        element = ET.SubElement(link, kind)
        ET.SubElement(element, "origin", xyz=vec(centre))
        ET.SubElement(ET.SubElement(element, "geometry"), "box", size=vec(size))
        if kind == "visual":
            ET.SubElement(ET.SubElement(element, "material", name=name + "_colour"),
                          "color", rgba=vec(colour))


def joint(robot, name, parent, child, xyz, axis, limits):
    j = ET.SubElement(robot, "joint", name=name, type="revolute")
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    ET.SubElement(j, "origin", xyz=vec(xyz))
    ET.SubElement(j, "axis", xyz=vec(axis))
    ET.SubElement(j, "limit", lower=str(limits[0]), upper=str(limits[1]),
                  effort=str(ROBOT.torque_limit), velocity=str(ROBOT.servo_speed))
    ET.SubElement(j, "dynamics", damping="0.0001", friction="0.0002")


def build():
    r = ET.Element("robot", name="sg90_biped_10dof")
    box_link(r, "torso", ROBOT.torso_mass, (0.045, 0.070, 0.055))
    for side, sign in (("left", 1), ("right", -1)):
        hip, thigh, shin, ankle, foot = [f"{side}_{n}" for n in ("hip", "thigh", "shin", "ankle", "foot")]
        box_link(r, hip, ROBOT.hip_mass, (0.023, 0.014, 0.018), (0, 0, -0.009))
        box_link(r, thigh, ROBOT.thigh_mass, (0.016, 0.022, ROBOT.thigh_length),
                 (0, 0, -ROBOT.thigh_length / 2), (0.85, 0.3, 0.2, 1))
        box_link(r, shin, ROBOT.shin_mass, (0.014, 0.018, ROBOT.shin_length),
                 (0, 0, -ROBOT.shin_length / 2), (0.2, 0.7, 0.5, 1))
        box_link(r, ankle, ROBOT.ankle_mass, (0.022, 0.018, 0.018))
        box_link(r, foot, ROBOT.foot_mass, (0.060, 0.040, 0.006),
                 (0.005, 0, -ROBOT.foot_drop), (0.9, 0.7, 0.2, 1))
        joint(r, f"{side}_hip_roll", "torso", hip, (0, sign * ROBOT.hip_spacing / 2, -ROBOT.hip_drop), (1, 0, 0), JOINT_LIMITS[0])
        joint(r, f"{side}_hip_pitch", hip, thigh, (0, 0, -ROBOT.hip_pitch_drop), (0, 1, 0), JOINT_LIMITS[1])
        joint(r, f"{side}_knee", thigh, shin, (0, 0, -ROBOT.thigh_length), (0, 1, 0), JOINT_LIMITS[2])
        joint(r, f"{side}_ankle_pitch", shin, ankle, (0, 0, -ROBOT.shin_length), (0, 1, 0), JOINT_LIMITS[3])
        joint(r, f"{side}_ankle_roll", ankle, foot, (0, 0, 0), (1, 0, 0), JOINT_LIMITS[4])
    ET.indent(r, space="  ")
    return ET.tostring(r, encoding="unicode", xml_declaration=True) + "\n"


if __name__ == "__main__":
    (ROOT / "biped.urdf").write_text(build())
    print(f"Wrote {ROOT / 'biped.urdf'}; nominal mass {ROBOT.total_mass:.3f} kg")
