from typing import Tuple

import numpy as np


class Quaternion:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=0, x=0, y=0, z=0):
        """Quaternion is a number systerm of representing a 3D rotation that has
        computational advantages including speed and numerical robustness.
        A quaternion can be considered as a rotation about a vector in space where
        q = cos (theta/2) sin(theta/2) <vx vy vz>
        where <vx vy vz> is a unit vector.
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return "{} < {}, {}, {} >".format(self.w, self.x, self.y, self.z)

    def conj(self):
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    def norm_square(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def is_unit(self):
        return abs(self.norm_square() - 1) < 1e-12

    def to_unit(self):
        """Return an equivalent unit quaternion
        """
        squared_norm = self.norm_square()
        if squared_norm < 1e-12:
            return Quaternion(1, 0, 0, 0)
        norm = np.sqrt(squared_norm)
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def to_vector(self) -> np.array:
        return np.array([self.w, self.x, self.y, self.z])

    def get_vector(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    def __iadd__(self, other):
        if isinstance(other, Quaternion):
            self.w += other.w
            self.x += other.x
            self.y += other.y
            self.z += other.z
        elif isinstance(other, (int, float)):
            self.w += other
        else:
            raise "Right hand side should be int, float, or Quaternion"
        return self

    def __add__(self, other):
        qr = Quaternion(self.w, self.x, self.y, self.z)
        qr += other
        return qr

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w + other, self.x, self.y, self.z)
        raise "Left hand side should be int, float"

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(other - self.w, -self.x, -self.y, -self.z)
        raise "Left hand side should be int, float"

    def __imul__(self, other):
        if isinstance(other, Quaternion):
            # self.matrix() @ other.vector()
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z

            self.w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            self.x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            self.y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            self.z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        elif isinstance(other, (int, float)):
            self.w *= other
            self.x *= other
            self.y *= other
            self.z *= other
        else:
            raise "Right hand side should be int, float, or Quaternion"
        return self

    def __mul__(self, other):
        qr = Quaternion(self.w, self.x, self.y, self.z)
        qr *= other
        return qr

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other, self.x * other, self.y * other, self.z * other
            )
        raise "Left hand side should be int, float"

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __eq__(self, other):
        """Returns true if the following is true for each element:
        `absolute(a - b) <= (atol + rtol * absolute(b))`
        """
        if isinstance(other, Quaternion):
            return (
                abs(self.w - other.w) < 1e-12
                and abs(self.x - other.x) < 1e-12
                and abs(self.y - other.y) < 1e-12
                and abs(self.z - other.z) < 1e-12
            )
        return False

    def inv(self):
        if self.is_unit():
            return self.conj()
        squared_norm = self.norm_square()
        if squared_norm < 1e-12:
            return Quaternion(1, 0, 0, 0)
        return self.conj() / squared_norm

    def __itruediv__(self, other):
        if isinstance(other, Quaternion):
            self *= other.inv()
        elif isinstance(other, (int, float)):
            self.w /= other
            self.x /= other
            self.y /= other
            self.z /= other
        else:
            raise "Right hand side should be int, float, or Quaternion"
        return self

    def __truediv__(self, other):
        qr = Quaternion(self.w, self.x, self.y, self.z)
        qr /= other
        return qr

    def __rtruediv__(self, other):
        return Quaternion(
            other / self.w, other / self.x, other / self.y, other / self.z
        )

    def to_left_matrix(self) -> np.array:
        """Return 1 of 48 distinct 4x4 matrix representations
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])

    def to_right_matrix(self) -> np.array:
        """Return 1 of 48 distinct 4x4 matrix representations
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])

    def to_rotation(self) -> np.array:
        """Return an equivalent rotation matrix.
        """
        n = self.norm_square()
        s = 0
        if n:
            s = 2 / n
        wx = s * self.w * self.x
        wy = s * self.w * self.y
        wz = s * self.w * self.z
        xx = s * self.x * self.x
        xy = s * self.x * self.y
        xz = s * self.x * self.z
        yy = s * self.y * self.y
        yz = s * self.y * self.z
        zz = s * self.z * self.z

        return np.array(
            [
                [1 - yy - zz, xy - wz, xz + wy],
                [xy + wz, 1 - xx - zz, yz - wx],
                [xz - wy, wx + yz, 1 - xx - yy],
            ]
        )

    def to_rpy(self) -> Tuple[float, float, float]:
        q_norm = self.norm_square()
        roll = np.arctan2(
            2 * (self.w * self.x + self.y * self.z),
            q_norm - 2 * (self.x ** 2 + self.y ** 2),
        )

        yaw = np.arctan2(
            2 * (self.w * self.z + self.x * self.y),
            q_norm - 2 * (self.y ** 2 + self.z ** 2),
        )

        pitch = 0
        sinp = 2 * (self.w * self.y - self.x * self.z)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        return roll, pitch, yaw

    def rotate_vector(self, v: np.array) -> np.array:
        """
        Rotate the vector v by a unit quaternion q defining an
        Euler rotation (ZYX, RPY)
        """
        if v.shape != (3,):
            raise "v should be a vector of 3"
        q = self.to_unit()
        q_vec = q.get_vector()
        t = 2 * np.cross(q_vec, v)
        result = v + q.w * t + np.cross(q_vec, t)
        return result


def vector_to_quaternion(arr: np.array) -> Quaternion:
    return Quaternion(arr[0], arr[1], arr[2], arr[3])