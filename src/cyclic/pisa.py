import numpy as np
from scipy import interpolate


def soil_clay():
    data_clay = [50, 0, 10, 0,
                 100, 1, 28, 1,
                 160, 2, 45, 2,
                 160, 2.5, 70, 3.5,
                 110, 4, 95, 5,
                 110, 4.8, 105, 6,
                 125, 8, 122, 8,
                 140, 10, 160, 10,
                 155, 11.5, 228, 12.2,
                 215, 30, 390, 30, 442.027, 100, 1027.078, 100]
    clay = np.reshape(data_clay, (11, 4))
    X1 = clay[:, 1]
    Y1 = clay[:, 0]
    X2 = clay[:, 3]
    Y2 = clay[:, 2] * 1000
    su = interpolate.interp1d(X1, Y1)
    G0 = interpolate.interp1d(X2, Y2)
    return (su, G0)


###############################################################################
class PISA_:
    def __init__(self, soil, D, L):
        self.diameter = D
        self.Length = L
        self.su = soil[0]
        self.G0 = soil[1]

    def later_curve(self, e, Depth):
        "Distributed Later Load"
        Dist_Lateral_X = 200
        Dist_Lateral_Y = -1.11 * Depth / self.diameter + 8.17
        Dist_Lateral_W = -0.07 * Depth / self.diameter + 0.92
        Dist_Lateral_Z = 11.66 - 8.64 * np.exp(-0.37 * Depth / self.diameter)

        X = Dist_Lateral_X
        Y = Dist_Lateral_Y
        W = Dist_Lateral_W
        Z = Dist_Lateral_Z
        return self.cubic(X, Y, W, Z, e, Depth)

    "Distributed moment"

    def moment_curve(self, e, Depth):
        "Distributed moment"
        Dist_Moment_X = 10
        Dist_Moment_Y = -0.12 * Depth / self.diameter + 0.98
        Dist_Moment_W = 0
        Dist_Moment_Z = -0.05 * Depth / self.diameter + 0.38

        X = Dist_Moment_X
        Y = Dist_Moment_Y
        W = Dist_Moment_W
        Z = Dist_Moment_Z
        return self.cubic(X, Y, W, Z, e, Depth)

    "Based_Shear"

    def based_shear(self, e, Depth=0):
        Depth = self.Length
        "Based Shear Hb"
        Base_Shear_X = 300
        Base_Shear_Y = -0.32 * self.Length / self.diameter + 2.58
        Base_Shear_W = -0.04 * self.Length / self.diameter + 0.76
        Base_Shear_Z = 0.07 * self.Length / self.diameter + 0.59

        X = Base_Shear_X
        Y = Base_Shear_Y
        W = Base_Shear_W
        Z = Base_Shear_Z

        print(X, Y, W, Z)
        return self.cubic(X, Y, W, Z, e, Depth)

    "Base Momet"

    def based_moment(self, e, Depth=0):
        "Base_Moment"
        Depth = self.Length
        Base_Moment_X = 200
        Base_Moment_Y = -0.002 * self.Length / self.diameter + 0.19
        Base_Moment_W = -0.15 * self.Length / self.diameter + 0.99
        Base_Moment_Z = -0.07 * self.Length / self.diameter + 0.65

        X = Base_Moment_X
        Y = Base_Moment_Y
        W = Base_Moment_W
        Z = Base_Moment_Z
        return self.cubic(X, Y, W, Z, e, Depth)

    def cubic(self, X, Y, W, Z, e, Depth):
        IR = self.G0(Depth) / self.su(Depth)
        y_ = e * IR

        a = 1 - 2 * W
        b = 2 * W * y_ / X - (1 - W) * (1 + y_ * Y / Z)
        c = y_ / Z * Y * (1 - W) - W * (y_ ** 2 / X ** 2)

        if 0 < abs(y_) <= X:
            sigma_ = Z * 2 * c / (-b + np.sqrt(b ** 2 - 4 * a * c))
        else:
            sigma_ = Z

        out = sigma_ * self.su(Depth) * self.diameter ** 3
        return out

    def stiffness(self, case, end, n, depth):
        switcher = {
            'Lateral': self.later_curve,
            'Rotational': self.moment_curve,
            'Base_Shear': self.based_shear,
            'Base_Moment': self.based_moment
        }
        func = switcher.get(case)
        e = np.geomspace(0.0000001, end, n)
        # e = np.delete(e, 0)
        # out = [[0, 0]]
        out = []
        for i in e:
            res = func(i, depth)
            out.append([i, res])
        out = np.reshape(out, [n, 2])
        return out


###############################################################################

def parallel_stiffness(stiffness):
    e = stiffness[:, 0]
    sigma = stiffness[:, 1]
    n = int(stiffness.size / 2)
    mat_a = np.zeros([n - 1, n - 1])
    mat_b = np.zeros(n - 1)

    for i in range(n - 1):
        E = (sigma[i + 1] - sigma[i]) / (e[i + 1] - e[i])
        mat_a[i, i:] = 1
        mat_b[i] = E
    H = np.linalg.solve(mat_a, mat_b)

    mat_c = np.zeros([n - 1, n - 1])
    mat_d = np.zeros(n - 1)
    for i in range(n - 1):
        mat_c[i, 0:i + 1] = 1
        mat_d[i] = sigma[i + 1] - e[i + 1] * np.sum(H[i + 1:])
    k = np.linalg.solve(mat_c, mat_d)

    return [H, k]


#########################################################

def kinematic_solver(e,a, spring):

    n = np.size(spring[0])
    H = spring[0]
    k = spring[1]
    if np.size(a) == 0:
        a = np.zeros(n)
    for i in range(n):
        x = H[i]*(e-a[i])
        if abs(x) > k[i]:
            a[i]=e-np.sign(x)*k[i]/H[i]
        else:
            pass

    sigma = sum(H*(e-a))
    return sigma,a
