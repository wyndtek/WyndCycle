import numpy as np

import cyclic.pisa as ps

clay = ps.soil_clay()
pile1 = ps.PISA_(clay, 0.764, 4)

stiffness = pile1.stiffness("Lateral", 1, 100, 0)
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
    mat_d[i] = sigma[i + 1] - e[i + 1] * np.sum(H[i+1:])
k = np.linalg.solve(mat_c, mat_d)
