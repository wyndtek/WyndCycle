import numpy as np
import matplotlib.pyplot as plt
import cyclic.pisa as ps

clay = ps.soil_clay()

pile1 = ps.PISA_(clay, 6, 20)
res = pile1.stiffness("Lateral", 1, 200, 10)
# plt.plot(res[:,0],res[:,1])
# plt.show()

springs = ps.parallel_stiffness(res)
e = [.001* x* np.sin(x) for x in np.linspace(0, 10 * 3.14, 1000)]
a = []
s = []
for e0 in e:
    sigma, a = ps.kinematic_solver(e0, a, springs)
    s.append(sigma)

plt.plot(e,s)