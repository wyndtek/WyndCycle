import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cyclic.pisa as ps

clay = ps.soil_clay()

pile1 = ps.PISA_(clay, 0.764, 4)
res3 = pile1.stiffness("Lateral", .2, 200, 0)
# plt.plot(res[:,0],res[:,1])

redwin = pd.read_excel('redwin1.xlsx')
redwin = redwin[~redwin.duplicated()]
res1 = redwin.values
# res = res1[:,[1,0]]
res= res1

springs = ps.parallel_stiffness(res)
e = [.002*  np.sin(x) for x in np.linspace(0,20*3.14, 1000)]
AB = np.arange(0,1.2e-3,.01e-3)
BC = np.arange(1.2e-3,.6e-3,-.011e-3)
CD = np.arange(.6e-3,1.2e-3,.01e-3)
DE = np.arange(1.2e-3,2e-3,.01e-3)
EF = np.arange(2e-3,-2e-3,-.01e-3)
FG = np.arange(-2e-3,2e-3,.01e-3)
e = np.concatenate([AB,BC,CD,DE,EF,FG])


a = []
s = []
for e0 in e:
    sigma, a = ps.kinematic_solver(e0, a, springs)
    s.append(sigma)

plt.plot(e,s)
plt.grid('on')
plt.xlim([-3e-3,3e-3])
plt.ylim([-4,4])