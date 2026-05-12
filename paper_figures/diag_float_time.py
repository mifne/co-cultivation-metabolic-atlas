import numpy as np
t = 0.0
dt = 0.2
hit = False
for i in range(500):
    t += dt
    if t == 80.0:
        hit = True
print(f"Hit 80.0 EXACTLY: {hit}")
