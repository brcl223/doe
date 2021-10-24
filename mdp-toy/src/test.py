import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


out = np.array([[0.1, 0.5, 0.8],
                [0.4, 0.6, 0.85],
                [0.7, 0.75, 1.0]])
fig, ax = plt.subplots()
im = ax.imshow(out)
ax.figure.colorbar(im, ax=ax)

manufacturers = [
    "Aero",
    "Defense",
    "Auto",
]

tiers = [
    "Industry",
    "Component",
    "Feature",
]

ax.set_title("Automotive Component Manufacturing\nPerfomance Matrix")
ax.set_ylabel("Manufacturing Tier")
ax.set_xlabel("Manufacturer")
ax.set_yticks(np.arange(3))
ax.set_xticks(np.arange(3))
ax.set_xticklabels(manufacturers)
ax.set_yticklabels(tiers)

plt.show()
