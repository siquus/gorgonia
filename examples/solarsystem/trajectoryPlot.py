#!/usr/bin/python3.6

import json
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

json_data = open("trajectoryData.json", "r")
data = json.load(json_data)

objectStrings = data["Objects"]
objectNrOf = len(objectStrings)

dimensionsNrOf = data["Dimensions"]

trajectory = data["Trajectory"]

if len(trajectory) % (dimensionsNrOf * objectNrOf):
    print("Trajectory Points are not divisible by number of objects times number of dimensions!")
    quit()


dataPointsNrOf = int(len(trajectory) / (dimensionsNrOf * objectNrOf))


objectTrajectories = np.ndarray(shape=(objectNrOf, dimensionsNrOf, dataPointsNrOf), dtype=np.float64)

currentObject = 0
currentDimension = 0
currentDatapoint = 0

for trajPt in trajectory:
    objectTrajectories[currentObject][currentDimension][currentDatapoint] = trajPt
    currentDimension += 1

    if currentDimension >= dimensionsNrOf:
        currentObject += 1
        currentDimension = 0

    if currentObject >= objectNrOf:
        currentDatapoint += 1
        currentObject = 0

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

plotall = True

if plotall:
    for index, name in enumerate(objectStrings):
        ax.plot(objectTrajectories[index][0], objectTrajectories[index][1], objectTrajectories[index][2], label=name)
else:
    index = 2
    name = objectStrings[2]
    ax.plot(objectTrajectories[index][0], objectTrajectories[index][1], objectTrajectories[index][2], label=name)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


