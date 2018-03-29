import math
import random
import matplotlib.pyplot as plt


plt.ion()
for i in range(10):
    plt.clf()
    circle_small = plt.Circle((0, 0), radius=1, color='b', fill=False)
    plt.gca().add_patch(circle_small)
    circle_large = plt.Circle((0, 0), radius=2, color='b', fill=False)
    plt.gca().add_patch(circle_large)
    circle_center = plt.Circle((0, 0), radius=1.5, color='g', fill=False)
    plt.gca().add_patch(circle_center)
    x, y, angle = random.random(), random.random(), random.random() * math.pi
    agent = plt.Circle((x, y), radius=0.1, color='r', fill=False)
    plt.gca().add_patch(agent)
    plt.plot([x, x + math.cos(angle) * 0.2], [y, y + math.sin(angle) * 0.2], linestyle='-', color='r')
    plt.axis('scaled')
    plt.show()
    plt.pause(0.001)
    print("step: %d" % i)
plt.ioff()
plt.show()
