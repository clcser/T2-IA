import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

data = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])

label = {(2, 10): 'A', 
          (2, 5): 'B', 
          (8, 4): 'C', 
          (5, 8): 'D', 
          (7, 5): 'E', 
          (6, 4): 'F', 
          (1, 2): 'G', 
          (4, 9): 'H'
        }

color = np.array([
    "#F6EAB9",
    "#FF3442",
    "#FF8AAF"
])

# eps = radio de vecindad considerada
# min_points = minimo numero de vecinos para considerarlo core

# DBSCAN
eps = 2.0 #sqrt(10)
min_points = 2
n = len(data)
assigned = np.full(n, False)

is_core = np.full(n, False)

for i in range(n):
    count = np.sum(np.linalg.norm(data - data[i], axis=1) <= eps)
    if count >= min_points:
        is_core[i] = True
print(is_core)

clusters = []

for i in range(n):
    if not is_core[i] or assigned[i]: 
        continue
    cluster = []
    stack = [i]
    
    while stack:
        idx = stack.pop()
        if assigned[idx]: 
            continue
        
        assigned[idx] = True
        cluster.append(idx)
        
        dists = np.linalg.norm(data - data[idx], axis=1)
        neighbors = np.where((dists <= eps))[0]
        
        for neighbor in neighbors:
            if assigned[neighbor]: continue
            if is_core[neighbor]:
                stack.append(neighbor)
            else:
                assigned[neighbor] = True
                cluster.append(neighbor)
    
    clusters.append(cluster)

outliers = []
for i in range(n):
    if not assigned[i]:
        outliers.append(i)
        assigned[i] = True

for i, cluster in enumerate(clusters):
    nombres = [label[tuple(data[idx])] for idx in cluster]
    print(f"Cluster {i+1}: {nombres}")

# dibujar clusters
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 9])
ax.set_ylim([0, 11])
plt.title(f"DBSCAN: eps={eps}, min_points={min_points}")

for j, cluster in enumerate(clusters):
    arr = np.array([data[i] for i in cluster])
    x, y = np.split(arr, 2, axis=1)
    
    for l in range(len(x)):
        pt = (int(x[l]), int(y[l]))
        if pt in label:
            ax.text(x[l] + 0.1, y[l] + 0.2, label[pt])

    plt.scatter(x, y, c=color[j], edgecolors='black', linewidth=0.5, s=150)

# dibujar outliers
if outliers:
    arr = np.array([data[i] for i in outliers])
    x, y = np.split(arr, 2, axis=1)

    for l in range(len(x)):
        pt = (int(x[l]), int(y[l]))
        if pt in label:
            ax.text(x[l] + 0.1, y[l] + 0.2, label[pt])

    plt.scatter(x, y, c="gray", edgecolors='black', linewidth=0.5, s=150, label='Outliers')

plt.show()
