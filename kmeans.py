import matplotlib.pyplot as plt
import numpy as np

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

rgb = np.array([
    "#F6EAB9",
    "#FF3442",
    "#FF8AAF"
])

# k-means
k = 3
centroids = np.array([[2.0, 10.0], [5.0, 8.0], [1.0, 2.0]])
    
def recalculate_clusters(data, centroids, k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    for x in data:
        dist = []
        for j in range(k):
            dist.append(np.linalg.norm(x-centroids[j]))
        #print(x)
        #print(dist)
        #print("----------")
        clusters[dist.index(min(dist))].append(x)
    return clusters

def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
        #print(np.average(clusters[i], axis=0))
    return centroids


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 9])
ax.set_ylim([0, 11])

# dibujar estado inicial
plt.title(f"K-Means estado inicial")

x, y = np.split(data, 2, axis=1)
for l in range(len(x)): ax.text(x[l]+0.1, y[l]+0.2, label[int(x[l]), int(y[l])])

plt.scatter(x, y, c="grey", edgecolors='black', linewidth=0.5, s=100)

for j in range(3):
    plt.scatter(centroids[j][0], centroids[j][1], c=rgb[j], 
                marker=(5, 1), linewidth=0.5, edgecolors='black', s=160)
plt.show()

# iteraciones
for i in range(3):
    clusters = recalculate_clusters(data, centroids, k) 
    for j in range(3):
        print(f"clusters {j}:")
        print(clusters[j])
    
    # recalcular centroides
    centroids = recalculate_centroids(centroids, clusters, k)
    print(centroids)
    
    print("------------")    
    # dibujar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 9])
    ax.set_ylim([0, 11])
    plt.title(f"K-Means iteración {i+1}")
    
    for j in range(3):
        
        arr = np.array(clusters[j])
        x, y = np.split(arr, 2, axis=1)
        for l in range(len(x)): ax.text(x[l]+0.1, y[l]+0.2, label[int(x[l]), int(y[l])])
        
        plt.scatter(x, y, c=rgb[j], edgecolors='black', linewidth=0.5, s=100)
        plt.scatter(centroids[j][0], centroids[j][1], c=rgb[j], 
                    marker=(5, 1), linewidth=0.5, edgecolors='black', s=160)

    plt.show()
    
