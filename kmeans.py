import matplotlib.pyplot as plt
import numpy as np

data = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])

labels = {(2, 10): 'A', 
          (2, 5): 'B', 
          (8, 4): 'C', 
          (5, 8): 'D', 
          (7, 5): 'E', 
          (6, 4): 'F', 
          (1, 2): 'G', 
          (4, 9): 'H'
        }



rgba = np.array([
    "#F6EAB9",
    "#FF3442",
    "#FF8AAF"
])


# k-means
k = 3
centroids = np.array([[2, 10], [5, 8], [1, 2]])
    
def recalculate_clusters(data, centroids, k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    for x in data:
        dist = []
        for j in range(k):
            dist.append(np.linalg.norm(x-centroids[j]))
        print(x)
        print(dist)
        print("----------")
        clusters[dist.index(min(dist))].append(x)
    return clusters

def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids


# iteraciones
for i in range(3):
    clusters = recalculate_clusters(data, centroids, k) 
    print(clusters)
    
    centroids = recalculate_centroids(centroids, clusters, k)
    
    print(centroids)
    
    # graficar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 9])
    ax.set_ylim([0, 11])
    plt.title(f"K-Means iteración {i+1}")
    
    for j in range(3):
        
        arr = np.array(clusters[j])
        x, y = np.split(arr, 2, axis=1)
        for l in range(len(x)): ax.text(x[l]+0.1, y[l]+0.2, labels[int(x[l]), int(y[l])])
        
        plt.scatter(x, y, c=rgba[j], edgecolors='black', linewidth=0.5, s=80)
        plt.scatter(centroids[j][0], centroids[j][1], c=rgba[j], marker=(5, 1), linewidth=0.5, edgecolors='black', s=140)

    plt.show()
    
