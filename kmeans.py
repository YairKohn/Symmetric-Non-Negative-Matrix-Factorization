import math

def euclidean_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.
    :param p1: The first point.
    :type p1: list
    :param p2: The second point.
    :type p2: list
    :return: The Euclidean distance between the two points.
    :rtype: float
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def mean(points):
    """
    Calculates the mean of a list of points.
    The centroid calculation step:
    compute the mean of all points assigned to a cluster to update the cluster centroid.
    :param points: The list of points.
    :type points: list of lists
    :return: The mean of the points.
    :rtype: list
    """
    d = len(points[0])
    n = len(points)
    return [sum(point[i] for point in points) / n for i in range(d)]

def kmeans(data, K, max_iter=300 ):
    """ 
    Runs the k-means clustering algorithm.
    :param data: The data points.
    :type data: list of lists
    :param K: The number of clusters.
    :type K: int
    :param max_iter: The maximum number of iterations.
    :type max_iter: int
    :return: The centroids.
    :rtype: list of lists
    """
    epsilon = 1e-4
    centroids = [data[i][:] for i in range(K)] # Initialize centroids with the first K points, doing shallow copies.
    for _ in range(max_iter):
        clusters = [[] for _ in range(K)]

        for point in data: # Assign points to the nearest centroid
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            min_index = distances.index(min(distances))
            clusters[min_index].append(point)

        converged = True
        new_centroids = []
        for i in range(K): # Update centroids
            if clusters[i]: # Avoid empty cluster
                new_centroid = mean(clusters[i])
            else:
                new_centroid = centroids[i]  # Keep the old centroid if cluster is empty

            dist = euclidean_distance(centroids[i], new_centroid)
            if dist >= epsilon:
                converged = False

            new_centroids.append(new_centroid)

        centroids = new_centroids

        if converged:
            break
    return centroids