import sys
import math


def print_error_and_exit():
    print("An Error Has Occurred")
    sys.exit(1)


def read_input():
    points = []
    for line in sys.stdin:
        line = line.strip() #delete leading and trailing whitespace
        if not line:
            continue
        vector = list(map(float, line.split(',')))
        points.append(vector)
    return points

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def mean(points):
    d = len(points[0])
    n = len(points)
    return [sum(point[i] for point in points) / n for i in range(d)]

def kmeans(data, K, max_iter=400 ):
    epsilon=1e-4
    centroids = [data[i][:] for i in range(K)] # Initialize centroids with the first K points, doing shallow copies.
    for _ in range(max_iter):
        clusters = [[] for _ in range(K)]

        # 3.Assign points to the nearest centroid
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            min_index = distances.index(min(distances))
            clusters[min_index].append(point)

        converged = True
        new_centroids = []

        # 4.Update centroids
        for i in range(K):
            if clusters[i]:  # Avoid empty cluster
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

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=str, help="Number of clusters (1 < K < N)")
    parser.add_argument("ITER", type=str, nargs='?', default="400", help="Max number of iterations (1 < ITER < 1000)")

    args = parser.parse_args()

    try:
        args.K = float(args.K)
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)
    try:
        args.ITER = float(args.ITER)
    except ValueError:
        print("Incorrect maximum iteration!")
        sys.exit(1)    
    

    
    if not int(args.K) == args.K:
        print("Incorrect number of clusters!")
        sys.exit(1)
    if not int(args.ITER) == args.ITER:
        print("Incorrect maximum iteration!")
        sys.exit(1)

    K = int(args.K)
    ITER = int(args.ITER)

    data = read_input()

    if not (1 < K < len(data)):
        print("Incorrect number of clusters!")
        sys.exit(1)
    if not (1 < ITER < 1000):
        print("Incorrect maximum iteration!")
        sys.exit(1)

    # After reading data
    if len(data) > 0:
        dim = len(data[0])
        for i, vec in enumerate(data):
            if len(vec) != dim:
                print_error_and_exit()

    centroids = kmeans(data, K, ITER)
    for centroid in centroids:
        print(",".join("%.4f" % x for x in centroid))
