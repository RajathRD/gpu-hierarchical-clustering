from sklearn.metrics import pairwise_distances
import numpy as np
import sys

tests = []

with open('unittest/tests.txt', mode='r', encoding='utf8') as f:
    tests = f.read().split('\n')

for test in tests:
    print(test)
    N = 0
    M = 0
    dataset = None
    cluster_labels = dict()
    with open('unittest/tests/'+test, mode='r', encoding='utf8') as f:
        data = []
        for i, line in enumerate(f.readlines()):
            if i == 0:
                N = int(line.split(' ')[0])
                M = int(line.split(' ')[1])
            else:
                data.append([int(x) for x in line.split(' ')])
                cluster_labels[i-1] = [i-1]
        dataset = np.array(data)
    
    # Compute distance matrix
    dists = pairwise_distances(dataset)
    #print(dists)

    for iter in range(N-1):
        # Iterate over the remaining clusters and find the two clusters to be merged
        min_cluster1 = -1
        min_cluster2 = -1
        min_dist = sys.maxsize
        for cluster1 in cluster_labels.keys():
            for cluster2 in cluster_labels.keys():
                if cluster1 != cluster2 and (cluster1 != min_cluster2 and cluster2 != min_cluster1):
                    for v1 in cluster_labels[cluster1]:
                        for v2 in cluster_labels[cluster2]:
                            if dists[v1][v2] < min_dist:
                                min_dist = dists[v1][v2]
                                min_cluster1 = cluster1
                                min_cluster2 = cluster2

        # Merge clusters by updating dists and cluster_labels
        cluster_labels[min_cluster1] = cluster_labels[min_cluster1] + cluster_labels[min_cluster2]
        cluster_labels.pop(min_cluster2)
        print("({} <- {})".format(min_cluster1, min_cluster2))     
        #print(cluster_labels)   

#    with open('unittest/correct_outputs/'+test, mode='w+', encoding='utf8') as f:
#        for merge in correct_res:
#            new_clust = merge[0]
#            old_clust = merge[1]
#            f.write("({} <- {})\n".format(new_clust, old_clust))