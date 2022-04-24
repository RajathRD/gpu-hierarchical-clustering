from sklearn.cluster import AgglomerativeClustering as AG
from sklearn.metrics import pairwise_distances
import numpy as np

tests = []

with open('unittest/tests.txt', mode='r', encoding='utf8') as f:
    tests = f.read().split('\n')

for test in tests:
    dataset = None
    with open('unittest/tests/'+test, mode='r', encoding='utf8') as f:
        data = []
        for i, line in enumerate(f.readlines()):
            if i > 0:
                data.append([int(x) for x in line.split(' ')])
        dataset = np.array(data)

    correct_res = None

    ag = AG(linkage='single', compute_distances=True).fit(dataset)
    correct_res = ag.children_.tolist()

    with open('unittest/correct_outputs/'+test, mode='w+', encoding='utf8') as f:
        for merge in correct_res:
            new_clust = merge[0]
            old_clust = merge[1]
            f.write("({} <- {})\n".format(new_clust, old_clust))