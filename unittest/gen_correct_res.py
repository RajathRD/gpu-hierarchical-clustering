from sklearn.cluster import AgglomerativeClustering as AG
from sklearn.metrics import pairwise_distances
import numpy as np

datasets = []

with open('unittest/test_cases.txt', mode='r', encoding='utf8') as f:
    test_cases = f.read().split('\n\n')
    for test_case in test_cases:
        data = []
        test_input = test_case.split('\n')
        for sample in test_input[1:]:
            test_data = [int(x) for x in sample.split(' ')]
            data.append(test_data)
        datasets.append(np.array(data))

correct_res = []

for dataset in datasets:
    ag = AG(linkage='single', compute_distances=True).fit(dataset)
    correct_res.append(ag.children_.tolist())

with open('unittest/correct_res.txt', mode='w+', encoding='utf8') as f:
    for res in correct_res:
        f.write(str(res) + "\n")