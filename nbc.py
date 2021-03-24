from typing import List
import numpy as np
from scipy.special import softmax


def NBC(population):
    """Returns SEEDS ONLY"""
    n = len(population)
    if n == 1:
        return population

    mem = [[None]*n for _ in range(n)]

    def dist(i, j) -> float:
        if mem[i][j] is not None:
            return mem[i][j]
        x = p[i]
        y = p[j]
        t = np.sum(((x.val-y.val)**2))**.5
        mem[i][j] = t
        mem[j][i] = t
        return t

    p = sorted(population)  # better to worse, reverse lt
    edges = []
    for i in range(1, n):
        # ans = i
        # maxyet=0
        # for j in range(0,i):
        #     t = np.sum(((p[i].val-p[j].val)**2))
        #     if (t<maxyet):
        #         maxyet=t
        #         ans=j

        j = min(range(0, i), key=lambda j: dist(i, j))
        edges.append((i, j))

    mu = sum(mem[i][j] for i, j in edges) / len(edges)

    phi = 2

    seeds = [p[0]]  # Biggest agent is always a seed
    seeds.extend(p[i] for i, j in edges if mem[i][j] > phi * mu)

    return seeds


def permute_softmax(edges, edge_lengths, temp=1):
    soft = softmax(edge_lengths*temp)
    print(edge_lengths)
    assert (np.min(soft) > 0)

    permutation = np.random.choice(
        len(edges), len(edges), replace=False, p=soft)

    return [edges[i] for i in permutation]


def permute_softmax2(edges, edge_lengths, temp=1):

    ans = []

    while edges:

        soft = softmax(edge_lengths*temp)

        index = np.random.choice(
            len(edges), 1, replace=False, p=soft)[0]

        ans.append(edges[index])
        del edges[index]
        edge_lengths = np.delete(edge_lengths, index)

    return ans


def permute_softmax3(edges, edge_lengths, temp=1):
    ans = []
    count = 0

    while edges:

        soft = softmax(edge_lengths*temp)
        num_indices = len(edges) - np.sum(soft == 0)

        indices = np.random.choice(
            len(edges), num_indices, replace=False, p=soft)

        for i in indices:
            ans.append(edges[i])
        for index in sorted(indices, reverse=True):
            del(edges[index])
        edge_lengths = np.delete(edge_lengths, indices)

        # print(count)
        # count += 1

    return ans


def NBC_minsize(population, minsize, temp=-1, phi=1,num_clusters=float("inf")):  # phi = 1   MAGIC numbers
    p = sorted(population)  # better to worse, reverse lt
    n = len(p)

    if (n == 1):
        return [population]

    mem = {}

    def dist(i, j) -> float:
        if (i, j) in mem:
            return mem[(i, j)]
        if (j, i) in mem:
            return mem[(j, i)]
        x = p[i]
        y = p[j]
        t = np.sum(((x.val-y.val)**2))**.5
        mem[(j, i)] = t
        return t

    ans = {i: [] for i in range(0, n)}
    for i in range(1, n):
        t = min(range(0, i), key=lambda j: dist(i, j))
        ans[t].append(i)

    parent = [-1]*n
    edges = []
    for i in ans:
        for j in ans[i]: # i is better, arrow points upwards,child up parent
            parent[j] = i
            edges.append((i, j))

    mu = sum(dist(p, c) for p, c in edges)/len(edges)

    fol = {}

    def follow(node):
        if node not in fol:
            if node in ans:
                fol[node] = 1 + sum(follow(i) for i in ans[node])
            else:
                fol[node] = 1
        return fol[node]

    if phi!=0:
        edges = [e for e in edges if dist(e[0], e[1]) > phi * mu] # adding counter
    edge_lengths = np.array([dist(e[0], e[1]) for e in edges])
    # assert (np.min(edge_lengths) > 0)
    # max_len = np.max(edge_lengths)
    # edges = permute_softmax3(edges, edge_lengths, temp=temp)

    if (temp != -1):
        edges = permute_softmax3(edges, edge_lengths, temp=temp)
    else:
        edges.sort(key=lambda x: dist(x[0], x[1]), reverse=True)

    # if (temp==100000):# confirm
    #     print("Normal",edges)
    #     print("Inf",sorted(edges,key=lambda x: dist(x[0], x[1]), reverse=True))

    # if (dist(edges[0][0], edges[0][1]) == max_len):
    #     print("good")
    # else:
    #     print("bad")

    # edges.sort(key=lambda x: dist(x[0], x[1]), reverse=True)

    counter = 1

    for e in edges:
        if(counter >= num_clusters):
            break

        if follow(e[1]) >= minsize:
            t = e[0]
            while parent[t] != -1:
                t = parent[t]
            if follow(t) - follow(e[1]) >= minsize:
                counter += 1
                ans[e[0]].remove(e[1])
                parent[e[1]] = - 1
                t = e[0]
                while parent[t] != -1:
                    fol[t] -= follow(e[1])
                    t = parent[t]

    def rec(node, homies):
        homies.append(node)
        if node in ans:
            for i in ans[node]:
                rec(i, homies)

    species = []
    for s in range(n):
        if parent[s] == -1:  # if i is a seed
            l = []
            rec(s, l)
            species.append(l)

    species = [[p[j] for j in i] for i in species]
    return species
