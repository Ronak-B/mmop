import numpy as np 
from mpl_toolkits import mplot3d
import networkx as nx
import matplotlib.pyplot as plt
import shared,math
from cec2013.cec2013 import CEC2013 
from utils import uniform_rand_init,get_bounds


f=CEC2013(20)

shared.i = 0
shared.MaxFes = f.get_maxfes()
dim=f.get_dimension()
ub,lb=get_bounds(f)
print(ub,lb)
NP = math.ceil(shared.MaxFes/200)
#NP=200
seed=99
np.random.seed(seed)

#randomly generate population
population=uniform_rand_init(lb,ub,dim,NP,f)

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
    #G=nx.DiGraph()
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
    #nx.draw(G,nx.get_node_attributes(G,'pos'),node_size=10)
    #plt.scatter(x,y)
    plt.show()
    for i in species:
        print(len(i),end=' ')
    
NBC_minsize(population,2)