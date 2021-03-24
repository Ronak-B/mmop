from scipy.spatial import Delaunay
import numpy as np

points=np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
tri = Delaunay(points)

def circumcenter(S,d):
    A=np.insert(S*2,d,1,axis=1)
    Ainv=np.linalg.inv(A)
    B=np.sum(np.square(S),axis=1)
    x=np.dot(Ainv,B)
    c=np.delete(x,d)
    return c

def circumradius(S,center):
    return np.sum(((S-center)**2))**.5

for i in tri.simplices:
    print(points[i])
    c=circumcenter(points[i],2)
    print(c)
    print(circumradius(points[i],c))