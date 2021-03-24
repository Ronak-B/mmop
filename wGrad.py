import numpy as np 
from mpl_toolkits import mplot3d
import networkx as nx
import matplotlib.pyplot as plt
import shared,math
from cec2013.cec2013 import CEC2013 
from utils import uniform_rand_init,get_bounds




#f=CEC2013(20)
def wgrad(f,population,alpha,feval):
    # shared.i = 0
    # shared.MaxFes = f.get_maxfes()
    #dim=f.get_dimension()
    #ub,lb=get_bounds(f)
    #print(ub,lb)
    NP = len(population)
    #seed=2
    #np.random.seed(seed)

    #randomly generate population
    #population=uniform_rand_init(lb,ub,dim,NP,f)
    def distance(a,b):
        sum=0
        for i in range(0,len(a)):
            sum+=(a[i]-b[i])**2
        return math.sqrt(sum)

    def gradient(a,b,dist):
        if dist==0:
            print(population[a].val,population[b].val)
            print(distance(population[a].val,population[b].val))
        return (feval[b]-feval[a])/dist
        #return (population[b].fitness-population[a].fitness)/dist
    #assign unique number to each agent
    #ht={i: j for i,j in enumerate(population)}

    #directed graph 
    dg=[[] for i in range(0,NP)]
    #feval=[0 for i in range(0,NP)]

    

    maxDist=-math.inf
    maxGrad=-math.inf
    minDist=math.inf
    minGrad=math.inf
    distMat=np.empty((NP,NP))
    gradMat=np.empty((NP,NP))

    #print('check1')

    for i in range(0,NP):
        for j in range(0,NP):
            if i!=j and distance(population[i].val,population[j].val)!=0:
                distMat[i][j]=distance(population[i].val,population[j].val)
                #if distMat[i][j]==0:
                    #print('lllllllllllllllll')
                gradMat[i][j]=gradient(i,j,distMat[i][j])
                maxDist=max(distMat[i][j],maxDist)
                maxGrad=max(gradMat[i][j],maxGrad)
                minDist=min(distMat[i][j],minDist)
                minGrad=min(gradMat[i][j],minGrad)
                #print('c')
                
    #print(maxDist,maxGrad,minDist,minGrad)
    cv=-math.inf
    #G=nx.DiGraph()
    # for i,j in enumerate(population):
    #     G.add_node(i,pos=j.val)
        #print(j.val)
    #print('check2')
    for i in range(0,NP):
        cv=-math.inf
        edge=(0,0)
        for j in range(0,NP):
            if i!=j and distance(population[i].val,population[j].val)!=0:
                normDist=(maxDist-distMat[i][j])/(maxDist-minDist)
                normGrad=(gradMat[i][j]-minGrad)/(maxGrad-minGrad)
                ncv=(1-alpha)*normGrad+normDist*alpha
                #print(ncv)
                if(ncv>cv):
                    #print(ncv,cv)
                    cv=ncv
                    edge=(i,j)
                    #print(edge)
                
        e1,e2=edge
        #G.add_edge(e1,e2)
        #print(edge)
        dg[e1].append(e2)
        dg[e2].append(e1)
    # x=[i.val[0] for i in population]
    # y=[i.val[1] for i in population]
    # z=[f.evaluate(i.val) for i in population] 
    #nx.draw(G,nx.get_node_attributes(G,'pos'),node_size=10)
    species=[]
    vis=[False for i in range(0,len(dg))]

    def dfs(i):
        vis[i]=True
        #spec.append(i)
        #print(i)
        spec.append(population[i])
        for j in range(0,len(dg[i])):
            if vis[dg[i][j]]==False:
                dfs(dg[i][j])
    #print('call')
    species=[]
    spec=[]
    for i in range(0,len(dg)):
        if vis[i]==False:
            dfs(i)
        if len(spec)>0:
            #print(len(spec))
            species.append(spec)
        spec=[]
    #print('\n')
    #for i in species:
        #print(len(i),end=" ")
    return species
#plt.scatter(x,y)
#plt.show()
#print(dg)





