# utils.py
from typing import List
import numpy as np
import shared
from cec2013.cec2013 import how_many_goptima
import pickle

def get_bounds(f):
    dim = f.get_dimension()
    ub = np.zeros(dim)
    lb = np.zeros(dim)
    # Get lower, upper bounds
    for k in range(dim):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)
    return ub, lb


def crossover(v, x, CR, f):
    j_rand = int(np.random.uniform() * len(x))
    t = 1 * (np.random.uniform(size=len(x)) <= CR)
    ans = v*t + (1-t)*x
    ans[j_rand] = v[j_rand]
    return Agent(ans, f)


class TerminationCondition(Exception):
    def __init__(self, where):
        self.where = where

    def __str__(self):
        return self.where

import functools

class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self, fget):
        self.fget = fget

        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value

# class NewAgent():
#     def __init__(self, x, f):
#         # print(shared.i,shared.MaxFes)

#         self.val = x
#         self.stagnation_count = 0
#         try:
#             # TODO: actually, check bounds 
#             self.t = f.evaluate(self.val)
#         except:
#             self.t = float('-inf')
        

#             # OR

#             # self.val = np.maximum(x, lb)
#             # self.val = np.minimum(x, ub)
#             #and try again

#             # print("math error:",x)        

#     def __lt__(self, other):
#         # inverse because of min heap
#         return self.fitness >= other.fitness

#     @lazy_property
#     def fitness(self):
#         if shared.i > shared.MaxFes:
#             print(shared.i)
#             raise TerminationCondition("YOLO")
#         else:
#             shared.i += 1

#         if hasattr(self.t, "__iter__"):
#             return self.t[0]
#         else:
#             return self.t


class Agent():
    def __init__(self, x, f):
        # print(shared.i,shared.MaxFes)

        self.val = x
        self.stagnation_count = 0
        try:
            # TODO: actually, check bounds 
            self.t = f.evaluate(self.val)
        except:
            print("maybeHappened",x)
            self.t = float('-inf')
        

            # OR

            # self.val = np.maximum(x, lb)
            # self.val = np.minimum(x, ub)
            #and try again

            # print("math error:",x)   
        if shared.i > shared.MaxFes:
            print(shared.i)
            raise TerminationCondition("YOLO")
        else:
            shared.i += 1

        if hasattr(self.t, "__iter__"):
            self.fitness = self.t[0]
        else:
            self.fitness = self.t     

    def __lt__(self, other):
        # inverse because of min heap
        return self.fitness >= other.fitness

   


def uniform_rand_init(lb, ub, dim, N, f) -> List[Agent]:
    return [Agent(np.random.uniform(lb, ub, size=dim), f) for _ in range(N)]

def judge(population, f):
    accuracy = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    print('jugding')
    for a in accuracy:
        count, seeds = how_many_goptima(population, f, a)
        print(
            # "In the current population there exist",
            count, " global optimizers at ",a)
        print("Global optimizers:", seeds)

def save(pop, name):
    dbfile = open("runs/"+name, 'ab')
    pickle.dump(pop, dbfile)
    dbfile.close()


def load(name):
    # for reading also binary mode is important
    dbfile = open("runs/"+name, 'rb')
    pop = pickle.load(dbfile)
    dbfile.close()
    return pop
