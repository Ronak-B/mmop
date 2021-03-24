
import sys

import shared
import math
from cec2013.cec2013 import CEC2013, how_many_goptima

import time
from typing import List
import numpy as np
import heapq
from itertools import chain
from nbc import NBC_minsize
from mutations import rand1, rand2, keypoint1, keypoint2
from utils import get_bounds,  TerminationCondition, uniform_rand_init, Agent, crossover, judge, save, load
from wGrad import wgrad
# k = 4
# f = CEC2013(k)
# print(f.get_info())
# dim = f.get_dimension()
# shared.i = 0
# shared.MaxFes = f.get_maxfes()
# ub, lb = get_bounds(f)


def goodness(s, p):
    def avg_fitness(s):
        return sum(i.fitness for i in s)/len(s)
    bias = 2**0.5  # TODO good BIAS VALUE?
    f = avg_fitness(s)
    f += bias * np.sqrt(np.log(p/len(s)))
    return f


def select_best(specie):
    s = NBC_minsize(specie, 0, temp=-1)  # TODO MAGIC PARAM
    p = sum(len(i) for i in s)
    return max(s, key=lambda i: goodness(i, p))
# assert max(i.fitness for i in best) == best[0].fitness


def generate_MAB_iter_rec(specie, num):
    copy = specie[:]
    for i in range(num):
        s = generate_MAB_rec(select_best(copy), 1)
        copy.extend(s)
    return copy[-num:]


def generate_MAB_iter(specie, num):
    copy = specie[:]
    # print(copy)
    for i in range(num):
        s = generate(select_best(copy), 1)
        copy.extend(s)
    return copy[-num:]


def generate_MAB_rec(specie, num, base_size=5, repeat_size=10000):
    if(len(specie) <= base_size or len(specie) >= (repeat_size-5)):
        return generate(select_best(specie), num)

    return generate_MAB_rec(select_best(specie), num, repeat_size=len(specie))


def generate_MAB(specie, num):
    return generate(select_best(specie), num)


def generate(s, n):
    noise = np.random.normal(0, 0.1, size=(len(s[0].val), n))
    w = np.copy(s[0].val)
    w = w.reshape((w.shape[0], 1))
    t = noise + w

    l = lb.reshape((lb.shape[0], 1))
    u = ub.reshape((ub.shape[0], 1))

    t = np.maximum(t, l)
    t = np.minimum(t, u)
    t = t.T
    return [Agent(t[i], f) for i in range(n)]


def generate_old(s, n):
    noise = np.random.normal(0, 0.1, size=(len(s[0].val), n))
    w = np.copy(s[0].val)
    w = w.reshape((w.shape[0], 1))
    t = noise + w

    x = [p.val for p in s]
    lb = np.min(x,  axis=0)
    ub = np.max(x,  axis=0)

    lb = lb.reshape((lb.shape[0], 1))
    ub = ub.reshape((ub.shape[0], 1))

    t = np.maximum(t, lb)
    t = np.minimum(t, ub)
    t = t.T
    return [Agent(t[i], f) for i in range(n)]


def within_bounds(ans):
    if ((ans <= ub).all() and (ans >= lb).all()):
        return True
    else:
        return False


def mutate(x, spe):
    type1 = np.random.uniform()
    type2 = np.random.uniform()
    # global i
    alpha = 0.5  # Magic numbers
    F1 = np.random.uniform(0.2, 0.8)  # Magic
    F2 = 0.5  # Magic
    per = 1 - (shared.i / shared.MaxFes)**alpha

    if type1 < per:
        if type2 < 0.5:
            v = rand1(x.val, spe, F1)
            while (not within_bounds(v)):
                v = rand1(x.val, spe, F1)
        else:
            v = rand2(x.val, spe, F2)
            while (not within_bounds(v)):
                v = rand2(x.val, spe, F2)

    else:
        if type2 < 0.5:
            v = keypoint1(x.val, spe, F1)
            while (not within_bounds(v)):
                v = keypoint1(x.val, spe, F1)
        else:
            v = keypoint2(x.val, spe, F2)
            while (not within_bounds(v)):
                v = keypoint2(x.val, spe, F2)

    # assert((v < ub).all())
    # assert((v > lb).all())

    return v


def balance(species):
    nums = [len(i) for i in species]
    rest = 0
    Lambda = 2  # Magic Numbers
    mu_avg = sum(nums)/len(nums)
    mu_lambda = round(Lambda * mu_avg)

    s = []
    for i in range(len(nums)):
        t = nums[i]
        if t > mu_lambda:
            rest += t - mu_lambda
            nums[i] = mu_lambda
        elif t < mu_avg:
            s.append(i)

    temp = int(rest/len(s))
    for i in s:
        nums[i] += temp

    if len(s) == 0:
        rest = 0
    else:
        rest %= len(s)

    i = 0
    while rest > 0:
        nums[s[i]] += 1
        i += 1
        rest -= 1

    return nums


def dist(x: Agent, y: Agent) -> float:
    return np.sum(((x.val-y.val)**2))**.5


def nearest(x, population, M):
    return heapq.nsmallest(M+1, population, key=lambda y: dist(x, y))


def solve(f, lb, ub, dim, temp=-1, num_clusters=-1, T=20, M_factor=1, gen_mult=1, generate_strategy=generate_MAB):  # ,pop_hint=-1
    CR = 0.9  # MAGIC
    MaxGens = (200 if dim < 5 else 300)*gen_mult

    NP = math.ceil(shared.MaxFes/MaxGens)  # intialize pop

    if (num_clusters != -1):
        # NP = min(5 + MaxGens // 2, max(10, 3*dim))
        NP = int(num_clusters * max(10, 3*dim) * 2)  # let minsize be

    # if (pop_hint != -1):
    #     NP = max(math.ceil(shared.MaxFes/MaxGens),
    #              int(pop_hint * max(10, 3*dim) * 2))
    print(NP)
    g = 0
    try:
        population = uniform_rand_init(lb, ub, dim, NP, f)
        # old = time.time()
        # for i in population:
        #     print(i.val)
        #print(i.val for i in population)
        archive = []
        bestworse = []
        alpha=0.6
        while True:  # g < MaxGens:
            genz = []
            # for i in population:
            #     print(i.val)
            minsize = min(5+g//2, max(10, 3*dim))  # TODO Magic numbers!!!
            #print(i.val for i in population)
            # if (num_clusters != -1):
            #     species = NBC_minsize(
            #         population, minsize, phi=0, temp=temp, num_clusters=num_clusters)
            # else:
            #     species = NBC_minsize(population, minsize, temp=temp)
            feval=[0 for i in range(0,len(population))]
            for i,j in enumerate(population):
                feval[i]=f.evaluate(j.val)
            species=wgrad(f,population,alpha,feval)
            alpha-=0.005
            if alpha<=0.4:
                alpha=0.4
            # for i in species:
                #print(len(i),end=' ')
            #print('\n')
            # better to worse, reverse lt
            species = [sorted(x) for x in species]

            avg_fit = sum(x[0].fitness for x in species) / len(species)
            print("Avg fitness:",avg_fit,"No. of species:", len(species),"Progress:", shared.i/shared.MaxFes * 100)
            # # print(len(species), NP, minsize, num_clusters)

            #nums = balance(species)  # RETURN SAME FLAG
            nums= [len(i) for i in species]
            for s, n in zip(species, nums):
                for i in range(n):
                    x = s[i]
                    #    blah = shared.i
                    v = mutate(x, s)
                     #   print(shared.i-blah)
                    u = crossover(v, x.val, CR, f)

                    x.stagnation_count += 1  # Added

                        # if u.val[0]!=v[0]:
                        #     print("nice")

                        # if ((u.val <= ub).all() and (u.val >= lb).all()):
                        #     pass
                        # else:
                        #     print("happened",u.val)
                        #     u.fitness = float('-inf')
                        # TODO: try generating new ind here

                    w = u if u.fitness > x.fitness else x
                    genz.append(w)

                    # else:
                    #     ws = generate_strategy(s, n-len(s))
                    #     genz.extend(ws)
                    #     break

            population = genz

            if True:

                population = {i: True for i in population}

                # M = int(4 + np.random.uniform() * (21-4))
                # M = sum(nums)/len(nums) # TODO use minsize?
                M = (1-M_factor)*minsize + M_factor*sum(nums)/len(nums)

                for x in population:
                    if population[x] and x.stagnation_count >= T:
                        # print("Archived")
                        neighbours = nearest(x, population, math.floor(M))
                        worse = [i for i in neighbours if i.fitness < x.fitness]
                        worse.append(x)  # x is last

                        bestworse.append(x)  # x is last

                        for i in worse:
                            population[i] = False
                        archive.append(worse)
                # print(population)

                population = [i for i in population if population[i]]
                # print(population)
                # before = shared.i
                population.extend(uniform_rand_init(lb, ub, dim, NP-len(population), f))
                # check_i(before, 0)

            g += 1
            # print("                         ", time.time()-old)
            # old=time.time()

    except (TerminationCondition, KeyboardInterrupt):
        # return np.array([i.val for i in population])

        # print("gens:", g)
        return np.array([i.val for i in chain(population, bestworse)])


# if __name__ == "__main__":
#     #  print(f.evaluate([1]))
#     seed = 42
#     np.random.seed(seed)
#     name = f"fbkde_{k}_{seed}"
#     run = True
#     if run:
#         pop = solve(f, lb, ub, dim)
#         try:
#             save(pop, name)
#         except Exception as e:
#             print(e)
#     else:
#         pop = load(name)
#     judge(pop, f)

# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate


def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return (mean, float('nan'), float('nan'))  # float('nan')
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    return (mean, variance, sampleVariance)


def tester(solver, file_name):
    num_runs = 3

    PR = [(0, 0, 0)]*5
    SR = [(0, 0, 0)]*5

    num_optima = f.get_no_goptima()

    for n in range(num_runs):
        print("Test run no.:",n+1,"started")
        shared.i = 0
        pop = solver()
        accuracy = [0.1, 0.01, 0.001, 0.0001, 0.00001]

        print("-"*20)

        print("Test run no.:",n+1 ,"completed")

        for index, a in enumerate(accuracy):
            count, seeds = how_many_goptima(pop, f, a)

            # PR[index] += (count-PR[index])/(n+1)
            # SR[index] += ((1 if count == num_optima else 0) -
            #               SR[index])/(n+1)

            PR[index] = update(PR[index], count/num_optima)
            SR[index] = update(SR[index], 1 if count == num_optima else 0)

            print(count," optimas found at accuracy ",a)


            # PR = map(lambda x: x/num_optima, PR)
            # print(*PR, sep=', ')
            # print(*SR, sep=', ')

        print("-"*20)


    PR = [finalize(x) for x in PR]
    SR = [finalize(x) for x in SR]

    with open(file_name, 'a') as the_file:
        # the_file.write('Hello\n')
        print(*PR, sep=', ', file=the_file)
        print(*SR, sep=', ', file=the_file)

    print("Results written to file: ",file_name)


if __name__ == "__main__":

    test = False

    if test:

        # k, gen_strat, temp, T, gen_mult, M_factor = eval(sys.argv[1])
        k, phi_g, gen_strat, temp, T, M_factor = eval(sys.argv[1])
        gen_mult = 1

        # # range(4,8):
        # for x in [7]:
        #     for test_temp in [-1]:  # , 0, 0.5, 1, 2, 5, 8, 10, 15, 50, 100

        # k = x
        f = CEC2013(k)
        # print(k, test_temp, ':', f.get_info())
        # file_name = str(x)+'-'+str(test_temp)+'.txt'
        file_name = 'results/' + sys.argv[1]
        with open(file_name, 'a') as the_file:

            print(k, temp, ':', f.get_info(), file=the_file)
        dim = f.get_dimension()
        shared.i = 0
        shared.MaxFes = f.get_maxfes()
        ub, lb = get_bounds(f)

        if(gen_strat == 'old'):
            generate_strategy = generate_old
        elif (gen_strat == 'new'):
            generate_strategy = generate
        elif (gen_strat == 'm'):
            generate_strategy = generate_MAB
        elif (gen_strat == 'mr'):
            generate_strategy = generate_MAB_rec
        elif (gen_strat == 'mi'):
            generate_strategy = generate_MAB_iter
        elif (gen_strat == 'mir'):
            generate_strategy = generate_MAB_iter_rec

        try:
            tester(lambda: solve(f, lb, ub, dim,
                                 temp=temp, T=T, M_factor=M_factor,
                                 gen_mult=gen_mult, generate_strategy=generate_strategy), file_name)
        except Exception as e:
            # print("FAILED:", e)
            with open(file_name, 'a') as the_file:

                print("FAILED:", e, file=the_file)

    else:

        k = 9

        f = CEC2013(k)
        print(f.get_info())
        dim = f.get_dimension()
        shared.i = 0
        shared.MaxFes = f.get_maxfes()
        ub, lb = get_bounds(f)

        seed = 42
        np.random.seed(seed)
        # name = f"fbkde_{k}_{seed}" #TODO restore or find alternative
        name = "fbkde_{" + str(k) + "}_{" + str(seed) + "}"

        run = True

        if run:
            pop = solve(f, lb, ub, dim)
            try:
                save(pop, name)
            except Exception as e:
                print(e)
        else:
            pop = load(name)
        judge(pop, f)
