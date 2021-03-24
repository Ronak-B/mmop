import numpy as np
from nbc import NBC


def rand_elem(l):
    return l[int(np.random.uniform() * len(l))]


def rand1(x, spe, F):
    r1 = rand_elem(spe).val
    r2 = rand_elem(spe).val
    r3 = rand_elem(spe).val
    # assert(((r1 + F * (r2-r3)) > np.array([0.25, 0.25])).all())
    return r1 + F * (r2-r3)


def rand2(x, spe, F):
    r1 = rand_elem(spe).val
    r2 = rand_elem(spe).val
    r3 = rand_elem(spe).val
    r4 = rand_elem(spe).val
    r5 = rand_elem(spe).val
    # assert(((r1 + F * (r2-r3) + F * (r4-r5)) > np.array([0.25, 0.25])).all())

    return r1 + F * (r2-r3) + F * (r4-r5)


def keypoint1(x, spe, F):
    r1 = rand_elem(NBC(spe)).val
    r2 = rand_elem(spe).val
    r3 = rand_elem(spe).val
    # assert(((r1 + F * (r2-r3)) > np.array([0.25, 0.25])).all())

    return r1 + F * (r2-r3)


def keypoint2(x, spe, F):
    r1 = rand_elem(NBC(spe)).val
    r2 = rand_elem(spe).val
    r3 = rand_elem(spe).val
    r4 = rand_elem(spe).val
    r5 = rand_elem(spe).val
    # assert(((r1 + F * (r2-r3) + F * (r4-r5)) > np.array([0.25, 0.25])).all())

    return r1 + F * (r2-r3) + F * (r4-r5)
