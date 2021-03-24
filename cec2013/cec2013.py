###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

from builtins import object

import math

import numpy as np

from . import functions as simple_functions
from . import CF1
from . import CF2
from . import CF3
from . import CF4


class CEC2013(object):
    _nfunc = -1
    _functions = {1: simple_functions.five_uneven_peak_trap,
                  2: simple_functions.equal_maxima,
                  3: simple_functions.uneven_decreasing_maxima,
                  4: simple_functions.himmelblau,
                  5: simple_functions.six_hump_camel_back,
                  6: simple_functions.shubert,
                  7: simple_functions.vincent,
                  8: simple_functions.shubert,
                  9: simple_functions.vincent,
                  10: simple_functions.modified_rastrigin_all,
                  11: CF1.CF1,
                  12: CF2.CF2,
                  13: CF3.CF3,
                  14: CF3.CF3,
                  15: CF4.CF4,
                  16: CF3.CF3,
                  17: CF4.CF4,
                  18: CF3.CF3,
                  19: CF4.CF4,
                  20: CF4.CF4}
    _f = None
    _fopt = [200.0, 1.0, 1.0, 200.0, 1.031628453489877, 186.7309088310239, 1.0, 2709.093505572820, 1.0, -2.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _rho = [0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.2, 0.5, 0.2, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    _nopt = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]
    _maxfes = [50000, 50000, 50000, 50000, 50000, 200000, 200000, 400000, 400000, 200000,
               200000, 200000, 200000, 400000, 400000, 400000, 400000, 400000, 400000, 400000]
    _dimensions = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

    def __init__(self, nofunc):
        assert 0 < nofunc <= 20
        self._nfunc = nofunc
        if 0 < self._nfunc < 11:
            self._f = self._functions[self._nfunc]
        else:
            self._f = self._functions[self._nfunc](self.get_dimension())

    def evaluate(self, function):
        function_ = np.asarray(function)
        assert len(function_) == self.get_dimension()
        if 0 < self._nfunc < 11:
            return self._f(function_)

        return self._f.evaluate(function_)

    def get_lbound(self, n):
        assert 0 <= n < self._dimensions[self._nfunc-1]
        result = 0
        if self._nfunc in (1, 2, 3):
            result = 0
        elif self._nfunc == 4:
            result = -6
        elif self._nfunc == 5:
            tmp = [-1.9, -1.1]
            result = tmp[n]
        elif self._nfunc in (6, 8):
            result = -10
        elif self._nfunc in (7, 9):
            result = 0.25
        elif self._nfunc == 10:
            result = 0
        elif self._nfunc > 10:
            result = self._f.get_lbound(n)
        return result

    def get_ubound(self, n):
        assert 0 <= n < self._dimensions[self._nfunc-1]
        result = 0
        if self._nfunc == 1:
            result = 30
        elif self._nfunc in (2, 3):
            result = 1
        elif self._nfunc == 4:
            result = 6
        elif self._nfunc == 5:
            tmp = [1.9, 1.1]
            result = tmp[n]
        elif self._nfunc in (6, 8):
            result = 10
        elif self._nfunc in (7, 9):
            result = 10
        elif self._nfunc == 10:
            result = 1
        elif self._nfunc > 10:
            result = self._f.get_ubound(n)
        return result

    def get_fitness_goptima(self):
        return self._fopt[self._nfunc-1]

    def get_dimension(self):
        return self._dimensions[self._nfunc-1]

    def get_no_goptima(self):
        return self._nopt[self._nfunc-1]

    def get_maxfes(self):
        return self._maxfes[self._nfunc - 1]

    def get_rho(self):
        return self._rho[self._nfunc-1]

    def get_info(self):
        return {'fbest': self.get_fitness_goptima(),
                'dimension': self.get_dimension(),
                'nogoptima': self.get_no_goptima(),
                'maxfes': self.get_maxfes(),
                'rho': self.get_rho()}


def how_many_goptima(pop, function, accuracy):

    npop = pop.shape[0]

    # Evaluate population
    fits = np.zeros(npop)
    for i in range(npop):
        fits[i] = function.evaluate(pop[i])

    # Descending sorting
    order = np.argsort(fits)[::-1]

    # Sort population based on its fitness values
    sorted_pop = pop[order, :]
    spopfits = fits[order]

    # find seeds in the temp population (indices!)
    seeds_idx = find_seeds_indices(sorted_pop, function.get_rho())

    count = 0
    goidx = []
    for idx in seeds_idx:
        # evaluate seed
        seed_fitness = spopfits[idx]

        if math.fabs(seed_fitness - function.get_fitness_goptima()) <= accuracy:
            count = count + 1
            goidx.append(idx)

        # save time
        if count == function.get_no_goptima():
            break

    # gather seeds
    seeds = sorted_pop[goidx]

    return count, seeds


def find_seeds_indices(sorted_pop, radius):
    seeds = []
    seeds_idx = []
    # Determine the species seeds: iterate through sorted population
    for i, x in enumerate(sorted_pop):
        found = False
        # Iterate seeds
        for _, sx in enumerate(seeds):
            # Calculate distance from seeds
            dist = math.sqrt(sum((x - sx)**2))

            # If the Euclidean distance is less than the radius
            if dist <= radius:
                found = True
                break
        if not found:
            seeds.append(x)
            seeds_idx.append(i)

    return seeds_idx
