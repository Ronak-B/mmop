###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

import numpy as np
from . import cfunction as cf


class CF1(cf.CFunction):

    def __init__(self, dim):
        super(CF1, self).__init__(dim, 6)

        # Initialize data for composition
        self._sigma = np.ones(self._nofunc)
        self._bias = np.zeros(self._nofunc)
        self._weight = np.zeros(self._nofunc)
        self._lambda = np.array([1.0, 1.0, 8.0, 8.0, 1.0/5.0, 1.0/5.0])

        # Lower/Upper Bounds
        self._lbound = -5.0 * np.ones(dim)
        self._ubound = 5.0 * np.ones(dim)

        # Load optima
        if self.o.shape[1] >= dim:
            self._O = self.o[:self._nofunc, :dim]
        else:  # randomly initialize
            self._O = self._lbound + \
                      (self._ubound - self._lbound) * np.random.rand((self._nofunc, dim))

        # M_: Identity matrices
        self._M = [np.eye(dim)] * self._nofunc

        # Initialize functions of the composition
        self._function = {0: cf.grienwank,
                          1: cf.grienwank,
                          2: cf.weierstrass,
                          3: cf.weierstrass,
                          4: cf.sphere,
                          5: cf.sphere}

        # Calculate fmaxi
        self._calculate_fmaxi()

    def evaluate(self, x):
        return self._evaluate_inner(x)
