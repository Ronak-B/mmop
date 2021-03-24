###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

import numpy as np
from . import cfunction as cf


class CF3(cf.CFunction):
    def __init__(self, dim):
        super(CF3, self).__init__(dim, 6)

        # Initialize data for composition
        self._sigma = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        self._bias = np.zeros(self._nofunc)
        self._weigh_ = np.zeros(self._nofunc)
        self._lambda = np.array([1.0/4.0, 1.0/10.0, 2.0, 1.0, 2.0, 5.0])

        # Lower/Upper Bounds
        self._lbound = -5.0 * np.ones(dim)
        self._ubound = 5.0 * np.ones(dim)

        if self.o.shape[1] >= dim:
            self._O = self.o[:self._nofunc, :dim]
        else:  # randomly initialize
            self._O = self._lbound + \
                      (self._ubound - self._lbound) * np.random.rand((self._nofunc, dim))

        # Load M_: Rotation matrices
        if dim in (2, 3, 5, 10, 20):
            fname = self.function_data_file(3, dim)  # "data/CF3_M_D{}.dat".format(dim)  # + str(dim) + ".dat"
            self._load_rotmat(fname)
        else:
            # M_ Identity matrices # TODO: Generate dimension independent rotation matrices
            self._M = [np.eye(dim)] * self._nofunc

        # Initialize functions of the composition
        self._function = {0: cf.fef8f2,
                          1: cf.fef8f2,
                          2: cf.weierstrass,
                          3: cf.weierstrass,
                          4: cf.grienwank,
                          5: cf.grienwank}

        # Calculate fmaxi
        self._calculate_fmaxi()

    def evaluate(self, x):
        return self._evaluate_inner(x)
