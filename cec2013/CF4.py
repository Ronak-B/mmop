###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

import numpy as np
from . import cfunction as cf


class CF4(cf.CFunction):

    def __init__(self, dim):
        super(CF4, self).__init__(dim, 8)

        # Initialize data for composition
        self._sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self._bias = np.zeros(self._nofunc)
        self._weight = np.zeros(self._nofunc)
        self._lambda = np.array([4.0, 1.0, 4.0, 1.0, 1.0/10.0, 1.0/5.0, 1.0/10.0, 1.0/40.0])

        # Lower/Upper Bounds
        self._lbound = -5.0 * np.ones(dim)
        self._ubound = 5.0 * np.ones(dim)

        if self.o.shape[1] >= dim:
            self._O = self.o[:self._nofunc, :dim]
        else:  # randomly initialize
            self._O = self._lbound + (self._ubound - self._lbound) * np.random.rand((self._nofunc, dim))

        # Load M_: Rotation matrices
        if dim in (2, 3, 5, 10, 20):
            fname = self.function_data_file(4, dim)  # os.path.join(my_path, "data/CF4_M_D{}.dat".format(dim))
            self._load_rotmat(fname)
        else:
            # M_ Identity matrices # TODO: Generate dimension independent rotation matrices
            self._M = [np.eye(dim)] * self._nofunc

        # Initialize functions of the composition
        self._function = {0: cf.rastrigin,
                          1: cf.rastrigin,
                          2: cf.fef8f2,
                          3: cf.fef8f2,
                          4: cf.weierstrass,
                          5: cf.weierstrass,
                          6: cf.grienwank,
                          7: cf.grienwank}

        # Calculate fmaxi
        self._calculate_fmaxi()

    def evaluate(self, x):
        return self._evaluate_inner(x)
