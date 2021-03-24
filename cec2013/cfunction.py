###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################

from builtins import object
import os

import numpy as np

# UNCOMMENT APPROPRIATELY
# MINMAX = 1	# Minimization
MINMAX = -1		# Maximization


class CFunction(object):
    _dim = -1
    _nofunc = -1
    _C = 2000.0
    _M = None
    _weight = None
    _fi = None
    _z = None
    _f_bias = 0
    _fmaxi = None
    _tmpx = None

    def __init__(self, dim, nofunc):
        self._dim = dim
        self._nofunc = nofunc

        self._lbound = []
        self._ubound = []
        self._O = []
        self._lambda = []
        self._function = []
        self._bias = []
        self._sigma = []

        # Load optima
        self.path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(self.path, "data/optima.dat")
        self.o = np.loadtxt(file_path)

    def function_data_file(self, fn, dim):
        return os.path.join(self.path, "data/CF{}_M_D{}.dat".format(fn, dim))

    def evaluate(self, x):
        pass

    #
    # n.b. get_lbound / get_ubound don't appear to be used in current codebase
    #

    # try / except (IndexError) would be more pythonic to check rather than assert
    def get_lbound(self, ivar):
        assert 0 <= ivar < self._dim, ["ivar is not in valid variable range: %d not in [0,%d]" % ivar, self._dim]
        return self._lbound[ivar]

    # try / except (IndexError) would be more pythonic to check rather than assert
    def get_ubound(self, ivar):
        assert 0 <= ivar < self._dim, ["ivar is not in valid variable range: %d not in [0,%d]" % ivar, self._dim]
        return self._ubound[ivar]

    def _evaluate_inner(self, x):
        if self._function is None:
            raise NameError("Composition functions' dict is uninitialized")
        self._fi = np.zeros(self._nofunc)

        self._calculate_weights(x)
        for i in range(self._nofunc):
            self._transform_to_z(x, i)
            self._fi[i] = self._function[i](self._z)

        tmpsum = np.zeros(self._nofunc)
        for i in range(self._nofunc):
            tmpsum[i] = self._weight[i] * (self._C * self._fi[i] / self._fmaxi[i] + self._bias[i])

        return sum(tmpsum) * MINMAX + self._f_bias

    def _calculate_weights(self, x):
        self._weight = np.zeros(self._nofunc)
        for i in range(self._nofunc):
            mysum = sum((x-self._O[i])**2)
            self._weight[i] = np.exp(-mysum/(2.0 * self._dim * self._sigma[i] * self._sigma[i]))
        maxw = np.max(self._weight)
        # maxi = self._weight.argmax(axis=0)

        maxw10 = maxw**10
        for i in range(self._nofunc):
            if self._weight[i] != maxw:
                # if i != maxi:
                self._weight[i] = self._weight[i] * (1.0 - maxw10)

        mysum = np.sum(self._weight)
        for i in range(self._nofunc):
            if mysum == 0.0:
                self._weight[i] = 1.0 / (1.0 * self._nofunc)
            else:
                self._weight[i] = self._weight[i] / mysum

    def _calculate_fmaxi(self):
        self._fmaxi = np.zeros(self._nofunc)
        if self._function is None:
            raise NameError('Composition functions\' dict is uninitialized')

        x5 = 5 * np.ones(self._dim)

        for i in range(self._nofunc):
            self._transform_to_z_noshift(x5, i)
            self._fmaxi[i] = self._function[i](self._z)

    def _transform_to_z_noshift(self, x, index):
        # z_i = (x)/\lambda_i
        tmpx = np.divide(x, self._lambda[index])
        # Multiply z_i * M_i
        self._z = np.dot(tmpx, self._M[index])

    def _transform_to_z(self, x, index):
        # Calculate z_i = (x - o_i)/\lambda_i
        tmpx = np.divide((x - self._O[index]), self._lambda[index])
        # Multiply z_i * M_i
        self._z = np.dot(tmpx, self._M[index])

    def _load_rotmat(self, fname):
        self._M = []

        with open(fname, 'r') as f:
            tmp = np.zeros((self._dim, self._dim))
            cline = 0
            ctmp = 0
            for line in f:
                line = line.split()
                if line:
                    line = [float(i) for i in line]
                    # re initialize array when reached dim
                    if ctmp % self._dim == 0:
                        tmp = np.zeros((self._dim, self._dim))
                        ctmp = 0

                    # add line to tmp
                    tmp[ctmp] = line[:self._dim]
                    # if we loaded self._nofunc * self._dim-1 lines break
                    if cline >= self._nofunc * self._dim-1:
                        break
                    # add array to _M when it is fully created
                    if cline % self._dim == 0:
                        self._M.append(tmp)
                    ctmp = ctmp + 1
                    cline = cline + 1


# Sphere function
def sphere(x):
    return (x**2).sum()


# Rastrigin's function
def rastrigin(x):
    return np.sum(x**2-10.*np.cos(2.*np.pi*x)+10)


# Griewank's function
def grienwank(x):
    i = np.sqrt(np.arange(x.shape[0])+1.0)
    return np.sum(x**2)/4000.0 - np.prod(np.cos(x/i)) + 1.0


# Weierstrass's function
def weierstrass(x):
    alpha = 0.5
    beta = 3.0
    kmax = 20
    dimensions = x.shape[0]
    # exprf = 0.0

    c1 = alpha**np.arange(kmax+1)
    c2 = 2.0*np.pi*beta**np.arange(kmax+1)
    f = 0
    c = -dimensions*np.sum(c1*np.cos(c2*0.5))

    for i in range(dimensions):
        f += np.sum(c1*np.cos(c2*(x[i]+0.5)))
    return f + c


def f8f2(x):
    f2 = 100.0 * (x[0]**2 - x[1])**2 + (1.0 - x[0])**2
    return 1.0 + (f2**2)/4000.0 - np.cos(f2)


# FEF8F2 function
def fef8f2(x):
    dimensions = x.shape[0]
    f = 0
    for i in range(dimensions-1):
        f += f8f2(x[[i, i+1]] + 1)
    f += f8f2(x[[dimensions-1, 0]] + 1)
    return f
