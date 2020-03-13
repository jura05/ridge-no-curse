import random 
import logging
from math import erf, sqrt
from time import time

import numpy as np
from numpy.polynomial import polynomial
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sympy


# Utility functions

def embed_polynomials_l2(p1, p2, l=1.0):
    """Find lambda for inclusion p1->p2, i.e., such that p1(t) ~ p2(t/lambda), |t|<l.
    
    Params:
        p1, p2  -- instances of polynomial.Polynomial class
        l       -- defines embedding segment [-l,l]
    """

    # we minimize S(mu) = int_{-l}^l |p1(t)-p2(mu t)|^2 dt
    # S(mu) is a polynomial in mu; to calculate it we use sympy and Polynomial class
    mu = sympy.Symbol('mu')
    assert p1.degree() == p2.degree()
    q = polynomial.Polynomial([p1.coef[i] - p2.coef[i] * mu**i for i in range(p1.degree())])
    q = q**2
    q = q.integ()
    S = q(l) - q(-l)
    S_coeff = [float(c) for c in reversed(sympy.Poly(S.expand()).all_coeffs())]  # sympy magic
    S_poly = polynomial.Polynomial(S_coeff)
    min_value, min_mu = minimize_polynomial(S_poly, -1, 1)
    return 1 / min_mu


def minimize_polynomial(poly, a, b):
    """Find minimal value and argmin for poly(t)->min, a <= t <= b."""

    # polynomial P has minimum either in P'(x)=0, or x=a, or x=b
    roots = poly.deriv().roots()
    real_roots = [root.real for root in roots if abs(root.imag) < 1e-8]
    active_roots = [root for root in real_roots if a <= root <= b]
    points = active_roots + [a, b]

    values = polynomial.polyval(points, poly.coef)
    min_idx = np.argmin(values)
    return values[min_idx], points[min_idx]


class RidgeSolver:
    """Recover a ridge function f(x)=phi(<a,x>) using f evaluations.

    Params:
        n       --  dimension of the problem
        f_eps   --  function with error
        a       --  true vector a (may be None; used for quality analysis)
        phi     --  true function phi (may be None; use for quality analysis)
    Also some techical params for internal use:
        M, N1, N2, N3, l, ...
    """

    def __init__(self, n, f_eps, M, N1, N2, N3, l=1.0, a=None, phi=None):
        self.n = n
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.M = M
        self.l = l
        self.f_eps = f_eps
        self.a = np.array(a) if a is not None else None
        self.phi = phi

    def get_random_unit_vector(self):
        v = np.array([random.gauss(0, 1) for _ in range(self.n)])
        return v / np.linalg.norm(v)

    def fit_polynomial(self, gamma):
        """Fit polynomial to a function phi(t_k <a,gamma>), |t_k|<=1."""
        ts = np.linspace(-1, 1, 2 * self.N1 + 1)
        ys = [self.f_eps(t * gamma) for t in ts]
        return polynomial.Polynomial.fit(ts, ys, deg=self.M)

    def check_fitting(self, gamma, poly):
        v_gamma = np.dot(gamma, self.a)
        ts = np.linspace(-self.l, self.l, 10)
        return max(abs(poly(t) - self.phi(v_gamma * t)) for t in ts)

    def solve(self):
        typical_gamma = self.step_get_typical_gamma()
        newa = self.step_approximate_a(typical_gamma)
        newphr = self.step_approximate_phi(newa)

    def step_get_typical_gamma(self):
        """Find gamma with 0.45<|v_gamma|<0.75."""

        N2 = self.N2

        # Generate N2 gammas
        gammas = [self.get_random_unit_vector() for _ in range(N2)]

        if self.a is not None:
            # Theoretical considerations about v_i (unknown in the reality
            real_v = sqrt(self.n) * np.array([np.dot(self.a, gamma) for gamma in gammas])
            real_abs_v = np.abs(real_v)

        # Second part of algorithm
        all_poly = [self.fit_polynomial(gamma) for gamma in gammas]  # polynom coefficients for all gammas

        logging.warning('start embeddings ...')
        embed_info = {i: {j: None for j in range(N2)} for i in range(N2)}  # if phi_i -> phi_j, insert[i][j] = corresponding lambda
        bound_minus = N2 * 0.4
        bound_plus = N2 *0.5
        v0 = -1
        best_err = 1000
        for j in range(N2):
            logging.warning('embed j=%d', j)
            am_good = 0
            for i in range(N2):
                vall = embed_polynomials_l2(all_poly[i], all_poly[j], l=self.l)
                if abs(vall) != 1:
                    embed_info[i][j] = vall
                if embed_info[i][j] is not None:
                    am_good += 1
            print('am_good', am_good)
            if am_good < bound_plus and am_good >= bound_minus:
                v0 = j
                break
            err = max(abs(am_good - bound_plus), abs(am_good - bound_minus))
            if err < best_err:
                best_err = err
                v0 = j

        if self.a is not None:
            print(real_abs_v[v0], "check that this number is in [0.45, 0.75]")
        if j == N2 - 1:
            print("Maybe error with v0")

        if 0:
            quality1 = 0
            max_quality1 = -100
            indmaxij = 0
            for i in range(N2):
                for j in range(N2):
                    if abs(real_v[j] / real_v[i]) > 1.01:
                        quality1 += abs(real_v[j]/real_v[i] - embed_info[i][j])
                        if abs(real_v[j]/real_v[i] - embed_info[i][j]) > max_quality1:
                            indmaxij = (i, j)
                            max_quality1 = abs(real_v[j]/real_v[i] - embed_info[i][j])
                        
            #print("Quality of lambdas: ", quality1, max_quality1, indmaxij)
            #errs = [abs(real_v[j]/real_v[i] - embed_info[i][j]) for i in range(N2) for j in range(N2)]
            #errs = np.array(errs)
        return gammas[v0]

    def step_approximate_a(self, gamma):
        """Approximate vector a.
        
        Params:
            gamma   --  vector with typical |v_gamma| < 3/4
        """

        n = self.n
        w = np.zeros(n)  # ws[k] will approximate a[k]*sqrt(n) / |v_gamma|

        poly0 = self.fit_polynomial(gamma)
        max_lambda = -1
        for i in range(self.n):
            ei = np.zeros(self.n)
            ei[i] = 1
            poly_ei = self.fit_polynomial(ei)
            lambda_i = embed_polynomials_l2(poly0, poly_ei, l=self.l)
            if abs(lambda_i) > max_lambda:
                max_lambda = abs(lambda_i)
                max_idx = i

        if self.a is not None:
            self.sign = 1 if self.a[max_idx] >= 0 else -1

        for i in range(n):
            if i == max_idx:
                w[i] = max_lambda
            else:
                fi = np.zeros(n)
                fi[max_idx] = 0.9
                fi[i] = 0.1
                poly_fi = self.fit_polynomial(fi)
                lambda_fi = embed_polynomials_l2(poly0, poly_fi, l=self.l)
                w[i] = 10*abs(lambda_fi) - 9*max_lambda

        newa = w / np.linalg.norm(w)
        if self.a is not None:
            a_compare = self.a * self.sign
            print("Approximation error of a, linf-norm:", max(abs(a_compare - newa)))
            print("Approximation error of a, l2-norm:", np.linalg.norm(a_compare - newa))

        return newa

    def step_approximate_phi(self, newa):
        ts = np.linspace(-1, 1, self.N3)

        values_phi = np.array([self.f_eps(t * newa) for t in ts])
        if self.phi is not None:
            values_phi_real = np.array([self.phi(t * self.sign) for t in ts])

        #plt.plot(ts, values_phi)
        #plt.plot(ts, values_phi_real - values_phi)
        #plt.show()
        if self.phi is not None:
            print("Approximation error of phi in C:", max(np.abs(values_phi - values_phi_real)))
        #print("Omega1", omega1())



def test_cosinus():
    n = 50
    seed = int(time() % 10000)
    print('seed:', seed)
    random.seed(seed)
    eps = 0.0001

    a = np.array([random.gauss(0, 1) for _ in range(n)])
    a = a / np.linalg.norm(a)
     
    N1 = 200
    M = 3
    N2 = 25
    N3 = 200
    gp = 2 ** random.random()
    gq = random.random() * 2 * np.pi
    l = 0.7

    def phi(x):
        return np.cos(gp * x + gq)

    def f(x):
        return phi(np.dot(a, x)) + eps * (2 * random.random() - 1)

    def f_eps(x):
        return f(x) + eps * (2 * random.random() - 1)

    solver = RidgeSolver(n=n, f_eps=f_eps, M=M, N1=N1, N2=N2, N3=N3, a=a, phi=phi)
    solver.solve()


#multiplies argument by sqrt(n) and calculates polynom with coefficients = coeff (highest power first)
def polynom_normed(coeff, x):
    x *= sqrt(n)
    res = 0
    power = 1
    for i in range(len(coeff)-1, -1, -1):
        res += coeff[i] *power
        power *= x
    return res


def quality1step(ind):
    tmax = min(1/real_abs_v[ind], 10)
    p = tmax * np.arange(-20, 20)/20.
    y = np.zeros(40)
    yy = np.zeros(40)
    v = gammas[ind] #np.ones(n)/np.sqrt(n)
    vgamma = sum(a * v) * np.sqrt(n)
    coeff = fit_polynomial(v)
    for i in range(40):
        #y[i] = f_changed(p[i] * v, gp, gq)
        #yy[i] = polynom_normed(coeff, p[i]/np.sqrt(n))
        y[i] = phi(vgamma * p[i], gp, gq)
        yy[i] = polynom_normed(coeff, p[i])
    return max(abs(yy - y))


def omega1():
    res = -100
    index = -1
    for i in range(N2):
        nqual = quality1step(i)
        if nqual > res:
            res = nqual
            index = i 
    return (res, index)


if __name__ == "__main__":
    test_cosinus()
