import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt
from math import erf, sqrt
from copy import copy
import scipy
from sympy import Symbol, real_roots, Poly

x = Symbol('x')
y = Symbol('y')

random.seed(78)
n = 50
eps = 0.0001
a = []
for i in range(n):
    a.append(random.random())
a = np.array(a)
a = a.astype(np.float32)
a /= np.sqrt(sum(a*a))
N1 = 200
M = 3
N2 = 25
N3 = 200
gp = 2 ** random.random()
gq = random.random() * 2 * np.pi
l = 0.7

def phi(x, p, q):
    return np.cos(p * x + q)#x**3 + 4*x + 1

def f(x, p, q):
    return phi(sum(x*a), p, q)

def gener_random_vect_hypersph():
    vals = []
    for i in range(n):
        gau = random.gauss(0, 1)
        vals.append(gau)
    vals = np.array(vals)
    vals /= np.sqrt(sum(vals*vals))
    return vals

def f_changed(x, p, q):
    err = random.random()
    return f(x, p, q) + 2 * eps * err - eps

#return coefficients of polynomial, highest power first -- approximation of function
def first_step(gamma):
    xks = []
    t_psi = []
    values = []
    for i in range(-N1, N1 + 1):
        xk = i * gamma / N1
        xks.append(xk)
        t_psi.append(i / N1)
        values.append(f_changed(xk, gp, gq))
    return np.polyfit(t_psi, values, deg = M)

#multiplies argument by sqrt(n) and calculates polynom with coefficients = coeff (highest power first)
def polynom_normed(coeff, x):
    x *= sqrt(n)
    res = 0
    power = 1
    for i in range(len(coeff)-1, -1, -1):
        res += coeff[i] *power
        power *= x
    return res

def coef_poly_min(c1, c2):
    global l
    c3 = []
    power = 1
    for i in range(len(c1)):
        c3.append(c1[i] - c2[i]*power)
        power *= y
    c4 = np.polynomial.polynomial.polypow(c3, 2)
    c5 = np.polynomial.polynomial.polyint(c4)
    integr = np.polynomial.polynomial.polyval(l, c5) - np.polynomial.polynomial.polyval(-l, c5)
    integr1 = integr.expand()
    coeff = Poly(integr1).all_coeffs()
    minn, indmin = minimum_polynom(coeff)
    return indmin


def minimum_polynom(coef):#from high
    global l
    diff_coef = np.polynomial.polynomial.polyder(coef[::-1])[::-1]
    poly = Poly.from_list(diff_coef, x)
    rroots = real_roots(poly)
    for i in range(len(rroots)):
        rroots[i] = float(rroots[i])
    rroots.append(l)
    rroots.append(-l)    
    values = np.polyval(coef, rroots)
    minn = min(values)
    isk_t = np.where(abs(values - minn) < 1e-8)[0][0]
    return float(minn), rroots[isk_t]


# /////   possible visualizations ///////// 
def graph_distribution():
    x = np.arange(0,2000)/200.
    y = np.zeros(2000)
    for i in range(2000):
        y[i] = F_star(x[i])
    plt.plot (x, y)
    plt.show()

def graph_first_step_works(ind):
    p = np.arange(-20, 20)/20.
    y = np.zeros(40)
    yy = np.zeros(40)
    v = gammas[ind] #np.ones(n)/np.sqrt(n)
    vgamma = sum(a * v) * np.sqrt(n)
    coeff = first_step(v)
    for i in range(40):
        #y[i] = f_changed(p[i] * v, gp, gq)
        #yy[i] = polynom_normed(coeff, p[i]/np.sqrt(n))
        y[i] = phi(vgamma * p[i], gp, gq)
        yy[i] = polynom_normed(coeff, p[i])
    plt.plot(p, y)
    plt.plot(p, yy)
    plt.show()
#////////////////////////////////////////////////////////

def quality1step(ind):
    tmax = min(1/real_abs_v[ind], 10)
    p = tmax * np.arange(-20, 20)/20.
    y = np.zeros(40)
    yy = np.zeros(40)
    v = gammas[ind] #np.ones(n)/np.sqrt(n)
    vgamma = sum(a * v) * np.sqrt(n)
    coeff = first_step(v)
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

#Generate N2 gammas
gammas = []
for i in range(N2):
    gammas.append(gener_random_vect_hypersph())

#Theoretical considerations about v_i
real_v = []#v_i unknown in the reality
for i in range(N2):
    real_v.append(sqrt(n) * sum(a * gammas[i]))
real_abs_v = []#abs(v_i) unknown in the reality
for i in range(N2):
    real_abs_v.append(abs(sqrt(n) * sum(a * gammas[i])))
real_abs_v = np.array(real_abs_v)

#Second part of algorithm
all_coef = [] #polynom coefficients for all gammas
for i in range(N2):
    all_coef.append(first_step(gammas[i]))
insert_info = np.zeros((N2, N2)) #if phi_i -> phi_j, insert[i][j] = corresponding lambda
bound_minus = N2 * 0.4
bound_plus = N2 *0.5
v0 = -1
best_err = 1000
for j in range(N2):
    am_good = 0
    for i in range(N2):
        vall = coef_poly_min(all_coef[j][::-1], all_coef[i][::-1])#optimize_f(N_0, i, j)
        if abs(vall) > 1:
            insert_info[i][j] = vall
        if insert_info[i][j] != 0:
            am_good += 1
    if am_good < bound_plus and am_good >= bound_minus:
        v0 = j
        break
    err = max(abs(am_good - bound_plus), abs(am_good - bound_minus))
    if err < best_err:
        best_err = err
        v0 = j
print(real_abs_v[v0], "check that this number is in [0.45, 0.75]")
if j == N2 - 1:
    print("Maybe error with v0")

quality1 = 0
max_quality1 = -100
indmaxij = 0
for i in range(N2):
    for j in range(N2):
        if abs(real_v[j] / real_v[i]) > 1.01:
            quality1 += abs(real_v[j]/real_v[i] - insert_info[i][j])
            if abs(real_v[j]/real_v[i] - insert_info[i][j]) > max_quality1:
                indmaxij = (i, j)
                max_quality1 = abs(real_v[j]/real_v[i] - insert_info[i][j])

            
#print("Quality of lambdas: ", quality1, max_quality1, indmaxij)
#errs = [abs(real_v[j]/real_v[i] - insert_info[i][j]) for i in range(N2) for j in range(N2)]
#errs = np.array(errs)
j_tilda = v0
coef_gamma_m = []
coef_ej_m = []


lambdasforai = []
coefgamma = first_step(gammas[j_tilda])
index_abs_max = 0
max_lambd =  -1000
for i in range(n):
    ei = np.zeros(n)
    ei[i] = 1
    coefei = first_step(ei)
    cur = coef_poly_min(coefei[::-1], coefgamma[::-1])
    lambdasforai.append(cur)
    if abs(cur) > max_lambd:
        max_lambd = abs(cur)
        index_abs_max = i
amax = lambdasforai[index_abs_max]
sign = 1
if amax < 0:
    sign = -1
ai = []
coefgamma = first_step(gammas[j_tilda])
cis = []
for i in range(n):
    if i == index_abs_max:
        ai.append(amax)
    else:
        ei = np.zeros(n)
        ei[index_abs_max] = 0.9
        ei[i] = 0.1
        coef_f = first_step(ei)
        cur = coef_poly_min(coef_f[::-1], coefgamma[::-1])
        if (sign == 1 and cur > 0) or (sign == -1 and cur < 0) or (sign == 2 and abs(cur) > 1):
            ci = 10 * (cur/lambdasforai[index_abs_max] - 0.9)
            cis.append(ci)
        else:
            print("problems with a_i, i = ", i, sep = "")

cis = np.array(cis)
amaxnew = np.sqrt(1 / (1+sum(cis*cis)))

alr = False
newa = []
for i in range(n):
    if i != index_abs_max:
        newa.append(amaxnew * cis[i - alr])
    else:
        newa.append(amaxnew)
        alr = True
newa = np.array(newa)
print("Approximation error of a, C-norm:", max(abs(a - newa)))
print("Approximation error of a, L2-norm:", np.sqrt(sum((a - newa)**2)))

ts = np.linspace(-1, 1, N3)
values_phi = []
values_phi_real = []

for t_ in ts:
    values_phi.append(f_changed(t_ * newa, gp, gq))
    values_phi_real.append(phi(t_, gp, gq))
values_phi = np.array(values_phi)
values_phi_real = np.array(values_phi_real)
#plt.plot(ts, values_phi)
plt.plot(ts, values_phi_real - values_phi)
plt.show()
print("Approximation error of phi in C:", max(abs(values_phi - values_phi_real)))
print("Approximation error of phi in L2:", np.sqrt(sum((values_phi - values_phi_real)**2)))
#print(gp, gq)
#print("Omega1", omega1())
