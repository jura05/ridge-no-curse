
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
