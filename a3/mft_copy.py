import numpy as np
import matplotlib.pylab as plt

def bw_free_energy(n, m, h, kb, T):
    # Compute the Bragg-Williams free energy for the Ising model to n terms
    Gn = -m**2 / 2
    Gn -= h * m

    taylor_sum = 0
    for k in range(1, n+1):
        taylor_sum += m**(2*k) * (1/(2*k-1) - 1/(2*k))
    Gn += kb * T * taylor_sum

    return Gn

def bw_first_derivative(n, m, h, kb, T):
    Gn = -m
    Gn -= h

    taylor_sum = 0
    for k in range(1, n+1):
        taylor_sum += 2*k*m**(2*k - 1) * (1/(2*k-1) - 1/(2*k))
    Gn += kb * T * taylor_sum

    return Gn

def bw_binary_search(n, h, kb, T, m1, m2, target_accuracy=1e-3):
    # Employ the binary search method to find the minimum of the free energy
    accuracy = 1.0
    while accuracy > target_accuracy:
        f1 = bw_first_derivative(n, m1, h, kb, T)
        f2 = bw_first_derivative(n, m2, h, kb, T)
        assert f1 * f2 < 0
        m = (m1 + m2) / 2
        fmid = bw_first_derivative(n, m, h, kb, T)
        if fmid == 0.0:
            return m
        if fmid * f1 > 0:
            m1 = m
        else:
            m2 = m
        accuracy = np.abs(m1 - m2)

    return (m1 + m2) / 2

def exact_free_energy(m, h, kb, T):
    G = -m**2 / 2
    G -= h * m
    G += (1/2) * kb * T * ((1+m)*np.log(1+m) + (1-m)*np.log(1-m))
    return G

def exact_magnetization(m, h, kb, T):
    return np.tanh((1/(kb*T)) * (m + h))

def exact_binary_search(h, kb, T, m1, m2, target_accuracy=1e-3):
    # Employ the binary search method to find the minimum of the free energy
    accuracy = 1.0
    while accuracy > target_accuracy:
        f1 = m1 - exact_magnetization(m1, h, kb, T)
        f2 = m2 - exact_magnetization(m2, h, kb, T)
        assert f1 * f2 < 0
        m = (m1 + m2) / 2
        fmid = m - exact_magnetization(m, h, kb, T)
        if fmid == 0.0:
            return m
        if fmid * f1 > 0:
            m1 = m
        else:
            m2 = m
        accuracy = np.abs(m1 - m2)

    return (m1 + m2) / 2

def test():
    h = 0
    kb = 1.0
    n = 100

    m = np.linspace(-1, 1, 200)

    fig, ax = plt.subplots()
    Gn = bw_free_energy(n, m, h, kb, 1.5)
    ax.plot(m, Gn, label='1.5')
    Gn = exact_free_energy(m, h, kb, 1.5)
    ax.plot(m, Gn, label='1.5 (exact)')
    m1 = binary_search(n, h, kb, 1.5, -0.5, 0.5)
    #x = np.array([m])
    #y = bw_free_energy(n, m, h, kb, 1.5)
    ax.scatter(m1, bw_free_energy(n, m1, h, kb, 1.5))
    #ax.scatter(m, bw_free_energy(n, m, h, kb, 1.5), marker='o', color='k',
               #label='m = {0:.2f}'.format(m[0]))

    Gn = bw_free_energy(n, m, h, kb, 1.0)
    ax.plot(m, Gn, label='1.0')
    Gn = exact_free_energy(m, h, kb, 1.0)
    ax.plot(m, Gn, label='1.0 (exact)')
    #dGn = bw_first_derivative(n, m, h, kb, 1.0)
    #ax.plot(m, dGn, label='d 1.0')
    m1 = binary_search(n, h, kb, 1.0, -0.5, 0.5)
    ax.scatter(m1, bw_free_energy(n, m1, h, kb, 1.0), marker='+')

    Gn = bw_free_energy(n, m, h, kb, 0.77)
    ax.plot(m, Gn, label='0.77')
    Gn = exact_free_energy(m, h, kb, 0.77)
    ax.plot(m, Gn, label='0.77 (exact)')
    #dGn = bw_first_derivative(n, m, h, kb, 0.77)
    #ax.plot(m, dGn, label='d 0.77')
    m1 = binary_search(n, h, kb, 0.77, -1, -0.01)
    m2 = binary_search(n, h, kb, 0.77, 0.01, 1)
    m12 = np.array([m1, m2])
    ax.scatter(m12, bw_free_energy(n, m12, h, kb, 0.77))
    #ax.scatter(m, bw_free_energy(n, m, h, kb, 0.77), '+', color='k',
               #label='m = {0:.2f}, {1:.2f}'.format(m1,m2))
    ax.legend()

    plt.show()

if __name__ == '__main__':
    h = 0
    kb = 1.0
    n = 100

    m = np.linspace(-1, 1, 200)

    fig, ax = plt.subplots()

    T_list = np.arange(0.05, 1.5, 0.05)
    bw_root_list = []
    exact_root_list = []
    for T in T_list:
        print(T)
        # If T < Tc, the root is nontrivial
        if T < 0.99999:
            #Gn = bw_free_energy(n, m, h, kb, T)
            #ax.plot(m, Gn, label=str(T))
            bw_root = bw_binary_search(n, h, kb, T, 0.1, 2)
            exact_root = exact_binary_search(h, kb, T, 0.1, 2)
        else:
            #Gn = bw_free_energy(n, m, h, kb, T)
            #ax.plot(m, Gn, label=str(T))
            bw_root = bw_binary_search(n, h, kb, T, -0.5, 0.5)
            exact_root = exact_binary_search(h, kb, T, -0.5, 0.5)
        bw_root_list.append(bw_root)
        exact_root_list.append(exact_root)

    plt.plot(T_list, bw_root_list, label='Approximate')
    plt.plot(T_list, exact_root_list, label='Exact')

    ax.legend()
    plt.show()

