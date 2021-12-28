import numpy as np
import random as ran
import matplotlib.pylab as plt
import time
import seaborn as sns # Makes plots look nicer
from scipy.optimize import curve_fit

def step():
    # Discrete, unbiased step in 2 dimensions. Left, right, up or down
    dr = ran.choice([[-1, 0], [1, 0], [0, 1], [0, -1]])
    return np.array(dr)

def saw(t_sample):
    # Performs a self-avoiding random walk in 2 dimensions until in collides
    #  with itself. Returns R^2 values at the given sample steps

    # Start at the origin
    r = np.zeros(2)
    dr = np.zeros(2)
    r2_sample = []
    t_max = np.max(t_sample)

    # Keep track of all positions that have been visited
    visited = [list(r)]

    for t in range(1, t_max+1):
        # Generate a new 2d step
        new_dr = step()
        # Prevent backtracking to save computation time
        while (new_dr == -dr).all():
            new_dr = step()

        dr = new_dr
        r += dr

        # Break the loop if this position has already been visited
        if list(r) in visited:
            break
        else:
            visited.append(list(r))
            # Increment step number and record length if at a sample point
            if t in t_sample:
                r2_sample.append(r.dot(r))

    return r2_sample 

def sample(N, t_sample):
    # Performs N self-avoiding random walks, sampling at the given steps

    # This list will contain R^2 samples at each time sample point. We use
    #  lists instead of arrays beause we don't know the array size a priori
    r2_samples = [ [] for i in range(len(t_sample)) ]

    for n in range(N):
        r2_sample = saw(t_sample)

        # Loop over time points and append to the appropriate list
        for t,r2 in enumerate(r2_sample):
            r2_samples[t].append(r2)
        
        # Print out progress at powers of 10
        if np.log10(n+1).is_integer():
            print('Step %d' % (n+1))

    return r2_samples

def plot_length():
    sns.set(style='whitegrid') # Comment out if not using seaborn
    N = int(1e8)
    t_sample = [2**i for i in range(11)]

    r2_samples = sample(N, t_sample)

    fig, ax = plt.subplots(figsize=(11,8))

    r2_mean = []
    r2_err = []
    r2_std = []
    counts = []
    for t,r2 in enumerate(r2_samples):
        if len(r2) > 1:
            r2_mean.append(np.mean(r2))
            r2_std.append(np.std(r2))
            r2_err.append(np.std(r2) / np.sqrt(len(r2)))
            counts.append(len(r2))
    
    ax.errorbar(t_sample[:len(r2_mean)], r2_mean, yerr=r2_err,
                marker='o', ls='None')
    # Save the data to a text file as well as plotting
    data = np.asarray([t_sample[:len(r2_mean)], r2_mean, r2_std, r2_err, counts])
    np.savetxt('saw.csv', data, delimiter=',')

    power_law = lambda x, nu: x**nu
    L = np.array(t_sample[:len(r2_mean)])
    # First, the scaling of a typical random walk
    R = power_law(L, 0.5)
    ax.plot(L, R**2, '-', label=r'$\nu = 0.5$')
    # Second, the two-dimensional theoretical exponent (given by Sethna)
    R = power_law(L, 0.75)
    ax.plot(L, R**2, '--', label=r'$\nu = 0.75$'))
    # Third, a least squares fit
    popt, pcov = curve_fit(power_law, L, np.sqrt(r2_mean))
    nu = popt[0]
    R = power_law(L, nu)
    ax.plot(L, R**2, ':', label=r'$\nu = %.3f$' % nu)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('$R^2$')
    ax.set_xlabel('$t$')

    ax.legend()
    fig.savefig('saw.pdf', bbox_inches='tight')

start_time = time.time()
plot_length()
print('Run time = %.1f seconds' % (time.time() - start_time))

