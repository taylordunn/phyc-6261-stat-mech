import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from matplotlib import colors
import time

def place_particles(N, L, R, initial_condition='ratio'):
    # Given the inital condition, set intial number of free and bound particles
    if initial_condition == 'ratio':
        Nb = int(N * R / (1+R))
        Nf = N - Nb
    elif initial_condition == 'bound':
        Nb = N
        Nf = 0
    elif initial_condition == 'unbound':
        Nf = N
        Nb = 0

    # Initialze the relevant arrays
    ring = np.zeros(L) # -1 = bound, 0 = unoccupied, 1 = unbound
    part_pos = np.zeros(N) # Particle positions on the ring
    part_states = np.zeros(N) # Particle states

    for i in range(N):
        # Keep picking a site until it's unoccupied
        site = np.random.randint(L)
        while ring[site] != 0:
            site = np.random.randint(L)

        part_pos[i] = site
        if i < Nb:
            # Place the particle on the ring in a bound state
            ring[site] = -1
            part_states[i] = -1
        else:
            # Otherwise place it unbound
            ring[site] = 1
            part_states[i] = 1

    return ring, part_pos, part_states

def simulate(N, L, ring, part_pos, part_states, kon, koff, tmax):
    # First determine the number of particles that are bound/unbound
    Nf = len(part_states[part_states == 1])
    Nb = N - Nf

    # The time step is the average time for each particle to take one step
    # Each Monte Carlo step will therefore increment the time by 1/N
    nsteps = N * tmax
    dt = 1 / N 

    # Use arrays to keep track of ring state and displacement versus time
    ring_t = np.zeros([L, tmax+1])
    ring_t[:,0] = ring
    part_disp = np.zeros(N)
    disp_t = np.zeros([N, tmax+1])

    # Use the Gillespie algorithm to determine the first event
    gamma = Nb*koff + Nf*kon # Sum of rates
    tnext = - (1/gamma) * np.log(np.random.random_sample())
    Pb = Nf*kon / gamma # Probability of binding

    t = 0.0
    for n in range(1, nsteps+1):
        if t >= tnext:
            # Determine what kind of event
            if np.random.random_sample() < Pb:
                # Binding event
                # Get the indicies of the unbound particles
                unbound = np.where(part_states == 1)[0]
                # Randomly choose the particle to be bound
                i = np.random.choice(unbound)
                part_states[i] = -1
                ring[part_pos[i]] = -1
                Nb += 1
                Nf -= 1
            else:
                # Unbinding event
                bound = np.where(part_states == -1)[0]
                i = np.random.choice(bound)
                part_states[i] = 1
                ring[part_pos[i]] = 1
                Nf += 1
                Nb -= 1

            # Find the next event
            gamma = Nb*koff + Nf*kon
            tnext = t - (1/gamma) * np.log(np.random.random_sample())
            Pb = Nf*kon / gamma

        # Pick a random particle to move
        i = np.random.randint(N)
        
        # If the particle is free
        if part_states[i] == 1:
            # Choose a direction
            dx = np.random.choice([-1, 1])
            
            new_pos = (part_pos[i] + dx) % L # Periodic boundary conditions
            
            # If the new position is unoccupied
            if ring[new_pos] == 0:
                # Update the ring occupancy
                ring[new_pos] = 1
                ring[part_pos[i]] = 0
                # Move the particle and update displacement
                part_pos[i] = new_pos
                part_disp[i] += dx

        # Update the time
        t += dt

        # If a full time step has passed (each particle has been moved once on
        #  average), record position and displacement
        if n % N == 0:
            ring_t[:,int(n/N)] = ring
            disp_t[:,int(n/N)] = part_disp

    return ring_t, disp_t

def plot_single_runs():
    L = 200 # Number of sites
    N = int(p_fixed * L) # Number of particles
    tmax = 1000 # Number of time steps

    # Values to test
    kon_primary = [10**i for i in range(-3,4)]
    R_fixed = 1.0 # Ratio of binding to unbinding
    p_fixed = 0.5 # Fraction of sites occupied

    fig, axes = plt.subplots(figsize=(20,7), ncols=len(kon_primary), nrows=2,
                             sharex=True, sharey='row')

    # Some settings for the discrete color map
    cmap = colors.ListedColormap(['red', 'white', 'blue'])
    bounds = [-1, -0.5, 0.5, 1] # red = bound, white = unoccupied, blue = free
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for col,kon in enumerate(kon_primary):
        koff = kon / R_fixed
        ring, part_pos, part_states = place_particles(N, L, R_fixed)
        ring_t, disp_t = simulate(N, L, ring, part_pos, part_states, kon,
                                  koff, tmax)


        axes[0,col].imshow(ring_t, origin='lower', cmap=cmap, norm=norm,
                           aspect='auto')
        axes[1,col].plot(np.mean(disp_t**2, axis=0))

        axes[0,col].set_title(r'$K_{{\rm on}} = {0:.1e}$'.format(kon))
        axes[1,col].set_xlabel('$t$')

    axes[0,0].set_ylim([0, L])
    axes[1,0].set_xlim([0, tmax])
    axes[0,0].set_ylabel('$x$')
    axes[1,0].set_ylabel('$<\Delta x^2>$')
    #fig.suptitle(r'$P = {0:.1f}$, $R = {1:.1f}$'.format(p_fixed, R_fixed))
    fig.tight_layout()
    fig.savefig('single_runs.pdf')

def plot_run_dependence():
    kon_fixed = 1e-1 # Binding rate
    R_fixed = 1.0
    koff_fixed = kon_fixed / R_fixed # Unbinding rate

    L = 200 # Number of sites
    tmax = 1000 # Number of time steps

    # Values to test
    p_secondary = [0.1, 0.5, 0.9]
    M_values = [5, 10, 100]

    fig, axes = plt.subplots(figsize=(20,7), ncols=len(M_values),
                             nrows=len(p_secondary), sharex=True, sharey='row')

    for col,M in enumerate(M_values):
        for row,p in enumerate(p_secondary):
            print('Simulating p = {0:.1f}, {1} runs'.format(p, M))

            N = int(p * L) # Number of particles

            # Hold the average particle displacements in an array
            disp = np.empty([M, tmax+1])

            start_time = time.time()
            for run in range(M):
                ring, part_pos, part_states = place_particles(N, L, R_fixed)
                ring_t, disp_t = simulate(N, L, ring, part_pos, part_states,
                                          kon_fixed, koff_fixed, tmax)
                disp[run,:] = np.mean(disp_t**2, axis=0)
            run_time = time.time() - start_time

            # Calculate mean and SEM
            avg = np.mean(disp, axis=0)
            err = np.std(disp, axis=0) / np.sqrt(disp.shape[0])

            axes[row,col].fill_between(np.arange(len(avg)), avg-err, avg+err,
                                       alpha=0.5)
            axes[row,col].plot(np.arange(len(avg)), avg)
    
            axes[row,col].set_title(r'$p = {0:.1f}$, {1} runs, {2:.1f} seconds' \
                                    .format(p, M, run_time))
            if col == 0:
                axes[row,col].set_ylabel('$<\Delta x^2>$')

        axes[-1,col].set_xlabel('$t$')


    axes[0,0].set_xlim([0, tmax])
    fig.tight_layout()
    fig.savefig('run_dependence.pdf')

def plot_param_space():
    L = 200 # Number of sites
    tmax = 1000 # Number of time steps
    M = 100 # Number of runs

    # Values to test
    kon_primary = [10**i for i in range(-3,4)]
    p_secondary = [0.1, 0.5, 0.9]
    R_fixed = 1.0

    # Define an exponential function for fitting the data
    fit_func = lambda t,a,alpha: a * t**alpha
    
    fig, axes = plt.subplots(figsize=(20,7), ncols=len(kon_primary),
                             nrows=len(p_secondary), sharex=True, sharey='row')

    data = {}
    for col,kon in enumerate(kon_primary):
        koff = kon / R_fixed
        for row,p in enumerate(p_secondary):
            print('Simulating kon = {0:.1e}, p = {1:.1f}'.format(kon, p))

            N = int(p * L) # Number of particles

            # Hold the average particle displacements in an array
            disp = np.empty([M, tmax+1])

            for run in range(M):
                ring, part_pos, part_states = place_particles(N, L, R_fixed)
                ring_t, disp_t = simulate(N, L, ring, part_pos, part_states,
                                          kon, koff, tmax)
                disp[run,:] = np.mean(disp_t**2, axis=0)

            # Calculate mean and SEM
            avg = np.mean(disp, axis=0)
            err = np.std(disp, axis=0) / np.sqrt(disp.shape[0])

            t = np.arange(len(avg))
            axes[row,col].fill_between(t, avg-err, avg+err, alpha=0.3)
            axes[row,col].plot(t, avg, alpha=0.5)

            # Fit the data and plot
            popt, pcov = curve_fit(fit_func, t, avg, [1, 0.5])
            axes[row,col].plot(t, fit_func(t, *popt), 'k--',
                               label=r'$a = {0:.2f}$, $\alpha = {1:.2f}$' \
                                     .format(popt[0], popt[1]))
    
            axes[row,col].legend(loc='upper left')
            axes[row,col].set_title(r'$K_{{\rm on}} = {0:.1e}, p = {1:.1f}$' \
                                    .format(kon, p))
            if col == 0:
                axes[row,col].set_ylabel('$<\Delta x^2>$')

            # Record the data, to be returned
            data[(kon,p)] = (avg,err,popt[0],popt[1])

        axes[-1,col].set_xlabel('$t$')


    axes[0,0].set_xlim([0, tmax])
    fig.tight_layout()
    fig.savefig('param_space.pdf')
    return data

start_time = time.time()

#plot_single_runs()
#plot_run_dependence()
#data = plot_param_space()
data = quick_plot()

run_time = time.time() - start_time
print('Finished running after {0:.1f} seconds'.format(run_time))
