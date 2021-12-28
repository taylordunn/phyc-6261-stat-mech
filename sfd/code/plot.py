import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors
from simulate import *

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

def plot_seed_dependence(seed):
    kon_fixed = 1e-1 # Binding rate
    p_fixed = 0.5 # Fraction of sites occupied
    R_fixed = 1.0 # Ratio of binding to unbinding
    tmax = 10000
    L = 200
    N = int(p_fixed * L)
    koff = kon_fixed / R_fixed

    fig, axes = plt.subplots(figsize=(11,8), ncols=2)

    np.random.seed(seed)
    ring, part_pos, part_states = place_particles(N, L, R_fixed)
    ring_t1, disp_t1 = simulate(N, L, ring, part_pos, part_states, kon_fixed,
                                koff, tmax)
    np.random.seed(seed)
    ring, part_pos, part_states = place_particles(N, L, R_fixed)
    ring_t2, disp_t2 = simulate(N, L, ring, part_pos, part_states, kon_fixed,
                                koff, tmax)
    
    colors = sns.color_palette()
    dx1 = np.mean(disp_t1**2, axis=0)
    dx2 = np.mean(disp_t2**2, axis=0)
    axes[0].plot(dx1, label='Run 1', color=colors[0])
    axes[0].plot(dx2+5, label='Run 2 (+5)', color=colors[2])
    axes[0].legend(loc='upper left')
    
    axes[1].plot(dx1 - dx2)
    axes[0].set_xlabel('Time')
    axes[1].set_xlabel('Time')
    axes[0].set_ylabel('$<\Delta x^2>$')
    axes[0].set_title('Seed = {0}, $K_{{\mathrm{{on}}}} = {1:.1e}$, $p = {2:.1f}$, $R = {3:.1f}$, $L = {4}$'.format(seed, kon_fixed, p_fixed, R_fixed, L))
    axes[1].set_title('Difference between runs')

    fig.savefig('../plots/seed_dependence.pdf')

def plot_param_space():
    sns.set(style='whitegrid')

    #plot_seed_dependence(6261)

    data_directory = '../data/m100_tm10000_l200/'

    #kon_primary = [0.001, 0.01, 0.1, 1, 10, 100]
    kon_primary = [0.001, 0.01, 0.1]
    p_secondary = [0.1, 0.5, 0.9]
    R = 1.0
    tmax = 2000

    fig, axes = plt.subplots(figsize=(11,8), ncols=len(kon_primary),
                             nrows=len(p_secondary), sharex=True, sharey='row',
                             subplot_kw={'yscale': 'log', 'xscale': 'log'})

    for col,kon in enumerate(kon_primary):
        koff = kon / R
        for row,p in enumerate(p_secondary):
            filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
            t, dx2, dx2_err = np.loadtxt(data_directory + filename,
                                         delimiter=',', unpack=True)

            axes[row,col].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3)
            axes[row,col].plot(t, dx2, alpha=0.8)

            axes[row,col].set_title(r'$K_{{\rm on}} = {0:.1e}, p = {1:.1f}$' \
                                     .format(kon, p))
            
            scaling_func = lambda t,a,alpha: a * t**alpha
            a = 1.0

            popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
            axes[row,col].plot(t, scaling_func(t, *popt), 'k:',
                               label=r'$\alpha = {0:.2f}$' \
                                     .format(popt[1]))
            if col == 0:
                axes[row,col].plot(t[1:], scaling_func(t[1:], a, 0.5),
                               ls='--', color='k',
                               label='$<\Delta x^2> \sim \sqrt{t}$')
            else:
                axes[row,col].plot(t[1:], scaling_func(t[1:], a, 0.5),
                               ls='--', color='k')
            axes[row,col].legend(loc='upper left')

    axes[0,0].set_xlim([0, tmax])
    fig.tight_layout()
    fig.savefig('../plots/param_space_loglog_sfd_fit.pdf')

#def plot_correlation():
if __name__ == '__main__':
    sns.set(style='whitegrid')
    data_directory = '../data/per_particle/'
    kon_list = [0.01, 0.1, 1.0]
    R = 1.0
    p = 0.5
    L = 100
    M = 100
    tmax = 1000
    dt = 100

    fig, axes = plt.subplots(figsize=(20,10), ncols=len(kon_list),
                             sharex=True, sharey=True,
                             subplot_kw={'yscale': 'log', 'xscale': 'linear'})

    for col,kon in enumerate(kon_list):
        koff = kon / R
        filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
        folder = 'm{0}_tm{1}_l{2}/'.format(M, tmax, L)

        data = pd.read_csv(data_directory + folder + filename, comment='#',
                           names=['run', 'particle', 'time', 'dx', 'dx2'])
        data = data[data.particle > 0]
        for (run,particle),df in data.groupby(['run', 'particle']):
            t = df.time.values
            dx = df.dx.values
            dx2 = df.dx2.values
            tm = len(t)

            sum = np.zeros(dt)
            count = np.zeros(dt)
            for ti in range(tm-dt):
                for tj in range(ti,ti+dt):
                    dxi = dx[ti]
                    dxj = dx[tj]
                    sum[tj-ti] += dxi*dxj
                    count[tj-ti] += 1

            corr = sum / count
            
            axes[col].plot(np.arange(dt), corr, alpha=0.1)

        axes[col].set_title(r'$K_{{\rm on}} = {0:.1e}'.format(kon))
        axes[col].set_xlabel('$t$')
    axes[0].set_ylabel('$<\Delta x(t_0 + t) \Delta x(t_0)>$')
    axes[0].set_xlim([0, dt])

    fig.suptitle(r'$p = {1:.1f}, R = {2:.1f}$' \
                 .format(kon, p, R))
    fig.savefig('../plots/correlation.pdf')


def plot_L_dependence():
    sns.set(style='whitegrid')

    data_directory = '../data/avg_runs/'
    kon = 0.1
    R = 1.0
    koff = kon / R
    p = 0.5
    L_list = [10, 20, 50, 100, 200]
    M = 100
    tmax = int(1e3)

    LM_list = [(10,2000), (20,1000), (50,400), (100,200), (200,100)]
    fig, axes = plt.subplots(figsize=(20,10), ncols=len(L_list),
                             sharex=True, sharey=True,
                             subplot_kw={'yscale': 'log', 'xscale': 'log'})
    for col,(L,M) in enumerate(LM_list):
        filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
        folder = 'm{0}_tm{1}_l{2}/'.format(M, tmax, L)

        t, dx2, dx2_err = np.loadtxt(data_directory + folder + filename,
                                     delimiter=',', unpack=True)
        axes[col].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3)
        axes[col].plot(t, dx2, alpha=0.8)
        axes[col].set_title('L = {0:.0f}, M = {1:.0f}'.format(L, M))

        scaling_func = lambda t,a,alpha: a * t**alpha
        a = 1.0

        popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
        axes[col].plot(t, scaling_func(t, *popt), 'k:',
                           label=r'$\alpha = {0:.2f}$' \
                           .format(popt[1]))
        if col == 0:
            axes[col].plot(t[1:], scaling_func(t[1:], a, 0.5),
                               ls='--', color='k',
                               label='$<\Delta x^2> \sim \sqrt{t}$')
        else:
            axes[col].plot(t[1:], scaling_func(t[1:], a, 0.5),
                               ls='--', color='k')
        axes[col].legend(loc='upper left')
        axes[col].set_xlabel('$t$')
    
    axes[0].set_xlim([0, tmax])
    axes[0].set_ylabel('$<\Delta x^2>$')
    fig.suptitle(r'$K_{{\rm on}} = {0:.1e}, p = {1:.1f}, R = {2:.1f}$' \
                 .format(kon, p, R))
    fig.savefig('../plots/L_dependence_fit.pdf')

