import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

#def plot_param_space2():
if __name__ == '__main__':
    sns.set(style='whitegrid')

    M = 100
    L = 200
    tmax = int(1e4)
    teq = 0

    data_directory = '../../data/avg_runs/m{0}_l{1}_tm{2}_te{3}/' \
                     .format(M, L, tmax, teq)

    kon_primary = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    p_secondary = [0.1, 0.5, 0.9]
    R = 1.0

    fig, axes = plt.subplots(figsize=(10,11), nrows=len(kon_primary),
                             ncols=len(p_secondary), sharex=True, sharey=True,
                             subplot_kw={'yscale': 'log', 'xscale': 'log'})

    fits = {0.1: ([], []), 0.5: ([], []), 0.9: ([], [])}
    scaling_func = lambda t,a: a * t**0.5
    for row,kon in enumerate(kon_primary):
        koff = kon / R
        axes[row,0].set_ylabel('$<\Delta x^2>$')
        for col,p in enumerate(p_secondary):
            print(kon,p)
            filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
            t, dx2, dx2_err = np.loadtxt(data_directory + filename,
                                         delimiter=',', unpack=True)

            axes[row,col].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3)
            axes[row,col].plot(t, dx2, alpha=0.8)

            axes[row,col].set_title(r'$K_{{\rm on}} = 10^{{{0}}}, p = {1:.1f}$' \
                                     .format(int(np.log10(kon)), p))
            

            t_fit = t[t > 1e2]
            dx2_fit = dx2[t > 1e2]
            sigma_fit = dx2_err[t > 1e2]
            popt, pcov = curve_fit(scaling_func, t_fit, dx2_fit, [dx2[1]],
                                   sigma=sigma_fit)
            perr = np.sqrt(np.diag(pcov))
            axes[row,col].plot(t, scaling_func(t, *popt), 'k--',
                               label=r'$A = {0:.2f} \pm {1:.1e}$' \
                                     .format(popt[0], perr[0]))
            fits[p][0].append(popt[0])
            fits[p][1].append(perr[0])

            axes[row,col].legend(loc='best')
            axes[-1,col].set_xlabel('$t$')

    axes[0,0].set_xlim([0, tmax])
    fig.tight_layout()
    fig.savefig('../../plots/param_space.png', rasterized=True)

    fig, axes = plt.subplots(ncols=2, figsize=(10,7),
                             sharey=True)
    kon = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    fits2 = {1e-3: ([], []), 1e-2: ([], []), 1e-1: ([], []), 1e0: ([], []),
             1e1: ([], []), 1e2: ([], []), 1e3: ([], [])}
    for p,(popt,perr) in fits.items():
        axes[0].errorbar(kon, popt, yerr=perr, marker='o',
                    label='$p = {0:.1f}$'.format(p))
    fits_kon = {}
    colors = sns.color_palette('husl', len(kon))
    scaling_func = lambda x,a,gamma: a*np.exp(-gamma * x)
    p_vals = np.array([0.1, 0.5, 0.9])
    #axins = zoomed_inset_axes(axes[1], 5, loc=1)
    for i,k in enumerate(kon):
        fits_kon[k] = ([], [])
        for p,(popt,perr) in fits.items():
            fits_kon[k][0].append(popt[i])
            fits_kon[k][1].append(perr[i])

        popt, pcov = curve_fit(scaling_func, p_vals, fits_kon[k][0],
                               [1, 0.5])
        perr = np.sqrt(np.diag(pcov))
        axes[1].plot(p_vals, scaling_func(p_vals, *popt), ls='--',
                     color=colors[i])
        print(k, popt)

        axes[1].errorbar(p_vals, fits_kon[k][0], yerr=fits_kon[k][1],
                         marker='o', color=colors[i], ls='None',
                         label=r'$K_{{\rm on}} = 10^{{{0}}}: \gamma = {1:.2f} \pm {2:.2f}$' \
                                .format(int(np.log10(k)), popt[1], perr[1]))
        """
        axins.errorbar(p_vals, fits_kon[k][0], yerr=fits_kon[k][1],
                         marker='o', color=colors[i], ls='None')
        axins.plot(p_vals, scaling_func3(p_vals, *popt), ls='--',
                     color=colors[i])
        """

    axes[0].set_xscale('log')
    #axes[1].set_xscale('log')
    axes[1].set_xlim([0, 1])
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xlabel(r'$K_{\rm on}$')
    axes[1].set_xlabel(r'$p$')
    axes[0].set_ylabel('$A$')
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    
    axes[1].text(0.6, 1, '$A(p) \sim \mathrm{e}^{-\gamma p}$')

    """
    axins.set_xlim([0.85, 0.95])
    axins.set_yscale('log')
    axins.set_ylim([0.005, 1e-1])
    """

    fig.savefig('../../plots/param_space_fits.png', bbox_inches='tight')
