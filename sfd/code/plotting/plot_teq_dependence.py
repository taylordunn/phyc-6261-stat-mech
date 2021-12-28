import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors, gridspec

#if __name__ == '__main__':
def plot_teq_dependence1():
    sns.set(style='whitegrid')

    data_directory = '../data/avg_runs/'
    kon = 0.1
    R = 1.0
    p = 0.5
    L = 200
    M = 100
    tmax = int(1e4)
    teq_list = [0, int(1e1), int(1e2), int(1e3), int(1e4)]

    colors = dict(zip(np.arange(len(teq_list)), sns.color_palette()))
    fig, ax = plt.subplots(figsize=(8,5),
                           subplot_kw={'yscale': 'log', 'xscale': 'log'})
    scaling_func = lambda t,a,alpha: a * t**alpha
    for col,teq in enumerate(teq_list):
        color = colors[col]
        filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
        folder = 'm{0}_l{1}_tm{2}_te{3}/'.format(M, L, tmax, teq)

        t, dx2, dx2_err = np.loadtxt(data_directory + folder + filename,
                                     delimiter=',', unpack=True)
        ax.fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3, color=color)
        popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
        ax.plot(t, dx2, alpha=0.8, color=color,
                label=r'$t_{{\rm eq}} = {0:.1e}: \alpha = {1:.2f}$' \
                       .format(teq, popt[1]))


    
    ax.plot(t[1:], scaling_func(t[1:], 0.2, 0.5), ls='--', color='k',
                                label='$<\Delta x^2> \sim \sqrt{t}$')
    ax.legend(loc='upper left')
    ax.set_xlim([0, tmax])
    ax.set_xlabel('$t$')
    ax.set_ylabel('$<\Delta x^2>$')
    ax.set_title(r'$K_{{\rm on}} = {0:.1e}, p = {1:.1f}, R = {2:.1f}$' \
                 .format(kon, p, R))
    fig.savefig('../plots/teq_dependence.pdf')

#def plot_teq_dependence1():
if __name__ == '__main__':
    sns.set(style='whitegrid')

    data_directory = '../../data/avg_runs/'
    kon = 0.1
    R = 1.0
    p = 0.5
    filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
    L = 200
    M = 100
    tmax = int(1e4)
    teq_list = [0, int(1e1), int(1e2), int(1e3), int(1e4)]

    colors = dict(zip(np.arange(len(teq_list)), sns.color_palette()))
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(5, 1, height_ratios=[6, 1, 1, 1, 1])

    axes = {}
    for i in range(5):
        axes[i] = plt.subplot(gs[i])
        axes[i].set_xlim([1, tmax])
        axes[i].set_xscale('log')
        if i < 4:
            axes[i].set_xticklabels(['', '', '', '', ''])


    folder = 'm{0}_l{1}_tm{2}_te{3}/'.format(M, L, tmax, 0)
    t, dx2_te0, dx2_err_te0 = np.loadtxt(data_directory + folder + filename,
                                     delimiter=',', unpack=True)

    scaling_func = lambda t,a,alpha: a * t**alpha
    for col,teq in enumerate(teq_list):
        color = colors[col]
        folder = 'm{0}_l{1}_tm{2}_te{3}/'.format(M, L, tmax, teq)

        t, dx2, dx2_err = np.loadtxt(data_directory + folder + filename,
                                     delimiter=',', unpack=True)
        axes[0].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3, color=color)
        popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
        if teq == 0:
            label = r'$t_{{\rm eq}} = 0: \alpha = {0:.2f}$' \
                    .format(popt[1])
        else:
            label = r'$t_{{\rm eq}} = 10^{{{0}}}: \alpha = {1:.2f}$' \
                    .format(int(np.log10(teq)), popt[1])
        axes[0].plot(t, dx2, alpha=0.8, color=color, label=label)

                axes[(row,col)].set_ylabel('Residual')
      if col > 0:
            axes[col].fill_between(t, -dx2_err_te0, dx2_err_te0,
                                   alpha=0.3, color=colors[0])
            axes[col].plot(t, np.zeros(len(dx2_te0)), alpha=0.8, color=colors[0])

            dx2 = dx2 - dx2_te0
            total_err = dx2_err + dx2_err_te0
            within_error = dx2[np.abs(dx2) < total_err]
            print(len(within_error) / len(dx2))

            axes[col].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3, color=color)
            axes[col].plot(t, dx2, alpha=0.8, color=color,
                    label='$<\Delta x^2> (t_{{eq}} = 10^{{{0}}}) - <\Delta x^2> (t_{{eq}} = 0)$'.format(int(np.log10(teq))))

            axes[col].set_ylabel('Residual')
            axes[col].set_ylim([-10, 10])
            axes[col].set_yticks([-10, -5, 0, 5, 10])
            #axes[col].legend(loc='upper left')
            axes[col].legend(bbox_to_anchor=(0.8, 1.15))

    axes[0].plot(t[1:], scaling_func(t[1:], 0.2, 0.5), ls='--', color='k',
                                label='$<\Delta x^2> \sim \sqrt{t}$')
    axes[0].set_yscale('log')
    axes[0].legend(loc='upper left')
    axes[4].set_xlabel('$t$')
    axes[0].set_ylabel('$<\Delta x^2>$')
    axes[0].set_title(r'$K_{{\rm on}} = 10^{{{0}}}, p = {1:.1f}, R = {2:.1f}$' \
                 .format(int(np.log10(kon)), p, R))
    fig.savefig('../../plots/teq_dependence_resid.png', bbox_inches='tight')

