import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors

#def plot_L_dependence():
if __name__ == '__main__':
    sns.set(style='whitegrid')

    data_directory = '../../data/avg_runs/'
    kon = 0.1
    R = 1.0
    p = 0.5
    filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)
    tmax = int(1e4)
    teq = 0
    LM_list = [(10, 2000), (20, 1000), (50, 400), (100, 200), (200, 100), (400, 50)]

    colors = dict(zip(LM_list, sns.color_palette('husl', len(LM_list))))
    fig, ax = plt.subplots(figsize=(5,5),
                             subplot_kw={'yscale': 'log', 'xscale': 'log'})
    scaling_func = lambda t,a,alpha: a * t**alpha
    for (L,M) in LM_list:
        color = colors[(L,M)]
        folder = 'm{0}_l{1}_tm{2}_te{3}/'.format(M, L, tmax, teq)

        t, dx2, dx2_err = np.loadtxt(data_directory + folder + filename,
                                     delimiter=',', unpack=True)
        ax.fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3, color=color)
        popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
        perr = np.sqrt(np.diag(pcov))
        ax.plot(t, dx2, alpha=0.8, color=color,
                label=r'$L = {0}, M = {1}: \alpha = {2:.2f}$' \
                        .format(L, M, popt[1]))
    
    ax.plot(t[1:], scaling_func(t[1:], 0.2, 0.5), ls='--', color='k',
                                label='$<\Delta x^2> \sim \sqrt{t}$')
    ax.legend(loc='upper left')
    ax.set_xlim([0, tmax])
    ax.set_xlabel('$t$')
    ax.set_ylabel('$<\Delta x^2>$')
    ax.set_title(r'$K_{{\rm on}} = 10^{{{0}}}, p = {1:.1f}, R = {2:.1f}$' \
                 .format(int(np.log10(kon)), p, R))
    fig.savefig('../../plots/L_dependence.png')


