import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import colors, gridspec

def plot_group1():
    sns.set(style='whitegrid')

    group_dict = {'Taylor': '../../data/avg_runs/',
                  'JP': '../../data/JP/',
                  'Stephen': '../../data/stephen/'}
    kon = 0.1
    R = 1.0
    p = 0.5
    L = 200
    M = 100
    tmax = int(1e4)
    teq_list = [0, int(1e4)]
    filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)

    name_color = dict(zip(group_dict.keys(), sns.color_palette()))
    teq_col = dict(zip(teq_list, np.arange(len(teq_list)))) 
    fig, axes = plt.subplots(ncols=len(teq_list), figsize=(10,5),
                             sharex=True, sharey=True, 
                             subplot_kw={'yscale': 'log', 'xscale': 'log'})
    scaling_func = lambda t,a,alpha: a * t**alpha
    t_plot = np.logspace(0, 4, 10)
    for teq in teq_list:
        col = teq_col[teq]
        sub_folder = 'm{0}_l{1}_tm{2}_te{3}/'.format(M, L, tmax, teq)
        for name,folder in group_dict.items():
            if name == 'Stephen' and teq == 0:
                continue
            color = name_color[name]

            t, dx2, dx2_err = np.loadtxt(folder + sub_folder + filename,
                                         delimiter=',', unpack=True)
            t, dx2, dx2_err = t[1:], dx2[1:], dx2_err[1:]
            if name == 'JP':
                dx2_err /= np.sqrt(M)

            axes[col].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3,
                                   color=color)
            popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
            axes[col].plot(t, dx2, alpha=0.8, color=color,
                           label=name + r'$: \alpha = {0:.2f}$'.format(popt[1]))
        
        if teq == 0:
            axes[col].set_title(r'$t_{{\rm eq}} = 0$')
        else:
            axes[col].set_title(r'$t_{{\rm eq}} = 10^{{{0}}}$'.format(int(np.log10(teq))))
        axes[col].plot(t_plot, scaling_func(t_plot, 0.2, 0.5), ls='--', color='k',
                                label='$<\Delta x^2> \sim \sqrt{t}$')

        axes[col].legend(loc='upper left')
        axes[col].set_xlim([0, tmax])
        axes[col].set_xlabel('$t$')
    axes[0].set_ylabel('$<\Delta x^2>$')
    fig.suptitle(r'$K_{{\rm on}} = 10^{{{0}}}, p = {1:.1f}, R = {2:.1f}, L = {3}, M = {4}$ runs' \
                 .format(int(np.log10(kon)), p, R, L, M))
    fig.savefig('../../plots/group.png', bbox_inches='tight')

#def plot_group_resid():
if __name__ == '__main__':
    sns.set(style='whitegrid')

    group_dict = {'Taylor': '../../data/avg_runs/',
                  'JP': '../../data/JP/',
                  'Stephen': '../../data/stephen/'}
    kon = 0.1
    R = 1.0
    p = 0.5
    L = 200
    M = 100
    tmax = int(1e4)
    teq_list = [0, int(1e4)]
    filename = 'k{0}_p{1}_r{2}.csv'.format(kon, p, R)

    #name_color = dict(zip(group_dict.keys(), sns.color_palette()))
    name_color = dict(zip(['Taylor', 'JP', 'Stephen'], sns.color_palette()))
    teq_col = dict(zip(teq_list, np.arange(len(teq_list)))) 

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])
    axes = {}
    for col in range(2):
        for row in range(3):
            axes[(row,col)] = plt.subplot(gs[row,col])
            axes[(row,col)].set_xscale('log')
            axes[(row,col)].set_xlim([1, tmax])
            if row < 2:
                axes[(row,col)].set_xticklabels(['']*5)
            if row == 0:
                axes[(row,col)].set_yscale('log')
                if col == 1:
                    axes[(row,col)].set_yticklabels(['']*4)
            else:
                axes[(row,col)].set_ylim([-5, 5])
                axes[(row,col)].set_yticks([-10, -5, 0, 5, 10])
                if col == 1:
                    axes[(row,col)].set_yticklabels(['']*5)
                else:
                    axes[(row,col)].set_ylabel('Residual')
    
    scaling_func = lambda t,a,alpha: a * t**alpha
    t_plot = np.logspace(0, 4, 10)
    for teq in teq_list:
        col = teq_col[teq]
        sub_folder = 'm{0}_l{1}_tm{2}_te{3}/'.format(M, L, tmax, teq)
        #for name,folder in group_dict.items():
        for name in ['Taylor', 'JP', 'Stephen']:
            folder = group_dict[name]
            if name == 'Stephen' and teq == 0:
                continue
            color = name_color[name]

            t, dx2, dx2_err = np.loadtxt(folder + sub_folder + filename,
                                         delimiter=',', unpack=True)
            t, dx2, dx2_err = t[1:], dx2[1:], dx2_err[1:]
            if name == 'JP':
                dx2_err /= np.sqrt(M)

            axes[(0,col)].fill_between(t, dx2-dx2_err, dx2+dx2_err, alpha=0.3,
                                   color=color)
            popt, pcov = curve_fit(scaling_func, t, dx2, [1, 0.5])
            axes[(0,col)].plot(t, dx2, alpha=0.8, color=color,
                           label=name + r'$: \alpha = {0:.2f}$'.format(popt[1]))

            if name == 'Taylor':
                t_base, dx2_base, dx2_err_base = t, dx2, dx2_err
                for row in range(1, 3):
                    axes[(row,col)].fill_between(t_base, -dx2_err_base, dx2_err_base,
                                             alpha=0.3, color=color)
                    axes[(row,col)].plot(t_base, np.zeros(len(dx2_base)),
                                       alpha=0.8, color=color)

            else:
                matching_vals = np.array([tb in t for tb in t_base])
                dx2 = dx2 - dx2_base[matching_vals]
                total_err = dx2_err + dx2_err_base[matching_vals]
                within_error = dx2[np.abs(dx2) < total_err]
                agree = len(within_error) / len(dx2)
                print(agree)

                if name == 'JP':
                    row = 1
                    label = 'JP - Taylor, {0:.1%} agreement'.format(agree)
                else:
                    row = 2
                    label = 'Stephen - Taylor, {0:.1%} agreement'.format(agree)


                axes[(row,col)].fill_between(t, dx2-dx2_err, dx2+dx2_err,
                                             alpha=0.3, color=color)
                axes[(row,col)].plot(t, dx2, alpha=0.8, color=color,
                                     label=label)


        
        if teq == 0:
            axes[(0,col)].set_title(r'$t_{{\rm eq}} = 0$')
        else:
            axes[(0,col)].set_title(r'$t_{{\rm eq}} = 10^{{{0}}}$'.format(int(np.log10(teq))))
        axes[(0,col)].plot(t_plot, scaling_func(t_plot, 0.2, 0.5), ls='--', color='k',
                                label='$<\Delta x^2> \sim \sqrt{t}$')

        axes[(0,col)].legend(loc='upper left')
        axes[(1,col)].legend(loc='upper left')
        axes[(2,col)].legend(loc='upper left')
        axes[(2,col)].set_xlabel('$t$')
    axes[(0,0)].set_ylabel('$<\Delta x^2>$')
    fig.suptitle(r'$K_{{\rm on}} = 10^{{{0}}}, p = {1:.1f}, R = {2:.1f}, L = {3}, M = {4}$ runs' \
                 .format(int(np.log10(kon)), p, R, L, M))
    fig.savefig('../../plots/group.png', bbox_inches='tight')

