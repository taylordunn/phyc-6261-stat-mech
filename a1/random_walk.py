import numpy as np
import random as ran
import matplotlib.pylab as plt
import seaborn as sns # Makes plots look nicer
import time

def step():
    # Discrete, unbiased step in 1 dimension
    return ran.choice([-1, 1])

def walk(t_sample):
    # Given sample time points, performs a discrete 1d random walk and returns
    #  net displacement x at those times/steps
    t_max = np.max(t_sample)
    
    x = 0
    x_sample = []
    for t in range(1, t_max+1):
        x += step()
        
        if t in t_sample:
            x_sample.append(x)

    return x_sample

def sample(N, t_sample):
    # Performs N random walks, sampling at the steps given by t_sample

    # This array will hold all measurements, with time separated by columns and
    #  each row corresponding to a single random walk
    samples = np.empty([N, len(t_sample)])

    for n in range(N):
        x_sample = np.array(walk(t_sample))

        samples[n,:] = x_sample

    return samples

def plot_dist():
    sns.set(style='whitegrid') # Comment out if not using seaborn
    N = 2000
    t_sample = [1, 10, 100, 1000]

    # Probability distributions on top row, residuals on bottom row,
    #  and a column per time/step sample point
    fig, axes = plt.subplots(ncols=len(t_sample), nrows=2, sharex='col',
                             sharey='row', figsize=(11,8))

    samples = sample(N, t_sample)

    # The diffusion coefficient with step size and time step 1
    D = 1 / 2
    # The gaussian describing the diffusion equation density
    p_diffusion = lambda x,t: np.exp(-x**2/(4*D*t)) / np.sqrt(4*np.pi*D*t)

    for i,t in enumerate(t_sample):
        x = samples[:,i]
        bins = np.arange(np.min(x)-2, np.max(x)+4, 2)

        p_meas,_ = np.histogram(x, bins=bins, normed=True)

        error_bars = np.sqrt(p_meas / N)
        axes[0,i].bar(bins[:-1], p_meas, alpha=0.7, align='center',
                      yerr=error_bars, label='$P(x,t)$',
                      error_kw={'ecolor': 'r', 'alpha': 0.7})

        p_theory = p_diffusion(bins, t)
        axes[0,i].plot(bins, p_theory, label=r'$\rho (x,t)$', alpha=0.8,
                       color='k')

        # p_theory has one extra data point due to binning
        resid = p_meas - p_theory[:-1]
        axes[1,i].bar(bins[:-1], resid, alpha=0.7, align='center',
                      label=r'$P - \rho$')
        axes[1,i].fill_between(bins[:-1], y1=error_bars, y2=-error_bars,
                               color='r', alpha=0.2,
                               label='Poisson error bar')

        # An alternative way to get error bars is to measure the variance
        #  in >= ~20 smaller histograms
        hist = np.empty([20, len(bins)-1])
        # Find the indicies at which to cut the data into 20 histograms
        indices = np.linspace(0, N, 21).astype(int)
        for j in range(len(indices)-1):
            # Subset the data to get a new histogram
            sub_x = x[indices[j]:indices[j+1]]
            p_meas,_ = np.histogram(sub_x, bins=bins, normed=True)
            hist[j,:] = p_meas
        
        # Now get the standard deviation per column/bin to find the error bars
        #  in the histogram. The axis=0 argument means perform the np.std()
        #  calculation per column
        error_bars = np.std(hist, axis=0) / np.sqrt(20)
        axes[1,i].fill_between(bins[:-1], y1=error_bars, y2=-error_bars,
                               color='g', alpha=0.2,
                               label='Error from 20 histograms')

        axes[0,i].set_title(r'$t = %d$' % t)
        axes[1,i].set_xlim(1.1*bins[0], 1.1*bins[-1])
        axes[1,i].set_xlabel('$x$')

    axes[0,0].set_yscale('log')
    axes[0,0].set_ylim([0.001,1])
    axes[0,0].set_yticks([0.001, 0.01, 0.1, 1])
    axes[0,0].set_yticklabels(['0.1%', '1%', '10%', '100%'])
    axes[1,0].set_yticks([-0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01,
                          0.015, 0.02])
    axes[1,0].set_yticklabels([-0.02, '', -0.01, '', 0, '', 0.01, '', 0.02])
    axes[0,-1].legend()
    axes[1,-1].legend()
    axes[0,0].set_ylabel('Probability distribution')
    axes[1,0].set_ylabel('Residual')
    plt.tight_layout()

    fig.savefig('random_walk.pdf', bbox_inches='tight')


start_time = time.time()
plot_dist()
print('Run time = %.1f seconds' % (time.time() - start_time))
