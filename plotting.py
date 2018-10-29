import matplotlib.pyplot as plt
import os
import numpy as np
import itertools
#from matplotlib.mlab import biivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from util import covariance
import matplotlib.mlab as mlab
from gmm import fit_one_gmm
import seaborn as sns
from scipy.stats import norm
sns.set_style("white")

n = 9
bnn_col = ["deep sky blue", "bright sky blue"]
gpp_bnn_col = ["red", "salmon"]
gp_col = ["green", "light green"]
colors = {"bnn": bnn_col, "gpp": gpp_bnn_col, "gp": gp_col}
sample_col = {"bnn": "bright sky blue", "gpp": "watermelon", "gp": "light lime"}
pal_col = {"bnn": sns.light_palette("#3498db", n_colors=n),  # nice blue
           "gpp": sns.light_palette("#e74c3c", n_colors=n),  # nice red
           "gp" : sns.light_palette("#2ecc71", n_colors=n)}  # nice green eh not so nice

def setup_plot():
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)
    return fig, ax

def setup_plots(show=True, num=2):
    fig_kw = {'figsize' : (20, 8), 'facecolor' : 'white'}
    f, ax = plt.subplots(num, sharex=True, **fig_kw)
    if show: plt.show(block=False)

    return f, ax

def plot_iter(ax, x, xp, y, p):
    plt.cla()
    ax.plot(x.ravel(), y.ravel(), color='g', marker='.')
    ax.plot(xp, p.T, color='r', marker='+')
    plt.draw()
    plt.pause(1.0 / 60.0)

def plot_fs(x, fs, xp, fgp, save_name):
    fig, ax = setup_plot()
    ax.plot(x, fs.T, color='r', label="hypernet")
    ax.plot(xp, fgp.T, color='g', label="gp")
    ax.set_title("final hypernet samples")
    plt.savefig("last-hypernet"+save_name+'.pdf', bbox_inches='tight')

def plot_mean_std(x_plot, p, D, title="", plot="bnn"):
    x, y = D
    col = colors[plot]
    col = sns.xkcd_rgb[col[1]]

    mean, std = np.mean(p, axis=1), np.std(p, axis=1)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'ko', ms=4)
    ax.plot(x_plot, mean, sns.xkcd_rgb[col[0]], lw=2)
    ax.fill_between(x_plot, mean - 1.96 * std, mean + 1.96 * std, color=col)  # 95% CI
    ax.tick_params(labelleft='off', labelbottom='off')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-8, 8])

    plt.savefig(title+plot+"-95 confidence.jpg", bbox_inches='tight')


def plot_deciles(x_plot, p, D, title="", plot="bnn"):
    x, y = D
    col = colors[plot]
    col = sns.xkcd_rgb[col[1]]

    mean, std = np.mean(p, axis=1), np.std(p, axis=1)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    zs = norm.ppf(1-0.5*0.1*np.linspace(1, 9, 9))  # critical vals for deciles
    ax.plot(x, y, 'ko', ms=4)
    ax.plot(x_plot, p, sns.xkcd_rgb[sample_col[plot]], lw=1)
    ax.plot(x_plot, mean, col, lw=1)
    pal = pal_col[plot]
    for z, col in zip(zs, pal):
        ax.fill_between(x_plot, mean - z * std, mean + z * std, color=col)
    ax.tick_params(labelleft='off', labelbottom='off')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-8, 8])

    plt.savefig(title+plot+"-deciles.pdf", bbox_inches='tight')


def plot_samples(x_plot, p, D, title="", plot="bnn"):
    x, y = D

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'ko', ms=4)
    ax.plot(x_plot, p, sns.xkcd_rgb[sample_col[plot]], lw=1)
    ax.tick_params(labelleft='off', labelbottom='off')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-8, 8])
    plt.savefig(title+plot+"-samples.pdf", bbox_inches='tight')


def plot_priors(x_plot, draws, title):

    f_gp_prior, f_bnn_prior, f_gp_bnn_prior = draws
    f, ax = plt.subplots(3, sharex=True)

    # plot samples
    ax[0].plot(x_plot, f_gp_prior, sns.xkcd_rgb["green"], lw=1)
    ax[1].plot(x_plot, f_gp_bnn_prior, sns.xkcd_rgb["red"], lw=1)
    ax[2].plot(x_plot, f_bnn_prior, sns.xkcd_rgb["blue"], lw=1)
    ax[1].set_title('learned bnn prior')
    ax[2].set_title('N(0,1) bnn prior')


    plt.tick_params(labelbottom='off')
    plt.savefig(title, bbox_inches='tight')

def plot_heatmap(moments, title):
    _, Sigma = moments
    sns.heatmap(Sigma, cmap="YlGnBu")
    plt.savefig(title, bbox_inches='tight')

def save_one_hist(ws, int, mu, sigma, save_name):
    plt.hist(ws[:,int], bins=30)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma))
    plt.savefig(save_name+"\w"+str(int)+".pdf", bbox_inches='tight')
    plt.clf()


def plot_weights(weights, save_name):  # [ns, nw]
    mus, var = fit_one_gmm(weights)
    print('ploting')
    fig_kw = {'figsize': (20, 20)}
    fig, axes = plt.subplots(10, 10, **fig_kw)
    for i, ax in enumerate(axes.reshape(-1)):
        int = np.random.randint(weights.shape[1])
        ws = weights[:, i]
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        mu = mus[:, int]
        sigma = np.sqrt(var[:, int])
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        ax.plot(x, mlab.normpdf(x, mu, sigma))
        #sns.distplot(ws, kde=False, norm_hist=True, rug=True, ax=ax)
        ax.hist(ws, bins=30)
        mu = mus[:, int]
        sigma = np.sqrt(var[:, int])
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        ax.plot(x, mlab.normpdf(x, mu, sigma))
#        save_one_hist(ws, int, mu, sigma, save_name)
    plt.savefig(save_name+"weightspace.pdf", bbox_inches='tight')
    plt.clf()
    for i in range(10*10):
        save_one_hist(weights, i, mu, sigma, save_name)


def plot_weights_function_space(weights, funcs, save_name, num=5):  # [ns, nw]
    for num in range(num):
        intw = np.random.randint(weights.shape[1]-1)
        intf = np.random.randint(funcs.shape[1]-1)
        ws1, ws2 = weights[:, intw], weights[:, intw+1]
        sns.jointplot(ws1, ws2, kind="scatter", space=0, color='b')
        plt.title('joint distribution of weights {} and {}'.format(intw,intw+1))
        plt.xlabel("weight {}".format(intw))
        plt.ylabel("weight {}".format(intw+1))
        plt.savefig("jointweightspaceof"+str(intw)+save_name+".pdf", bbox_inches='tight')

        plt.clf()
        f1, f2 = funcs[:, intf], funcs[:, intf + 1]
        sns.jointplot(f1, f2, kind='scatter', space=0, color='b')
        plt.title('joint distribution : $p( f(x_i), f(x_j)$ where i,j ={}{} '.format(intf,intf+1))
        plt.xlabel("f (x_i) ")
        plt.ylabel("f(x_j)")
        plt.savefig("fspace" + str(intf) + save_name + ".pdf", bbox_inches='tight')


def functions(xs, fgps, fnns, save_name):  # [ns, nw]
    fig, axes = plt.subplots(2, 2)
    for ax, f, fgp in zip(axes.reshape(-1), fnns, fgps):
        ax.set_title("f(x) for hypernet")

    plt.savefig("hyspace"+save_name+".pdf", bbox_inches='tight')
    plt.show()


def plot_weights_3d():
    # Create grid and multivariate normal
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    Z = bivariate_normal(X, Y, sigma_x, sigma_y, mu_x, mu_y)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
