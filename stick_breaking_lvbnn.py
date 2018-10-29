import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.scipy.special import gammaln

from bayesian_neural_net import bnn_predict, shapes_and_num


rs = npr.RandomState(0)


def rbf(x): return np.exp(-x**2)
def dim(x): return x[0].shape[0]
def relu(x):    return np.maximum(0, x)
def pack(data):
    #print(data[0].shape, data[1].shape)
    return np.concatenate(data, axis=1)
def beta(a,b): return np.exp(gammaln(a)+gammaln(b)-gammaln(a+b) )


def sample_kumaraswamy(a, b):  # [BS, nz-1]
    u = npr.uniform(size=b.shape)
    #print(a,b)
    return (1 - u ** (1 / b)) ** (1 / a)


def sample_sticks(a,b, n_samples=1):
    v_samples = sample_kumaraswamy(a,b)  # [nz-1]
    #print(v_samples)
    v = v_samples[None, :]
    vm = 1-v
    vs = [v[:, i][:, None]*np.prod(vm[:, :i], axis=1, keepdims=True) for i in range(1,v.shape[1])]
    vl = np.prod(vm, axis=1, keepdims=True)
    pi_vectors = [v[:,0][:,None]]+vs+[vl]
    pis = np.concatenate(pi_vectors, axis=1)
    #print(np.round(pis[0],3))
    #print(weights)

    return pis

def sample_sblatents(loga, logb, n_data):
    samples = sample_sticks(loga, logb)
    return np.tile(samples, (n_data, 1))  # [n_data, n_latent]

def sample_weights(mean, log_std, N_samples=5):
    return rs.randn(N_samples, mean.shape[0]) * np.exp(log_std) + mean  # [ns, nw]

def sample_bnn(params, x, N_samples, layer_sizes, act, noise=0.0):
    qw_mean, qw_log_std, qz_loga, qz_logb = params
    qz_loga, qz_logb = np.exp(-qz_loga), np.exp(-qz_logb)
    weights = sample_weights(qw_mean, qw_log_std, N_samples)
    latents = sample_sblatents(qz_loga, qz_logb, x.shape[0])  # [nd, dimz]
    inputs = np.concatenate([x, latents], -1)
    return bnn_predict(weights, inputs, layer_sizes, act)[:, :, 0]   # [ns, nd]


def gaussian_entropy(log_std):
    return 0.5 * log_std.shape[0] * (1.0 + np.log(2*np.pi)) + np.sum(log_std)


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)  # [ns]


def stick_breaking_kl(a, b, prior_beta=1, prior_alpha=1):

    kl = 0
    for i in range(1,10):
        kl += 1./(i+a*b) * beta(i/a, b)
    kl *= (prior_beta-1)*b

    # use another taylor approx for Digamma function
    psi_b_taylor_approx = np.log(b) - 1./(2 * b) - 1./(12 * b**2)
    kl += (a-prior_alpha)/a * (-0.57721 - psi_b_taylor_approx - 1/b)
    kl += np.log(a*b) + np.log(beta(prior_alpha, prior_beta))
    kl += -(b-1)/b

    return np.sum(kl)



def vlb_objective(params, x, y, layer_sizes, n_samples, model_sd=0.1, act=np.tanh):
    """ estimates lower bound = -H[q(w))] - E_q(w)[log p(D,w)] """
    qw_mean, qw_log_std, qz_loga, qz_logb = params
    qz_loga, qz_logb = np.exp(-qz_loga), np.exp(-qz_logb)
    weights = sample_weights(qw_mean, qw_log_std, n_samples)
    latents = sample_sblatents(qz_loga, qz_logb, x.shape[0])  # []
    entropy = gaussian_entropy(qw_log_std)
    f_bnn= bnn_predict(weights, np.concatenate([x, latents], 1), layer_sizes, act)[:, :, 0]   # [ns, nd]


    #f_bnn = sample_bnn(params, x, n_samples,layer_sizes, act)
    log_likelihood = np.mean(diag_gaussian_log_density(y.T, f_bnn, 1))
    qw_log_prior = np.mean(diag_gaussian_log_density(weights, 0, .1))
    sb_kl = stick_breaking_kl(qz_loga, qz_logb)

    print(sb_kl, log_likelihood, entropy)

    return -entropy - log_likelihood+qw_log_prior + sb_kl


def init_var_params(layer_sizes, dimz, scale=0.01, scale_mean=1):
    _, num_weights = shapes_and_num(layer_sizes)
    qw_init = [rs.randn(num_weights)*scale_mean, np.ones(num_weights)*scale]  # mean, log_std
    qz_init = [np.ones(dimz-1)*scale_mean, np.ones(dimz-1)*scale]  # mean, log_std
    return qw_init + qz_init


def sample_data(n_data=80, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-6,6,n_data)
    targets = np.sin(inputs)**3 + rs.randn(n_data) * noise_std
    return inputs[:, None], targets[:, None]


def train_SBLVbnn(inputs, targets, dimz=1, dimx=1, dimy=1,
                arch = [20, 20], lr=0.01, iters=500, n_samples=10, act=rbf):

    arch = [dimx+dimz] + arch + [dimy]
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)

    def objective(params, t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act)

    def callback(params, t, g):
        N_samples, nd = 5, 80
        plot_inputs = np.linspace(-8, 8, num=80)
        f_bnn = sample_bnn(params, plot_inputs[:, None], N_samples, arch, act)
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'k.')
        ax.plot(plot_inputs, f_bnn.T, color='r')
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("ITER {} | OBJ {}".format(t, -objective(params, t)))

    var_params = adam(grad(objective), init_var_params(arch, dimz),
                      step_size=lr, num_iters=iters, callback=callback)

    return var_params


if __name__ == '__main__':
    inputs, targets = sample_data()
    train_SBLVbnn(inputs, targets, dimz=2)






