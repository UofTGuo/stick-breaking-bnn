

### HYPERPARAMS ###
ystar_log_sigma=20
elbo_samples=25 # number of times to sample q(z|context,test) in ELBO calc for MC calculation
###################

import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

rs = npr.RandomState(0)

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import expit as sigmoid

from autograd import grad
from autograd.misc.optimizers import adam

def rbf(x): return np.exp((-x**2)/2)
def dim(x): return x[0].shape[0]
def relu(x):    return np.maximum(0, x)
def pack(data):
    #print(data[0].shape, data[1].shape)
    return np.concatenate(data, axis=1)

def aggregator(r): return np.mean(r, axis=0) # [dim_data , dim z]-> dim z


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:D], params[D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

#def multi_sample_diag_gaussian(mean, log_std, n_samples):
#    return rs.randn(n_samples, mean.shape[0]) * np.exp(log_std) + mean

def repeat_sample_diag_gaussian(mean, log_std, n_data):
    #samples = sample_diag_gaussian(mean, log_std)
    samples = sample_diag_gaussian(mean, log_std)
    return np.tile(samples, (n_data, 1))

def init_net_params(layer_sizes, scale=0.1, rs=npr.RandomState(0)):
    return [(scale * rs.randn(m, n), scale * rs.randn(n))
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b  # [N,D]
        #inputs = rbf(outputs)
        inputs = sigmoid(outputs)
        #inputs = relu(outputs)
    return outputs  # [N, dim_last]

def nn_predict_gaussian(params, inputs):
    return unpack_gaussian_params(aggregator(nn_predict(params, inputs)))

def sample_latent(encoder_params, data, n_samples):
    # n_samples is the number of times to repeat the z sample, it's usually number of rows of data
    #
    mean, log_std = nn_predict_gaussian(encoder_params, pack(data))
    #print("mean",mean.shape)
    return repeat_sample_diag_gaussian(mean, log_std, n_samples)

def decoder_predict(params, inputs, latent):
    return nn_predict(params, pack((inputs, latent)))

def likelihood(params, data, latent):
    x, y = data
    pred = decoder_predict(params, x, latent)
    return diag_gaussian_log_density(y, pred, np.log(ystar_log_sigma))


def lower_bound(decoder_params, encoder_params, data, target_data, context_data):

    mc_elbo=0
    for c in range(elbo_samples):
        # latent is dim(data) x dimz
        latent = sample_latent(encoder_params, data, dim(data))
        #print(latent.shape)

        context_moments = nn_predict_gaussian(encoder_params, pack(context_data))
        data_moments = nn_predict_gaussian(encoder_params, pack(data))

        q_context = diag_gaussian_log_density(latent, *context_moments)
        q_data = diag_gaussian_log_density(latent, *data_moments)

        # likelihood_target gives vector of p(y*|z, x*) where (x*,y*) are points in target dataset
        #likelihood_target = likelihood(decoder_params, target_data, sample_latent(encoder_params, data, dim(target_data)))
        likelihood_target = likelihood(decoder_params, target_data, latent[0:dim(target_data),:])
        #mc_elbo = mc_elbo + np.mean(likelihood_target) + np.mean(q_context - q_data)
        mc_elbo = mc_elbo + np.mean(likelihood_target) + np.mean(q_context - q_data)

    return mc_elbo/elbo_samples


def sample_data(n_data=80, noise_std=0.1, context_size=3):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-6,6,n_data)
    targets = np.sin(inputs)**3 + rs.randn(n_data) * noise_std
    return inputs[:, None], targets[:, None]


def get_context_and_target_data(data, context_size=3):
    inputs, targets =data
    idx= np.arange(inputs.shape[0])
    np.random.shuffle(idx)
    c_idx, t_idx = idx[:context_size], idx[context_size:]
    context_data = inputs[c_idx], targets[c_idx]
    target_data = inputs[t_idx], targets[t_idx]
    return context_data, target_data

def sample_functions(params, inputs, cond_data, num_functions):
    decoder_params, encoder_params = params
    fs = [decoder_predict(decoder_params, inputs, sample_latent(encoder_params, cond_data, inputs.shape[0])) for _ in range(num_functions)]
    #fs = sample_diag_gaussian(np.array([decoder_predict(decoder_params, inputs, sample_latent(encoder_params, cond_data, inputs.shape[0])) for _ in
    #      range(num_functions)]), -1)
    return np.concatenate(fs, axis=1)


if __name__ == '__main__':
    dimx, dimz, dimy = 1, 2, 1
    num_context = 20
    iters=1000

    encoder_arch = [dimx+dimy, 80, 80, 2*dimz]
    decoder_arch = [dimx+dimz, 80, 80, dimy]

    data = sample_data()
    context_data, target_data = get_context_and_target_data(data, num_context)
    #print(dim(context_data), dim(context_data))

    print("Context data:", context_data)

    init_encoder_params = init_net_params(encoder_arch, scale=1.5)
    init_decoder_params = init_net_params(decoder_arch, scale=1.5)
    combined_params = (init_decoder_params, init_encoder_params)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)

    def objective(combined_params, iter):
        decoder_params, encoder_params = combined_params
        return -lower_bound(decoder_params, encoder_params, data, context_data, target_data)

    x, y = context_data
    xf, yf = data

    def callback(params, t, g):
        plot_inputs = np.linspace(-8, 8, num=400)[:, None]
        preds = sample_functions(params, plot_inputs, context_data, 200)
        #print(preds.shape)
        # Plot data and functions.

        plt.cla()
        ax.plot(x.ravel(), y.ravel(), 'o')
        ax.plot(xf.ravel(), yf.ravel(), '.')

        ax.plot(plot_inputs, preds, color='grey', linewidth=0.1)
        ax.set_title("fitting to toy data")
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("ITER {} | -ELBO {}".format(t, objective(params, t)))


    var_params = adam(grad(objective), combined_params,
                      step_size=0.01, num_iters=iters, callback=callback)




