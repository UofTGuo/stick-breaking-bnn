from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr

import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.scipy.special import expit as sigmoid
from autograd.misc import flatten

from autograd.misc.optimizers import adam

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def relu(x):
    return np.maximum(0, x)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
#    for W, b in params[:-1]: # for weights and biases in all layers except last layer
#        outputs = np.dot(inputs, W) + b  # linear transformation
#        inputs = relu(outputs)                            # nonlinear transformation
#    outW, outb = params[-1]
#    outputs = np.dot(inputs, outW) + outb
#    return outputs
    for W, b in params:
        outputs = np.dot(inputs, W) + b  # [N,D]
        inputs = relu(outputs)
    return outputs  # [N, dim_last]

####

def encoder_h(params, inputs):
    # this is the encoder h that returns r_i = h(x_i, y_i)
    return neural_net_predict(params, inputs)

def aggregator_r(latent_input):
    # this is a simple aggregator function a that returns r = a((r_i)_{i=1}^n)
    return np.mean(latent_input, axis=0)

def get_z_params(input_r):
    # this takes in global representation r and returns parameters of latent variable distribution z
    # mu and sigma can be parameterized via NN
    #mu=input_r
    #sigma=np.exp(input_r)
    mu_z, log_sigma_z = input_r
    return ((mu_z, np.exp(log_sigma_z)))


def decoder_g(params, z_sample, x_star):
    # (u_ystar, log_std_ystar) = g(z_sample, x_star), g is NN
    # x_star is a column vector of xstars in test set
    # z_sample is a number
    # decoder_g returns a 2-column matrix of (mean, log_std) for ystar

    # I THINK THE BELOW LINE IS CAUSING AUTOGRAD THE PROBLEMS
    #inputs = np.append(x_star, np.zeros(x_star.shape[0]).reshape(x_star.shape[0], 1) + z_sample, axis=1)

    # THIS ONE WORKS. DON'T USE RESHAPE
    inputs = np.concatenate((x_star, np.zeros((x_star.shape[0], x_star.shape[1])) + z_sample ), axis=1)

    return neural_net_predict(params, inputs)

def logp_ystar_given_xstar_z(test_set, z, gen_params):
    # the first column of test_set contains x_star, the second column contains y_star
    # z is a number
    # gen_params is a list of weights
    u_ystar, log_std_ystar = unpack_gaussian_params(decoder_g(gen_params, z, test_set))
    return diag_gaussian_log_density(test_set, u_ystar, log_std_ystar)


def elbo(gen_params, rec_params, context, test, rs):
    # rs is random seed
    # context and test contain labels y

    # calculates parameters for q(z|context, test)
    q_means_both, q_log_stds_both = get_z_params(aggregator_r(encoder_h(rec_params, np.append(context, test, axis=0) )))# get posterior mean and std
    q_mean_context, q_log_stds_context = get_z_params(aggregator_r(encoder_h(rec_params, context )))

    # STEP 1: sample z ~ q(z|context, target)
    latent = sample_diag_gaussian(q_means_both, q_log_stds_both, rs)

    # STEP 2:
    loglikelihood = logp_ystar_given_xstar_z(test[:,1:], latent, gen_params).sum() # the 1: may need to change when scaling to higher dim

    q_z_given_context=diag_gaussian_log_density(latent, q_mean_context, q_log_stds_context)
    q_z_given_both=diag_gaussian_log_density(latent, q_means_both, q_log_stds_both)
    #q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
    #p_latents = diag_gaussian_log_density(latents, 0, 0)

    #likelihood = p_images_given_latents(gen_params, data, latents)

    return loglikelihood + np.mean(q_z_given_context - q_z_given_both)



#def generate_from_prior(gen_params, num_samples, noise_dim, rs):
#    latents = rs.randn(num_samples, noise_dim)
#    return sigmoid(neural_net_predict(gen_params, latents))


####

"""
def build_toy_dataset(n_data=80, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets
"""
def build_toy_dataset(n_data=80, noise_std=0.1, context_size=3):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-6,6,n_data)
    targets = np.sin(inputs)**3 + rs.randn(n_data) * noise_std
    return inputs[:, None], targets[:, None]

if __name__ == '__main__':

    # model param
    z_dim = 1 # this is the dimension of z
    data_dim = 1  # dimension of input data x
    latent_dim=1 # dimension of r
    gen_layer_sizes = [z_dim + data_dim, 200, 200, 2] # the architecture of generative network g. 2 corresponds to mu, sigma of y.
    rec_layer_sizes = [data_dim + 1, 200, 200, 2*latent_dim] # the architecture of encoding network h. + 1 corresponds to response y. 2 corresponds to mu and sigma for z

    # Training parameters
    param_scale = 0.01
    batch_size = 200
    num_epochs = 15
    step_size = 0.001

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes) # encoder initial params
    combined_init_params = (init_gen_params, init_rec_params)

    #init_scale = 0.1
    #weight_prior_variance = 10.0
    #init_params = init_random_params(init_scale, layer_sizes=[1, 4, 4, 1])

    inputs, targets = build_toy_dataset()
    inputs_with_y = np.hstack((inputs, targets))

    # SET CONTEXT POINTS
    context=inputs_with_y[0:50,:]
    test = inputs_with_y[50:, :]

    #print(inputs)
    #print(targets)

    seed = npr.RandomState(0)

    def objective(combined_params, iter):
        #data_idx = batch_indices(iter)
        gen_params, rec_params = combined_params
        return -elbo(gen_params, rec_params, context, test, seed)


    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params, t, g):
        rs=npr.RandomState(0)
        print("Iteration {} ELBO {}".format(t, -objective(params, t)))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx', ms=12)
        plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1)) # get gridpoints as vector

        gen_params, rec_params = params
        q_means_both, q_log_stds_both = get_z_params(aggregator_r(encoder_h(rec_params, inputs_with_y)))
        latent = sample_diag_gaussian(q_means_both, q_log_stds_both, rs)

        #outputs = np.exp(logp_ystar_given_xstar_z(plot_inputs, latent, gen_params)) # p(ystar|z, xstar)
        outputs, _ = unpack_gaussian_params(decoder_g(gen_params, latent, plot_inputs))

        #outputs = nn_predict(params, plot_inputs)
        #outputs

        ax.plot(plot_inputs, outputs, 'r', lw=3) # plot MAP estimate as a line in graph
        ax.set_ylim([-1, 1])
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0/60.0)

    print("Optimizing network parameters...")

    objective_grad = grad(objective)

    optimized_params = adam(objective_grad, combined_init_params,
                            step_size=0.005, num_iters=1000, callback=callback)
