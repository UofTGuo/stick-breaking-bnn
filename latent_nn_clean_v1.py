

######### HYPERPARAMETERS ##########

ystar_log_sigma=0.5
elbo_samples=1 # number of times to sample q(z|context,test) in ELBO calc for MC calculation
truncation_level = 7 # this is K
alpha=1 # alpha of Beta(alpha, beta)
beta=0.5 # beta of Beta(alpha, beta)
param_init_scale=1

#####################################

import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

rs = npr.RandomState(0)

import autograd.scipy.stats.norm as norm
from autograd.scipy.special import gammaln
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import expit as sigmoid

from autograd import grad
from autograd.misc.optimizers import adam

def rbf(x): return np.exp((-x**2))
def dim(x): return x[0].shape[0]
def relu(x):    return np.maximum(0, x)

def softplus(x): return np.log(1+np.exp(x))

def pack(data):
    #print(data[0].shape, data[1].shape)
    return np.concatenate(data, axis=1)

def aggregator(r): return np.mean(r, axis=0) # [dim_data , dim z]-> dim z

def Beta(a,b): return np.exp(gammaln(a)+gammaln(b)-gammaln(a+b) )


def diag_gaussian_log_density(x, mu, log_std):
    # x is a row
    # mu is a row
    # log_std is a row
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)


def unpack_gaussian_params(params):
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:D], params[D:]
    return mean, log_std


def init_net_params(layer_sizes, scale=0.1, rs=npr.RandomState(0)):
    return [(scale * rs.randn(m, n), scale * rs.randn(n))
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b  # [N,D]
        inputs = rbf(outputs)
        #inputs=softplus(outputs)
        #inputs = sigmoid(outputs)
        #inputs = relu(outputs)
    return outputs  # [N, dim_last]



def nn_predict_encoder(params, inputs):
    # takes input x_i and returns a_phi_i, b_phi_i
    return nn_predict(params, inputs)


def decoder_predict(params, inputs, latent):
    return nn_predict(params, pack((inputs, latent)))


def likelihood(params, data, latent):
    x, y = data
    pred = decoder_predict(params, x, latent)
    return diag_gaussian_log_density(y.T, pred[:,:1].T, ystar_log_sigma).sum()



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


def sample_functions(params, inputs, data, num_functions):
    # samples from approximate posterior
    # data contains x, y
    decoder_params, encoder_params = params

    data_moments = nn_predict_encoder(encoder_params, pack(data))
    log_a_encoder = np.mean(data_moments[:, :(truncation_level-1)], axis=0)
    log_b_encoder = np.mean(data_moments[:, (truncation_level-1):], axis=0)

    fs = [decoder_predict(decoder_params, inputs, np.tile(sample_latent_pi(log_a_encoder, log_b_encoder, 1), (inputs.shape[0],1)) )[:, :1] for
          _ in range(num_functions)]


    print("ONE SAMPLE MEAN OF STICKS IN PERCENT : ", np.mean(sample_latent_pi(log_a_encoder, log_b_encoder, inputs.shape[0]),axis=0)*100)

    return np.concatenate(fs, axis=1)



def sample_latent_sb(a,b, n_samples):
    # a, b have K-1 columns
    # this function samples from the Kumaraswamy distribution

    # sample from uniform distribution
    u=npr.uniform(0, 1, (n_samples, truncation_level-1)) # every row corresponds to a datapoint x_i
    a=np.exp(a)
    b=np.exp(b)

    return (1-u**(1/a))**(1/b)


def sample_latent_pi(aa,bb,n_samples):
    v_samples=sample_latent_sb(aa,bb,n_samples)
    v=v_samples
    vm=1-v_samples

    vs = [v[:, i][:, None] * np.prod(vm[:, :i], axis=1, keepdims=True) for i in range(1, v_samples.shape[1])]

    vl = np.prod(vm, axis=1, keepdims=True)

    w_vectors = [v_samples[:, 0][:, None]] + vs + [vl]
    weights = np.concatenate(w_vectors, axis=1)

    return weights


def lower_bound(decoder_params, encoder_params, data):
    # ELBO based on stickbreaking paper

    def stick_breaking_kl(a, b, prior_beta=1, prior_alpha=1):
        # a,b = np.exp(a), np.exp(b)

        kl = 0
        for i in range(1, 10):
            kl += 1. / (i + a * b) * Beta(i / a, b)
        kl *= (prior_beta - 1) * b

        # use another taylor approx for Digamma function
        psi_b_taylor_approx = np.log(b) - 1. / (2 * b) - 1. / (12 * b ** 2)

        kl += (a - prior_alpha) / a * (-0.57721 - psi_b_taylor_approx - 1 / b)
        kl += np.log(a * b) + np.log(Beta(prior_alpha, prior_beta))

        kl += -(b - 1) / b
        return np.sum(kl) # sum over every data point



    mc_elbo=0
    for c in range(elbo_samples):
        # latent is dim(data) x dimz

        data_moments = nn_predict_encoder(encoder_params, pack(data))

        log_a_encoder=np.mean(data_moments[:,:(truncation_level-1)], axis=0)
        log_b_encoder=np.mean(data_moments[:,(truncation_level-1):], axis=0)

        latent=np.tile(sample_latent_pi(log_a_encoder, log_b_encoder, 1), (data_moments.shape[0],1))


        likelihood_target = likelihood(decoder_params, data, latent)

        kl_term = stick_breaking_kl(a=np.exp(log_a_encoder), b=np.exp(log_b_encoder), prior_alpha=alpha, prior_beta=beta)
        mc_elbo = mc_elbo + (np.mean(likelihood_target) - kl_term)

        print("KL_term :", kl_term)
        print("likelihood : ", np.mean(likelihood_target))
    return mc_elbo/elbo_samples



if __name__ == '__main__':
    dimx, dimz, dimy = 1, truncation_level - 1, 1
    num_context = 1
    iters=200

    encoder_arch = [dimx+dimy, 8,8, 2*dimz]
    decoder_arch = [dimx + truncation_level, 10, 10, dimy]

    data = sample_data(20)
    context_data, target_data = get_context_and_target_data(data, num_context)

    print("Context data:", context_data)

    init_encoder_params = init_net_params(encoder_arch, scale=param_init_scale)
    init_decoder_params = init_net_params(decoder_arch, scale=param_init_scale)
    combined_params = (init_decoder_params, init_encoder_params)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)

    def objective(combined_params, iter):
        decoder_params, encoder_params = combined_params
        return -lower_bound(decoder_params, encoder_params, data)

    x, y = context_data
    xf, yf = data

    def callback(params, t, g):
        plot_inputs = np.linspace(-8, 8, num=400)[:, None]
        preds = sample_functions(params, plot_inputs, data, 200) # use full dataset
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


    objective_grad=grad(objective)

    var_params = adam(objective_grad, combined_params,
                      step_size=0.01, num_iters=iters, callback=callback)




