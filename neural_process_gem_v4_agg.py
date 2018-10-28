# this version uses the mean aggregator and ONE sample in the ELBO
# v4 contains context and test sets

### HYPERPARAMS ##########

#ystar_log_sigma=20
elbo_samples=1 # number of times to sample q(z|context,test) in ELBO calc for MC calculation
truncation_level = 10 # this is K
alpha=1 # alpha of Beta(alpha, beta)
beta=0.5 # beta of Beta(alpha, beta)
param_init_scale=0.5
#sum_trunc_level=10 # truncation level of infinite sum

##########################

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

""""
def sample_diag_gaussian(mean, log_std):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

#def multi_sample_diag_gaussian(mean, log_std, n_samples):
#    return rs.randn(n_samples, mean.shape[0]) * np.exp(log_std) + mean

def repeat_sample_diag_gaussian(mean, log_std, n_data):
    #samples = sample_diag_gaussian(mean, log_std)
    samples = sample_diag_gaussian(mean, log_std)
    return np.tile(samples, (n_data, 1))
"""

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


#def nn_predict_gaussian(params, inputs):
#    return unpack_gaussian_params(aggregator(nn_predict(params, inputs)))

def nn_predict_encoder(params, inputs):
    # takes input x_i and returns a_phi_i, b_phi_i
    return nn_predict(params, inputs)


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
    #return diag_gaussian_log_density(y, pred, np.log(ystar_log_sigma)) #  contains hyperparam ystar_log_sigma
    return diag_gaussian_log_density(y.T, pred[:,:1].T, pred[:,1:].T).sum()



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
    log_a_encoder = data_moments[:, :1].mean()
    log_b_encoder = data_moments[:, 1:].mean()
    #latent = sample_latent_pi(a_encoder, b_encoder, a_encoder.shape[0])

    #fs = [decoder_predict(decoder_params, inputs, sample_latent(encoder_params, cond_data, inputs.shape[0])) for _ in range(num_functions)]

    #fs = [decoder_predict(decoder_params, inputs, sample_latent_pi(a_encoder, b_encoder, a_encoder.shape[0]) )[:,:1] for _ in
    #      range(num_functions)]
    fs = [decoder_predict(decoder_params, inputs, sample_latent_pi(log_a_encoder, log_b_encoder, inputs.shape[0]))[:, :1] for
          _ in range(num_functions)]


    print("ONE SAMPLE OF STICKS IN PERCENT : ", np.mean(sample_latent_pi(log_a_encoder, log_b_encoder, inputs.shape[0]),axis=0)*100)

    #fs = sample_diag_gaussian(np.array([decoder_predict(decoder_params, inputs, sample_latent(encoder_params, cond_data, inputs.shape[0])) for _ in
    #      range(num_functions)]), -1)
    return np.concatenate(fs, axis=1)



def sample_latent_sb(a,b, n_samples):
    # this function samples from the Kumaraswamy distribution

    # sample from uniform distribution
    u=npr.uniform(0, 1, (n_samples, truncation_level)) # every row corresponds to a datapoint x_i
    a=np.exp(a)
    b=np.exp(b)
#    print("max(1/b) : ", max(1/b))
#    print("max(1/a) : ", max(1/a))

#    print("min(1/b) : ", min(1 / b))
#    print("min(1/a) : ", min(1 / a))
    uu=[(1-u[:,i:(i+1)]**(1/b))**(1/a) for i in range(truncation_level)]
    return np.concatenate(uu, axis=1)

"""
def sample_latent_pi(aa,bb):
    # aa, bb are floats
    pi=np.zeros(truncation_level) # initialize empty array as size K

    v=sample_latent_sb(aa,bb) # 1 dimensional np array of v's
    #pi=v
    #print("2v:", 2*v)
    #print("pi:", v[0]*v[1])
    #print(pi[0])
    pi[0] = v[0]

    for j in range(1, len(pi)): # index in 2 to len(pi)
        #pi[j] = sample_latent_sb(a=aa, b=bb) * np.prod(1-pi[0:(j)])
        pi[j] = v[j] * np.prod(1 - v[0:j])

    #pi[len(pi)-1]=1-np.sum(pi[0:(len(pi)-1)])
    #pi[len(pi) - 1] = sample_latent_sb(a=aa, b=bb) * np.prod(1-pi)
    return pi
"""

def sample_latent_pi(aa,bb,n_samples):
    v_samples=sample_latent_sb(aa,bb,n_samples)
    v=v_samples
    vm=1-v_samples

    vs=[v[:,i:(i+1)]*np.prod(vm[:,:i], axis=1, keepdims=True) for i in range(1, v_samples.shape[1])]

    vl = np.prod(vm, axis=1, keepdims=True)
    #    print(vl.shape)

    #w_vectors = [v_samples[0]] + vs # + [vl]
    #weights = np.concatenate(w_vectors, axis=1)
    #print(np.mean(weights))

    w_vectors=[v_samples[:,0:1]]+vs#+[vl]
    weights = np.concatenate(w_vectors, axis=1)
    #print(np.sum(weights, axis=1))


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
        #print("psi_b_taylor_approx :", max(abs(psi_b_taylor_approx)))
        #print("(a - prior_alpha) / a * (-0.57721 - psi_b_taylor_approx - 1 / b) :", max(abs((a - prior_alpha) / a * (-0.57721 - psi_b_taylor_approx - 1 / b))))
        #print("np.log(a * b) + np.log(Beta(prior_alpha, prior_beta)) :", max(abs(np.log(a * b) + np.log(Beta(prior_alpha, prior_beta)))))
        #print("-(b - 1) / b :", max(abs(-(b - 1) / b)))
        kl += (a - prior_alpha) / a * (-0.57721 - psi_b_taylor_approx - 1 / b)
        kl += np.log(a * b) + np.log(Beta(prior_alpha, prior_beta))

        kl += -(b - 1) / b
        return np.sum(kl) # sum over every data point



    mc_elbo=0
    for c in range(elbo_samples):
        # latent is dim(data) x dimz
        #latent = sample_latent(encoder_params, data, dim(data))

        #print(latent.shape)

        #context_moments = nn_predict_gaussian(encoder_params, pack(context_data)) # returns (array([a]), array([b])) for K(a,b) distribution
        data_moments = nn_predict_encoder(encoder_params, pack(data))
        log_a_encoder=data_moments[:,:1].mean()
        log_b_encoder=data_moments[:,1:].mean()
        #print("a:", a_encoder)
        #print("b:", b_encoder)

        #latent=np.zeros((dim(target_data), truncation_level)) # each row of latent is a latent sample corresponding to a data point
        #for i in range(dim(target_data)):
        #    s=sample_latent_pi(a_encoder[i,0], b_encoder[i,0])
        #    #print(s)
        #    latent[i, :]=s

        # a_encoder and b_encoder are exponentiated in sample_latent_pi

        #print(sample_latent_sb(a_encoder, b_encoder, a_encoder.shape[0]))
        latent=sample_latent_pi(log_a_encoder, log_b_encoder, data_moments.shape[0])


        #q_context = diag_gaussian_log_density(latent, *context_moments)
        #q_data = diag_gaussian_log_density(latent, *data_moments)

        # likelihood_target gives vector of p(y*|z, x*) where (x*,y*) are points in target dataset
        #likelihood_target = likelihood(decoder_params, target_data, sample_latent(encoder_params, data, dim(target_data)))
        #likelihood_target = likelihood(decoder_params, target_data, latent[0:dim(target_data),:])
        #likelihood_target = likelihood(decoder_params, target_data, latent)
        likelihood_target = likelihood(decoder_params, data, latent)
        #mc_elbo = mc_elbo + np.mean(likelihood_target) + np.mean(q_context - q_data)

        #eq_log_qvk=0
        #eq_log_pvk=0

        kl_term = stick_breaking_kl(a=np.exp(log_a_encoder), b=np.exp(log_b_encoder))
        mc_elbo = mc_elbo + (np.mean(likelihood_target) - kl_term)
        #print("a_min :", log_a_encoder.min())
        #print("a_max :", log_a_encoder.max())
        #print("b_min :", log_b_encoder.min())
        #print("b_max :", log_b_encoder.max())
        #print("KL_term :", kl_term)
        #print("likelihood : ", np.mean(likelihood_target))
    return mc_elbo/elbo_samples



if __name__ == '__main__':
    dimx, dimz, dimy = 1, 1, 1
    num_context = 1
    iters=1000

    encoder_arch = [dimx+dimy, 80,80, 2*dimz]
    #decoder_arch = [dimx+dimz, 8, 8, 2*dimy]
    decoder_arch = [dimx + truncation_level, 80,80, 2 * dimy]

    data = sample_data()
    context_data, target_data = get_context_and_target_data(data, num_context)
    #print(dim(context_data), dim(context_data))

    print("Context data:", context_data)

    init_encoder_params = init_net_params(encoder_arch, scale=param_init_scale)
    init_decoder_params = init_net_params(decoder_arch, scale=param_init_scale)
    combined_params = (init_decoder_params, init_encoder_params)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)

    #lower_bound(init_decoder_params, init_encoder_params, data, context_data)

    def objective(combined_params, iter):
        decoder_params, encoder_params = combined_params
        return -lower_bound(decoder_params, encoder_params, data)

    x, y = context_data
    xf, yf = data

    def callback(params, t, g):
        plot_inputs = np.linspace(-8, 8, num=400)[:, None]
        #preds = sample_functions(params, xf, data, 200) # use full dataset
        preds = sample_functions(params, plot_inputs, data, 200)  # use full dataset
        #print(np.mean(preds, axis=0))
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


    objective_grad=grad(objective)

    var_params = adam(objective_grad, combined_params,
                      step_size=0.01, num_iters=iters, callback=callback)




