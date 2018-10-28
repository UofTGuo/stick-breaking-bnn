import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import gammaln
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
rs = npr.RandomState(0)

def pad(x): return np.concatenate([x[:, None], np.ones((x.shape[0], 1))], axis=1)
def beta(a,b): return np.exp(gammaln(a)+gammaln(b)-gammaln(a+b) )

def shapes_and_num(layer_sizes):
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m + 1) * n for m, n in shapes)
    return shapes, N_weights

def unpack_layers(weights, layer_sizes):
    """ unpacks weights [ns, nw] into each layers relevant tensor shape"""
    shapes, _ = shapes_and_num(layer_sizes)
    n_samples = len(weights)
    for m, n in shapes:
        yield weights[:, :m * n].reshape((n_samples, m, n)), \
              weights[:, m * n:m * n + n].reshape((n_samples, 1, n))
        weights = weights[:, (m + 1) * n:]


def reshape_weights(weights, layer_sizes):
    return list(unpack_layers(weights, layer_sizes))


def bnn_predict(weights, inputs, layer_sizes, act):
    if len(inputs.shape)<3: inputs = np.expand_dims(inputs, 0)  # [1,N,D]
    weights = reshape_weights(weights, layer_sizes)
    for W, b in weights:
        outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
        inputs = act(outputs)
    return outputs


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)  # [ns]


def sample_bnn(weights, x, layer_sizes, act):
    return bnn_predict(weights, x, layer_sizes, act)[:, :, 0]  # [ns, nd]


def sample_kumaraswamy(a, b, n_samples=10):  # [ns, nz]
    u = npr.uniform(size=(n_samples, b.shape[0]))
    #a,b = np.exp(a), np.exp(b)
    return (1 - u ** (1 / b)) ** (1 / a)


def sample_stick_breaking_weights(params, n_samples):
    v_samples = sample_kumaraswamy(*params) # [ns, nw-1]
  #  print(v_samples)
    #exit()
    v = v_samples
    vm = 1-v_samples

    vs = [v[:, i][:,None]*np.prod(vm[:, :i], axis=1, keepdims=True) for i in range(1,v_samples.shape[1])]

 #   print(vs[0].shape)

    vl = np.prod(vm, axis=1, keepdims=True)
#    print(vl.shape)

    w_vectors=[v_samples[:,0][:,None]]+vs+[vl]
    weights = np.concatenate(w_vectors, axis=1)
    print(np.mean(weights))
    #print(weights)

    return weights


def stick_breaking_kl(a, b, prior_beta=1, prior_alpha=1):
    #a,b = np.exp(a), np.exp(b)

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
    """ estimates elbo =- E_q(pi)[log p(D|pi)] +KL{q(pi)||p(pi)}"""
    weights = sample_stick_breaking_weights(params, n_samples)

    f_bnn = sample_bnn(weights, x, layer_sizes, act)
    log_likelihood = diag_gaussian_log_density(y.T, f_bnn, .1)
    print(params[0].shape)
    kl_approx = stick_breaking_kl(*params)

    return - np.mean(log_likelihood)+kl_approx


def init_var_params(layer_sizes, scale=0.5, scale_mean=1):
    _, num_weights = shapes_and_num(layer_sizes)
    return np.ones(num_weights-1)*scale_mean, np.ones(num_weights-1)*scale

def build_toy_dataset(n_data=80, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets


def train_bnn(inputs, targets, arch = [1, 20, 20, 1], lr=0.01, iters=50, n_samples=1, act=np.tanh):

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)


    def objective(params,t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act)

    def callback(params, t, g):
        # Sample functions from posterior f ~ p(f|phi) or p(f|varphi)
        N_samples, nd = 5, 400
        plot_inputs = np.linspace(-8, 8, num=400)
        w = sample_stick_breaking_weights(params,5)
        f_bnn = sample_bnn(w, plot_inputs[:,None], arch, act)
        print(f_bnn)
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'k.')
        ax.plot(plot_inputs, f_bnn.T, color='r')
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("ITER {} | OBJ {}".format(t, -objective(params, t)))

    var_params = adam(grad(objective), init_var_params(arch),
                      step_size=lr, num_iters=iters, callback=callback)

    return var_params


if __name__ == '__main__':

    # Set up
    arch = [1, 20, 20, 1]
    inputs, targets = build_toy_dataset()
    train_bnn(inputs, targets)


