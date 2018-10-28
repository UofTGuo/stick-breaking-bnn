import autograd.scipy.stats.norm as norm
import autograd.numpy as np
import autograd.numpy.random as npr
from sklearn.datasets import load_digits
rs = npr.RandomState(0)

from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.misc.optimizers import adam

def rbf(x): return np.exp(-x**2)
def dim(x): return x[0].shape[0]
def relu(x):    return np.maximum(0, x)
def log_softmax(a): return a-logsumexp(a, axis=1, keepdims=True)

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:,:D], params[:,D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std):
    #print(mean.shape, log_std.shape)
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def multi_sample_diag_gaussian(mean, log_std, n_samples):
    return rs.randn(n_samples, mean.shape[0]) * np.exp(log_std) + mean

def init_net_params(layer_sizes, scale=0.1, rs=npr.RandomState(0)):
    return [(scale * rs.randn(m, n), scale * rs.randn(n))
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b  # [N,D]
        inputs = relu(outputs)
    return outputs  # [N, dim_last]


def nn_predict_gaussian(params, inputs):
    return unpack_gaussian_params(nn_predict(params, inputs))

def sample_latent(params, inputs):
    return sample_diag_gaussian(*nn_predict_gaussian(params, inputs))

def decoder_predict(params, latent):
    return log_softmax(nn_predict(params, latent))

def sample_preds(params, inputs):
    decoder_params, encoder_params = params
    z = sample_latent(encoder_params, inputs)
    return decoder_predict(decoder_params, z)

def lower_bound(decoder_params, encoder_params, inputs, targets, beta=1e-3):

    encoder_moments = nn_predict_gaussian(encoder_params, inputs)  # [batch, K]
    latent = sample_diag_gaussian(*encoder_moments)  #[BS, K]

    log_decoder = decoder_predict(decoder_params, latent)  # [nd, nc]
    log_encoder_prior = diag_gaussian_log_density(latent, 0, 1)
    log_encoder = diag_gaussian_log_density(latent, *encoder_moments)
    #print(targets.shape, log_decoder.shape, log_encoder_prior.shape, log_encoder.shape)

    return -np.mean(log_decoder * targets) + beta * np.mean(log_encoder - log_encoder_prior)

def accuracy(params, inputs, targets):
    preds = sample_preds(params, inputs) #; print(preds.shape)
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(preds, axis=1)
    return np.mean(predicted_class == target_class)

if __name__ == '__main__':

    iters = 200
    K = 24
    encoder_arch = [64, 124, 124,  2*K]
    decoder_arch = [K, 10]

    init_encoder_params = init_net_params(encoder_arch)
    init_decoder_params = init_net_params(decoder_arch)
    combined_params = (init_decoder_params, init_encoder_params)

    # Training parameters
    batch_size = 256
    num_epochs = 50
    step_size = 0.001
    train_images, train_labels = load_digits(n_class=10, return_X_y=True)
    train_labels = np.eye(10)[train_labels]
    num_batches = int(np.ceil(len(train_images) / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    def objective(params, iter):
        decoder_params, encoder_params =params
        idx = batch_indices(iter)
        return lower_bound(decoder_params, encoder_params, train_images[idx], train_labels[idx])

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_images, train_labels)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, 'dummy'))
    optimized_params = adam(grad(objective), combined_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)