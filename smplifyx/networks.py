import torch
import torch.nn as nn
import torch.nn.functional as F


def init_module(m, w_init, b_init):
    if hasattr(m, 'weight') and m.weight is not None and w_init is not None:
        w_init(m.weight)
    if hasattr(m, 'bias') and m.bias is not None and b_init is not None:
        b_init(m.bias)


class MLPBuilder:
    @staticmethod
    def from_cfg(config):
        return MLP(
            inp_dim=config['inp_dim'],
            outp_dim=config['outp_dim'],
            hidden_dims=config['hidden_dims'],
            hidden_activation=get_activation(config['hidden_activation']),
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            outp_layer=lambda inp, outp: LinearHead(inp, outp),
            batchnorm_first=config['bn_first'],
        )


class GaussianMLPBuilder:
    @staticmethod
    def from_cfg(config):
        return MLP(
            inp_dim=config['inp_dim'],
            outp_dim=config['outp_dim'],
            hidden_dims=config['hidden_dims'],
            hidden_activation=get_activation(config['hidden_activation']),
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            outp_layer=lambda inp, outp: GaussianLikelihoodHead(inp, outp),
            batchnorm_first=config['bn_first'],
        )


def get_activation(fname):
    activations = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'lrelu_0.2': (nn.LeakyReLU, 0.2),
        'lrelu_0.1': (nn.LeakyReLU, 0.1),
    }
    assert fname in activations.keys(), f'Unsupported hidden activation {fname}'

    return activations[fname]


class VAEBuilder:
    @staticmethod
    def from_cfg(config):
        enc = GaussianMLPBuilder.from_cfg(config['encoder'])
        dec = MLPBuilder.from_cfg(config['decoder'])
        return VariationalAutoEncoder(enc, dec)


class GaussianLikelihoodHead(nn.Module):
    def __init__(self, inp_dim, outp_dim):
        super().__init__()
        self.mu = nn.Linear(inp_dim, outp_dim)
        self.log_var = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        mean = self.mu(inp)
        log_var = self.log_var(inp)

        return mean, log_var


class LinearHead(nn.Module):
    def __init__(self, inp_dim, outp_dim):
        super().__init__()
        self.l = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        return self.l(inp)


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        hidden_dims,
        hidden_activation=nn.ReLU,
        outp_layer=nn.Linear,
        outp_activation=nn.Identity,
        weight_init=None,
        bias_init=None,
        weight_init_last=None,
        bias_init_last=None,
        batchnorm_first=False,
    ):
        super().__init__()
        self.w_init = weight_init
        self.b_init = bias_init
        self.w_init_last = weight_init_last
        self.b_init_last = bias_init_last
        if isinstance(hidden_activation, tuple):
            self.hidden_activation = hidden_activation[0]
            self.hidden_param = hidden_activation[1]
        else:
            self.hidden_activation = hidden_activation
            self.hidden_param = None
        self.outp_dim = outp_dim

        if batchnorm_first:
            self.input_bn = nn.BatchNorm1d(inp_dim, momentum=0.1, affine=False)
        else:
            self.input_bn = None

        layers = []
        current_dim = inp_dim
        for _, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.hidden_param is None:
                layers.append(self.hidden_activation())
            else:
                layers.append(self.hidden_activation(self.hidden_param))
            current_dim = hidden_dim

        layers.append(outp_layer(current_dim, outp_dim))
        if outp_activation is not None:
            layers.append(outp_activation())

        self.layers = nn.Sequential(*layers)
        self.init()

    def init(self):
        self.layers.apply(lambda m: init_module(m, self.w_init, self.b_init))
        self.layers[-2].apply(lambda m: init_module(m, self.w_init_last, self.b_init_last))

    def forward(self, inp):
        if self.input_bn is not None:
            inp = self.input_bn(inp)

        return self.layers(inp)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, enc, dec):
        super(VariationalAutoEncoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.latent_dim = enc.outp_dim

    def encode(self, inp):
        return self.enc(inp)

    def decode(self, z):
        return self.dec(z)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, inp):
        mu, log_var = self.encode(inp)
        z = self.reparametrize(mu, log_var)
        reconstructed = self.decode(z)

        return reconstructed, mu, log_var

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]


class AutoEncoder(nn.Module):
    def __init__(self, enc, dec):
        super(AutoEncoder, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, inp):
        z = self.enc(inp)
        # Do something fun here

        reconstructed = self.dec(z)

        return reconstructed
