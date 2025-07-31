import sys
import jax
import jax.numpy as jnp
import jax.random as jrnd
from typing import Tuple, List
from dataclasses import dataclass, field

import enum

from flax import linen as nn
from flax.linen import relu, elu, linear


import numpy as np
import numpy.random as random

class ActivationFunction(enum.Enum):
    RELU = enum.auto()
    ELU = enum.auto()
    SIN = enum.auto()
    SOFTPLUS = enum.auto()
    
def get_activation_function(activation_function: ActivationFunction):
    return {
        ActivationFunction.RELU: relu,
        ActivationFunction.ELU: elu,
        ActivationFunction.SIN: jnp.sin,
        ActivationFunction.SOFTPLUS: safe_softplus,
    }[activation_function]

def softplus(x, beta=100):
    return jnp.logaddexp(0, beta * x) / beta

def safe_softplus(x, beta=100):
    # revert to linear function for large inputs, same as pytorch
    return jnp.where(x * beta > 20, x, softplus(x))

def non_zero_mean(key, shape, dtype=jnp.float32):
    normal_random_values = jrnd.normal(key, shape, dtype=dtype)
    mu = jnp.sqrt(jnp.pi) / jnp.sqrt(shape[0])
    return  mu + 0.00001 * normal_random_values

def zero_mean_multires(key, shape, dtype=jnp.float32):
    if shape[0] <= 3:
        normal_random_values = jrnd.normal(key, shape, dtype=dtype)
        sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
        init_weight = sigma * normal_random_values
    if shape[0] > 3:
        normal_random_values = jrnd.normal(key, (3, shape[1]), dtype=dtype)
        sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
        constant = jnp.zeros((shape[0]-3,shape[1]), dtype=dtype)
        weights = jnp.concatenate((normal_random_values, constant), axis=0)
        init_weight = sigma * weights
    return init_weight

def zero_mean(key, shape, dtype=jnp.float32):
    normal_random_values = jrnd.normal(key, shape, dtype=dtype)
    sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
    init_weight = sigma * normal_random_values
    return init_weight


class MLP(nn.Module):
    d_in: int=3
    d_out: int=1
    dims: List[int]=field(default_factory=list)
    skip_layers: Tuple[int,...]=(4,)
    # activation: ActivationFunction = ActivationFunction.SOFTPLUS
    activation: List[str] = field(default_factory=list)
    geometry_init: bool = False
    init_radius: float=1.0
    multires: int=0
    feature_vector_size: int=0
    timespace: bool=False
    num_shape: int=0
    
    @nn.compact
    def __call__(self, x, t=None, condition=None):
        
        
        activation = list(map({
            'softplus': safe_softplus,
            'relu': relu,
            'sin':jnp.sin,
            'elu': elu
        }.get, self.activation))[0]
        
        if t is not None:
            x = jnp.concatenate([x, t], axis=-1)

        # positional encoding
        if self.multires > 0:
            x = posenc(x, min_deg=0, max_deg=self.multires, invertable=True)

        if condition is not None:
            # condition = jnp.tile(condition[None, :], (x.shape[0], 1))
            x = jnp.concatenate([x, condition], axis=-1)

        input_x = x
        dims = [x.shape[-1]] + [*self.dims] + [self.d_out]

        for i in range(len(dims)-2):
            
            out_dim = dims[i+1]
            
            if self.geometry_init:
                kernel_init = zero_mean
                if i == 0:
                    kernel_init = zero_mean_multires
            else:
                kernel_init = linear.default_kernel_init
                
            if i+1 in self.skip_layers:
                out_dim = dims[i+1] - dims[0]
                x = jnp.concatenate([x ,input_x], axis=-1) / jnp.sqrt(2)
            
            x = nn.Dense(features=out_dim, name=f'dense_{i}', kernel_init=kernel_init)(x)
            x = activation(x)
        
        # last layer
        kernel_init_final = non_zero_mean if self.geometry_init else linear.default_kernel_init
        bias_init_final = jax.nn.initializers.constant(-self.init_radius if self.geometry_init else 0.)
        x = nn.Dense(features=self.d_out, name=f'dense_{len(dims)-1}', kernel_init=kernel_init_final, bias_init=bias_init_final)(x)
    
        return x.squeeze()


###########################################################



class Decoder(nn.Module):
    num_shape: int=80
    feature_vector_size: int=64
    d_in: int=3
    d_out: int=1
    dims: List[int]=field(default_factory=list)
    skip_layers: Tuple[int,...]=(4,)
    activation: List[str] = field(default_factory=list)
    geometry_init: bool = False
    init_radius: float=0.5
    multires: int=0
    timespace: bool=False
    
    @nn.compact
    def __call__(self, x, t=None, index=[0,1], latent_vec=None):

        embedder = nn.Embed(num_embeddings=self.num_shape, features=self.feature_vector_size//2)
        
        # this is for training
        if index is not None:
            feature_x = embedder(jnp.array(index[0], jnp.int32)).squeeze()
            feature_y = embedder(jnp.array(index[1], jnp.int32)).squeeze()
        # this is for inference
        elif latent_vec is not None:
            feature_x, feature_y = latent_vec
        else:
            raise ValueError("Either index or latent must be provided.")
        
        mlp = MLP(d_in=self.d_in, d_out=self.d_out, dims=self.dims, skip_layers=self.skip_layers, activation=self.activation, geometry_init=self.geometry_init, init_radius=self.init_radius, multires=self.multires, timespace=self.timespace)

        # input = jnp.concatenate((embeddings, x), axis=-1)
        embeddings = jnp.concatenate((feature_x, feature_y),axis=-1)
        
        
        if (len(x.shape)>1) and (embeddings.shape[0] != x.shape[0]):
          embeddings = jnp.tile(embeddings, (x.shape[0], 1))

        
        x = mlp(x=x, t=t, condition=embeddings)
        
        return x

    def get_feature(self, params, index):
        return params['params']['Embed_0']['embedding'][index]
        



###########################################################



class SharedEmbedMLP(nn.Module):
    num_shape: int=80
    feature_vector_size: int=64
    di_in: int=3
    di_out: int=1
    dv_in: int=3
    dv_out: int=3
    dims: List[int]=field(default_factory=list)
    skip_layers: Tuple[int,...]=(4,)
    activation: List[str] = field(default_factory=list)
    geometry_init: bool = False
    init_radius: float=0.5
    multires: int=0
    timespace: bool=False
    
    @nn.compact
    def __call__(self, x, t=None, index=[0,1], latent_vec=None):

        embedder = nn.Embed(num_embeddings=self.num_shape, features=self.feature_vector_size//2)
        
        # this is for training
        if index is not None:
            feature_x = embedder(jnp.array(index[0], jnp.int32)).squeeze()
            feature_y = embedder(jnp.array(index[1], jnp.int32)).squeeze()
        # this is for inference
        elif latent_vec is not None:
            feature_x, feature_y = latent_vec
        else:
            raise ValueError("Either index or latent must be provided.")
        
        mlp_i = MLP(d_in=self.di_in, d_out=self.di_out, dims=self.dims, skip_layers=self.skip_layers, activation=self.activation, geometry_init=True, init_radius=self.init_radius, multires=self.multires, timespace=self.timespace)
        
        mlp_v = MLP(d_in=self.dv_in, d_out=self.dv_out, dims=self.dims, skip_layers=self.skip_layers, activation=self.activation, geometry_init=False, init_radius=self.init_radius, multires=self.multires, timespace=self.timespace)

        # input = jnp.concatenate((embeddings, x), axis=-1)
        embeddings = jnp.concatenate((feature_x, feature_y),axis=-1)
        
        
        if (len(x.shape)>1) and (embeddings.shape[0] != x.shape[0]):
          embeddings = jnp.tile(embeddings, (x.shape[0], 1))

        
        sdf = mlp_i(x=x, t=t, condition=embeddings)
        v = mlp_v(x=x, t=t, condition=embeddings)
        
        return sdf, v

    def get_feature(self, params, index):
        return params['params']['Embed_0']['embedding'][index]
        





###########################################################


def posenc(x, min_deg, max_deg, legacy_posenc_order=False, invertable=False):
    if min_deg == max_deg:
        return x
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    if legacy_posenc_order:
        xb = x[Ellipsis, None, :] * scales[:,None]
        four_feat = jnp.reshape(jnp.sin(jnp.stack([xb, xb+0.5*jnp.pi], -2)), list(x.reshape[-1])+[-1])
    else:
        xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
        four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    encoded = jnp.concatenate([x] + [four_feat], axis=-1)

    if invertable:
        coeff = 1 / jnp.sqrt(2*max_deg+1)
        y = jnp.ones_like(x)
        yb = jnp.reshape((y[Ellipsis, None, :] * scales[:, None]),
                     list(y.shape[:-1]) + [-1])
        yb_norm = jnp.concatenate([yb, yb],axis=-1)
        yb_norm = jnp.concatenate([y] + [yb_norm], axis=-1)
        encoded = coeff * encoded / yb_norm 
    
    return encoded
