
import jax 
import jax.numpy as jnp
from jax import vmap
from functools import partial
from datasets.pointshape import DeformPointCloud
from flax.training.train_state import TrainState
import optax
import numpy as np
import jax.random as jrnd
from jax import lax
from flax import traverse_util

import flax


def check_value(x):
    return jax.lax.cond(
        x > 0,
        lambda: jnp.array(True),  # Ensure output remains a JAX array
        lambda: jnp.array(False)
    )
    
    
def MLP_init(key, learning_rate_fn, MLP, conf):
    d_in = conf.d_in
    if conf.timespace:
        if conf.feature_vector_size > 0:
            params = MLP.init(key, jnp.ones(d_in), jnp.ones(1), jnp.ones(conf.feature_vector_size))
        else:
            params = MLP.init(key, jnp.ones(d_in), jnp.ones(1))
    else:
        params = MLP.init(key, jnp.ones(d_in))
        
    train_state = TrainState.create(
        apply_fn=MLP.apply,
        tx=optax.adam(learning_rate=learning_rate_fn),
        params=params
    )

    return train_state




def Decoder_init(key, learning_rate_fn, decoder_net, conf):
    d_in = conf.d_in
    if conf.feature_vector_size == 0:
        return MLP_init(key, learning_rate_fn=learning_rate_fn, MLP=decoder_net, conf=conf)
    else:
        if conf.timespace:
            params = decoder_net.init(key, jnp.ones(d_in), jnp.ones(1), [0,1])
        else:
            params = decoder_net.init(key, jnp.ones(d_in), [0,1])
        
    
        train_state = TrainState.create(
            apply_fn=decoder_net.apply,
            tx=optax.adam(learning_rate=learning_rate_fn),
            params=params
        )

    return train_state


def grid_slice(x, step=64**3):
    num_points, dim = x.shape
    x_arr = []
    for N in range(0, num_points, step):
        if N + step < num_points:
            x_arr.append(lax.slice(x, (N,0), (N + step,dim)))
        else:
            x_arr.append(lax.slice(x, (N,0), (num_points,dim)))
    return x_arr



def implicit_distance(f, points, t, distance_metric='l2'):
    sdf = f(points, t)
    if distance_metric == 'squared_l2':
        implicit_distance = (sdf ** 2).mean()
    elif distance_metric == 'l2':
        implicit_distance = jnp.abs(sdf).mean()
    else:
        raise ValueError(f'Unrecognized distance metric {distance_metric=}.')
    return implicit_distance

def sdf_loss(sdf, distance_metric='l2'):
    if distance_metric == 'squared_l2':
        implicit_distance = (sdf ** 2).mean()
    elif distance_metric == 'l2':
        implicit_distance = jnp.abs(sdf).mean()
    else:
        raise ValueError(f'Unrecognized distance metric {distance_metric=}.')
    return implicit_distance


def soft_sign(x, eps=1e-12):
    n = jnp.sqrt(x**2+eps)
    return x / n

def sq_norm(a, *args, **kwargs):
    return (a ** 2).sum(*args, **kwargs)

def soft_norm(x, *arg, **kwargs):
    eps=1e-12
    """Use l2 for large values and squared l2 for small values to avoid grad=nan at x=0."""
    return eps * (jnp.sqrt(sq_norm(x, *arg, **kwargs) / eps ** 2 + 1) - 1)

def safe_normalize(x, eps=1e-12):
    x_norm = soft_norm(x)
    return x / jnp.array([x_norm, eps]).max()

def match_loss(pointx, pointy):
    loss = soft_norm(pointx-pointy, axis=1)**2
    # loss = (pointx-pointy)**2
    return loss.mean()


def normal_loss(gt, df):

    # normalize
    df = safe_normalize(df)
    gt = safe_normalize(gt)
    
    loss = sq_norm(gt-df, axis=1).mean()

    return loss


def projection_matrix(df):
    df = safe_normalize(df)
    num_points, dim = df.shape
    identity = jnp.tile(jnp.eye(3), (num_points, 1, 1))
    # dft = jnp.transpose(df, (0,2,1))
    df = df[:,:,None]
    dfdft = jnp.einsum('bij,bjk->bjk', df, df.transpose(0,2,1))
    return identity - dfdft


def stretching_loss(df, dv):
    p = projection_matrix(df)
    num_points, dim1, dim2 = dv.shape
    identity = jnp.tile(jnp.eye(dim1), (num_points, 1,1))
    dv = identity + dv
    pt = jnp.transpose(p, (0,2,1))
    jtj = jnp.einsum('bij,bjk->bik', jnp.transpose(dv, (0,2,1)), dv)
    I_jtj =  jtj - identity
    loss_1 = jnp.einsum('bij,bjk->bik', pt, I_jtj)
    loss_2 = jnp.einsum('bij,bjk->bik', loss_1, p)
    loss = jnp.linalg.norm(loss_2, ord='fro', axis=(-2, -1))
    return loss.mean()


def R_term(df, dv):
    n = safe_normalize(df)
    n = n[:,:,None]
    R = jnp.sum(jnp.matmul(dv,n)* n, axis=1).squeeze() 
    return R


def deformation_rate(dv):
    D = (dv + jnp.transpose(dv, (0, 2, 1))) * 0.5
    part_1 = jnp.trace(D, axis1=1, axis2=2)
    DD = jnp.einsum('bij,bjk->bik', D, D)
    part_2 = jnp.trace(DD, axis1=1, axis2=2)
    
    # Second invariant calculation using the standard formula
    second_invariant = 0.5 * part_2 - (1/6) * part_1**2
    return second_invariant


def deformation_gradient(df1, df2, dv):
    num_points, dim1, dim2 = dv.shape
    identity = jnp.tile(jnp.eye(dim1), (num_points, 1,1))
    F = jnp.transpose(identity + dv, (0,2,1))
    # F = identity + dv
    Fn = jnp.einsum('bij, bjk->bik', F, df2[:,:,None])
    Fn = safe_normalize(Fn.squeeze())
    n = safe_normalize(df1)

    return normal_loss(Fn, n)


def any_nans(pytree):
    """Returns True if any leaf of the pytree contains a nan value."""
    return jnp.array(jax.tree_util.tree_flatten(jax.tree_map(lambda a: jnp.isnan(a).any(), pytree))[0]).any()


def safe_apply_grads(state, grads):
    nan_grads = any_nans(grads)
    state = jax.lax.cond(nan_grads, lambda: state, lambda: state.apply_gradients(grads=grads))
    return state, nan_grads


def mlp_init(key, learning_rate_fn, MLP, conf):
    d_in = conf.d_in
    if conf.timespace:
        if conf.feature_vector_size > 0:
            params = MLP.init(key, jnp.ones(d_in), jnp.ones(1), jnp.ones(conf.feature_vector_size))
        else:
            params = MLP.init(key, jnp.ones(d_in), jnp.ones(1))
    else:
        params = MLP.init(key, jnp.ones(d_in))
        
    trian_state = TrainState.create(
        apply_fn=MLP.apply,
        tx=optax.adam(learning_rate=learning_rate_fn),
        params=params
    )

    return trian_state


