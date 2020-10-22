# libraries
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import os

# Select GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# Change seeds for different runs
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import math
from itertools import product
import tensorflow_probability as tfp
from scipy.io import loadmat
import pandas as pd

slim=tf.contrib.slim


# Settings

learning_rate = 0.0005
epochs = 1000
dim_u = 8
dim_c = [16, 8]
dim_z = [16, 8]
num_hiddens_s = [16, dim_u]
num_hiddens_t = [16, dim_u]
weight_decay = 0.
dropout = 0.
dataset_str = 'TCGA'
use_features = 1
seperate_enc = 1


# helper functions

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs, adj):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs, adj):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvDense(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvDense, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs, adj):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvBi(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvBi, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act
        
    def __call__(self, inp_s, inp_t, adj):
        x = inp_s
        y = inp_t
        
        x = tf.nn.dropout(x, 1-self.dropout)
        y = tf.nn.dropout(y, 1-self.dropout)
        
        x = tf.matmul(x, self.vars['weights'])
        y = tf.matmul(y, self.vars['weights'])
        
        deg_sa = tf.reduce_sum(adj, axis=1, keepdims=True)
        deg_ta = tf.reduce_sum(adj, axis=0, keepdims=True)
        
        x = self.act(x + tf.sparse_tensor_dense_matmul((adj/deg_sa)/deg_ta, y))
        y = self.act(y + tf.sparse_tensor_dense_matmul(tf.transpose((adj/deg_sa)/deg_ta), x))
        return x, y


class GraphConvBiDense(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvBiDense, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act
        
    def __call__(self, inp_s, inp_t, adj):
        x = inp_s
        y = inp_t
        
        x = tf.nn.dropout(x, 1-self.dropout)
        y = tf.nn.dropout(y, 1-self.dropout)
        
        x = tf.matmul(x, self.vars['weights'])
        y = tf.matmul(y, self.vars['weights'])
        
        root_deg_sa = tf.sqrt(tf.reduce_sum(adj, axis=1, keepdims=True) + tf.ones([adj.shape[0], 1]))
        root_deg_ta = tf.sqrt(tf.reduce_sum(adj, axis=0, keepdims=True) + tf.ones([1, adj.shape[1]]))
        nrm_adj = (adj/root_deg_sa)/root_deg_ta
        
        x = self.act(x/root_deg_sa + tf.matmul(nrm_adj, y))
        y = self.act(y/tf.transpose(root_deg_ta) + tf.matmul(tf.transpose(nrm_adj), x))
        return x, y


class GraphConvTriDense(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvTriDense, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act
    
    def norm_adj(self, adj, adj_s, adj_t):
        deg_so = tf.reduce_sum(adj, axis=1, keepdims=True)+tf.sparse_reduce_sum(adj_s, axis=1, keepdims=True)
        root_deg_so = tf.sqrt(deg_so + tf.ones_like(deg_so))
        deg_to = tf.reduce_sum(adj, axis=0, keepdims=True)+tf.sparse_reduce_sum(adj_t, axis=0, keepdims=True)
        root_deg_to = tf.sqrt(deg_to + tf.ones_like(deg_to))
        
        nrm_adj_s = (adj_s/root_deg_so)/tf.transpose(root_deg_so)
        nrm_adj_t = (adj_t/root_deg_to)/tf.transpose(root_deg_to)
        nrm_adj = (adj/root_deg_so)/root_deg_to
        
        return nrm_adj, nrm_adj_s, nrm_adj_t, root_deg_so, root_deg_to
        
    def __call__(self, inp_s, inp_t, adj, adj_s, adj_t):
        x = inp_s
        y = inp_t
        
        x = tf.nn.dropout(x, 1-self.dropout)
        y = tf.nn.dropout(y, 1-self.dropout)
        
        x = tf.matmul(x, self.vars['weights'])
        y = tf.matmul(y, self.vars['weights'])
        
        nrm_adj, nrm_adj_s, nrm_adj_t, root_deg_so, root_deg_to = self.norm_adj(adj, adj_s, adj_t)
        
        x = self.act(x/root_deg_so + tf.sparse_tensor_dense_matmul(nrm_adj_s, x) + tf.matmul(nrm_adj, y))
        y = self.act(y/tf.transpose(root_deg_to) + tf.sparse_tensor_dense_matmul(nrm_adj_t, y) 
                     + tf.matmul(tf.transpose(nrm_adj), x))
        return x, y


class GraphConvShared(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvShared, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act

    def __call__(self, inp_s, inp_t, adj_s, adj_t):
        x = inp_s
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(adj_s, x)
        out_s = self.act(x)
        
        y = inp_t
        y = tf.nn.dropout(y, 1-self.dropout)
        y = tf.matmul(y, self.vars['weights'])
        y = tf.sparse_tensor_dense_matmul(adj_t, y)
        out_t = self.act(y)
        
        return out_s, out_t

    
class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, bias=True, **kwargs):
        super(Dense, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.act = act
        self.bias = bias
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
            if self.bias:
                self.vars['bias'] = tf.Variable(tf.zeros([output_dim], dtype=tf.float32), name='bias')

        if self.logging:
            self._log_vars()

    def __call__(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])
        
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


def dense_to_sparse(x, ignore_value=None, name=None):
    with tf.name_scope(name or 'dense_to_sparse'):
        x = tf.convert_to_tensor(x, name='x')
        if ignore_value is None:
            if base_dtype(x.dtype) == tf.string:
                # Exception due to TF strings are converted to numpy objects by default.
                ignore_value = ''
            else:
                ignore_value = as_numpy_dtype(x.dtype)(0)
            ignore_value = tf.cast(ignore_value, x.dtype, name='ignore_value')
        indices = tf.where(tf.not_equal(x, ignore_value), name='indices')
    return tf.SparseTensor(indices=indices,
                           values=tf.gather_nd(x, indices, name='values'),
                           dense_shape=tf.shape(x, out_type=tf.int64, name='dense_shape'))


class Normal(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = tf.pow(value - self.means, 2)
        log_prob *= - (1 / (2. * tf.exp(2*self.logscales)))
        log_prob -= self.logscales + .5 * math.log(2. * math.pi)
        return log_prob


def construct_feed_dict(adj_normal_s, adj_normal_t, adj_s, adj_t, features_s, features_t, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features_source']: features_s})
    feed_dict.update({placeholders['features_target']: features_t})
    feed_dict.update({placeholders['adj_source']: adj_normal_s})
    feed_dict.update({placeholders['adj_target']: adj_normal_t})
    feed_dict.update({placeholders['adj_orig_source']: adj_s})
    feed_dict.update({placeholders['adj_orig_target']: adj_t})
    return feed_dict

# dtype functions
def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def base_dtype(dtype):
  """Returns a non-reference `dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'base_dtype'):
    return dtype.base_dtype
  return dtype

# preprocessing functions

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


# probabilistic classes

def pairwise_dist (A, B):  
    """Computes pairwise distances between each elements of A and each elements of B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances"""
    with tf.variable_scope('pairwise_dist'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)
        
        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        
        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D


class LogitRelaxedBernoulli(object):
    def __init__(self, logits, temperature=0.3, **kwargs):
        self.logits = logits
        self.temperature = temperature

    def rsample(self):
        eps = tf.distributions.Uniform(low=1e-6, high=1-1e-6).sample(self.logits.shape)
        y = (self.logits + tf.math.log(eps) - tf.math.log(1. - eps)) / self.temperature
        return y

    def log_prob(self, value):
        return math.log(self.temperature) - self.temperature * value + self.logits                - 2 * tf.math.softplus(-self.temperature * value + self.logits)


def logitexp(logp):
    pos = tf.clip_by_value(logp, clip_value_min=-0.69314718056, clip_value_max=1e6)
    neg = tf.clip_by_value(logp, clip_value_min=-1e6, clip_value_max=-0.69314718056)
    neg_val = neg - tf.math.log(1 - tf.math.exp(neg))
    pos_val = -tf.math.log(tf.clip_by_value(tf.math.expm1(-pos), clip_value_min=1e-20, clip_value_max=1e6))
    return pos_val + neg_val


def sample_bipartite(Z1, Z2, g, training=True, temperature=0.3):
    logits = tf.reshape(tf.matmul(Z1, tf.transpose(Z2)), [-1, 1])
    A_vals = tf.cond(training, lambda: tf.sigmoid(LogitRelaxedBernoulli(logits=logits
                                                                        , temperature=temperature).rsample())
                     , lambda: tfp.distributions.Bernoulli(logits=logits, dtype=tf.float32).sample())
    
    
    A = tf.reshape(A_vals, [Z1.shape[0], Z2.shape[0]])
    return A


# model

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class BAYREL(Model):
    def __init__(self, placeholders, num_features_dict, num_nodes_dict, **kwargs):
        super(BAYREL, self).__init__(**kwargs)
        
        self.input_dim_s = num_features_dict['source']
        self.input_dim_t = num_features_dict['target']
        self.inp_dim_u = int(self.input_dim_s / 10.0)
        self.input_dim_s = self.input_dim_s - self.inp_dim_u
        self.input_dim_t = self.input_dim_t - self.inp_dim_u
        self.inputs_s, self.inp_u_s = tf.split(placeholders['features_source']
                                               , [self.input_dim_s, self.inp_dim_u], axis=1)
        self.inputs_t, self.inp_u_t = tf.split(placeholders['features_target']
                                               , [self.input_dim_t, self.inp_dim_u], axis=1)
        self.n_samples_s = num_nodes_dict['source']
        self.n_samples_t = num_nodes_dict['target']
        self.n_samples_ov = self.n_samples_s + self.n_samples_t
        self.adj_s = placeholders['adj_source']
        self.adj_t = placeholders['adj_target']
        self.adj_orig_s = placeholders['adj_orig_source']
        self.adj_orig_t = placeholders['adj_orig_target']
        self.dropout = placeholders['dropout']
        self.wu = placeholders['WU']
        self.pairwise_g_logscale = tf.Variable(math.log(math.sqrt(dim_u)), dtype=tf.float32)
        self.pairwise_g = lambda x: tf.reshape(logitexp(-.5 * x / tf.exp(self.pairwise_g_logscale))
                                               , [x.shape[0], 1])
        self.training = placeholders['training']
        self.build()

    def _build(self):
         # Source and target graph embedding
        self.h_u_s, self.h_u_t = GraphConvShared(input_dim=self.inp_dim_u
                                                 ,output_dim=num_hiddens_s[0]
                                                 ,act=tf.nn.relu
                                                 ,dropout=self.dropout
                                                 ,logging=self.logging)(self.inp_u_s, self.inp_u_t
                                                                        , self.adj_s, self.adj_t)
        self.u_mean_s, self.u_mean_t = GraphConvShared(input_dim=num_hiddens_s[0]
                                                       ,output_dim=num_hiddens_s[1]
                                                       ,act=lambda x: x
                                                       ,dropout=self.dropout
                                                       ,logging=self.logging)(self.h_u_s, self.h_u_t
                                                                              , self.adj_s, self.adj_t)
        self.u_lstd_s, self.u_lstd_t = GraphConvShared(input_dim=num_hiddens_s[0]
                                                       ,output_dim=num_hiddens_s[1]
                                                       ,act=lambda x: x
                                                       ,dropout=self.dropout
                                                       ,logging=self.logging)(self.h_u_s, self.h_u_t
                                                                              , self.adj_s, self.adj_t)
        self.u_s = self.u_mean_s + tf.random_normal([self.n_samples_s, num_hiddens_s[1]])*tf.exp(self.u_lstd_s)
        self.u_t = self.u_mean_t + tf.random_normal([self.n_samples_t, num_hiddens_t[1]])*tf.exp(self.u_lstd_t)
        
        
        self.recong_s = InnerProductDecoder(act=lambda x: x, logging=self.logging)(self.u_s)
        self.recong_t = InnerProductDecoder(act=lambda x: x, logging=self.logging)(self.u_t)
        
        
        # Bipartite graph learning
        self.A = sample_bipartite(self.u_s, self.u_t, self.pairwise_g, training=self.training)  
        
        # Embeding features to the same latent spaces
        self.h_emb_s = slim.fully_connected(self.inputs_s, dim_c[0])
        self.emb_s = slim.fully_connected(self.h_emb_s, dim_c[1], activation_fn=None)
        self.h_emb_t = slim.fully_connected(self.inputs_t, dim_c[0])
        self.emb_t = slim.fully_connected(self.h_emb_t, dim_c[1], activation_fn=None)
                                
        
        # zf, latent spaces for feature reconstruction q(zf|A,X,G)
        self.h_zf_s, self.h_zf_t = GraphConvTriDense(input_dim=dim_c[1]
                                                     ,output_dim=dim_z[0]
                                                     ,act=tf.nn.relu
                                                     ,dropout=self.dropout
                                                     ,logging=self.logging)(self.emb_s
                                                                            , self.emb_t
                                                                            , self.wu * self.A
                                                                            , self.adj_orig_s
                                                                            , self.adj_orig_t)
        self.zf_mean_s, self.zf_mean_t = GraphConvTriDense(input_dim=dim_z[0]
                                                           ,output_dim=dim_z[1]
                                                           ,act=lambda x: x
                                                           ,dropout=self.dropout
                                                           ,logging=self.logging)(self.h_zf_s
                                                                                  , self.h_zf_t
                                                                                  , self.wu * self.A
                                                                                  , self.adj_orig_s
                                                                                  , self.adj_orig_t)
        self.zf_lstd_s, self.zf_lstd_t = GraphConvTriDense(input_dim=dim_z[0]
                                                           ,output_dim=dim_z[1]
                                                           ,act=lambda x: x
                                                           ,dropout=self.dropout
                                                           ,logging=self.logging)(self.h_zf_s
                                                                                  , self.h_zf_t
                                                                                  , self.wu * self.A
                                                                                  , self.adj_orig_s
                                                                                  , self.adj_orig_t)
        self.q_zf_s = Normal(self.zf_mean_s, self.zf_lstd_s)
        self.q_zf_t = Normal(self.zf_mean_t, self.zf_lstd_t)
        self.zf_s = self.zf_mean_s + tf.random_normal([self.n_samples_s, dim_z[1]])*tf.exp(self.zf_lstd_s)
        self.zf_t = self.zf_mean_t + tf.random_normal([self.n_samples_t, dim_z[1]])*tf.exp(self.zf_lstd_t)
        
        
        # reconstruction p(G|zg), p(X|zf)
        self.h_reconf_s_1 = Dense(input_dim=dim_z[1], output_dim=dim_z[1])(self.zf_s)
        self.h_reconf_s = Dense(input_dim=dim_z[1], output_dim=self.input_dim_s)(self.h_reconf_s_1)
        self.reconf_s = Dense(input_dim=self.input_dim_s, output_dim=self.input_dim_s
                              , act=lambda x: x)(self.h_reconf_s)
        self.h_reconf_t_1 = Dense(input_dim=dim_z[1], output_dim=dim_z[1])(self.zf_t)
        self.h_reconf_t = Dense(input_dim=dim_z[1], output_dim=self.input_dim_s)(self.h_reconf_t_1)
        self.reconf_t = Dense(input_dim=self.input_dim_s, output_dim=self.input_dim_t
                              , act=lambda x: x)(self.h_reconf_t)
        
        
        # p(zf|A,U)
        self.p_zf_mean_s, self.p_zf_mean_t = GraphConvBiDense(input_dim=dim_u
                                                              ,output_dim=dim_z[1]
                                                              ,act=lambda x: x
                                                              ,dropout=self.dropout
                                                              ,logging=self.logging)(self.u_s
                                                                                     , self.u_t
                                                                                     , self.A)
        self.p_zf_s = Normal(self.p_zf_mean_s, tf.zeros_like(self.p_zf_mean_s))
        self.p_zf_t = Normal(self.p_zf_mean_t, tf.zeros_like(self.p_zf_mean_t))



# optimizer

class OptimizerVAE(object):
    def __init__(self, preds_dict, labels_dict, model, num_nodes_dict, pos_weight_dict, norm_dict):
        prd_g_s = preds_dict['adj_source']
        lbl_g_s = labels_dict['adj_source']
        prd_f_s = preds_dict['features_source']
        lbl_f_s = labels_dict['features_source']
        posw_s = pos_weight_dict['source']
        num_nodes_s = num_nodes_dict['source']
        norm_s = norm_dict['source']
        
        prd_g_t = preds_dict['adj_target']
        lbl_g_t = labels_dict['adj_target']
        prd_f_t = preds_dict['features_target']
        lbl_f_t = labels_dict['features_target']
        posw_t = pos_weight_dict['target']
        num_nodes_t = num_nodes_dict['target']
        norm_t = norm_dict['target']
        
        # Reconstruction loss
        self.nll_g_s = norm_s * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=prd_g_s
                                                                                        , targets=lbl_g_s
                                                                                        , pos_weight=posw_s))
        self.nll_f_s = tf.reduce_mean(tf.squared_difference(prd_f_s, lbl_f_s))
        self.nll_g_t = norm_t * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=prd_g_t
                                                                                        , targets=lbl_g_t
                                                                                        , pos_weight=posw_t))
        self.nll_f_t = tf.reduce_mean(tf.squared_difference(prd_f_t, lbl_f_t))
        self.nll_s = self.nll_g_s + self.nll_f_s
        self.nll_t = self.nll_g_t + self.nll_f_t
        self.nll = self.nll_s + self.nll_t
        
       # Latent loss U
        self.kl_u_s = (-0.5/num_nodes_s) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.u_lstd_s 
                                                                        - tf.square(model.u_mean_s) 
                                                                        - tf.square(tf.exp(model.u_lstd_s)), 1))
        self.kl_u_t = (-0.5/num_nodes_t) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.u_lstd_t 
                                                                        - tf.square(model.u_mean_t) 
                                                                        - tf.square(tf.exp(model.u_lstd_t)), 1))
        self.kl_u = self.kl_u_s + self.kl_u_t
        
        # Latent loss zg
        self.qpzf_s = tf.reduce_mean(tf.reduce_sum(model.q_zf_s.log_prob(model.zf_s) 
                                                   - 30*model.p_zf_s.log_prob(model.zf_s), 1))
        self.qpzf_t = tf.reduce_mean(tf.reduce_sum(model.q_zf_t.log_prob(model.zf_t) 
                                                   - 30*model.p_zf_t.log_prob(model.zf_t), 1))
        self.qpzf = self.qpzf_s + self.qpzf_t
        self.qpz = self.qpzf
        
        self.cost = self.nll + self.kl_u + self.qpz
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        
        self.recong_s_auc,_ = tf.metrics.auc(tf.sigmoid(prd_g_s), lbl_g_s)
        self.recong_t_auc,_ = tf.metrics.auc(tf.sigmoid(prd_g_t), lbl_g_t)
        
        self.corp_s = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(prd_g_s), 0.5), tf.int32)
                               ,tf.cast(lbl_g_s, tf.int32))
        self.acc_s = tf.reduce_mean(tf.cast(self.corp_s, tf.float32))
        self.corp_t = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(prd_g_t), 0.5), tf.int32)
                               ,tf.cast(lbl_g_t, tf.int32))
        self.acc_t = tf.reduce_mean(tf.cast(self.corp_t, tf.float32))
        
        self.reconf_s_rmse = tf.reduce_mean(tf.squared_difference(prd_f_s, lbl_f_s))
        self.reconf_t_rmse = tf.reduce_mean(tf.squared_difference(prd_f_t, lbl_f_t))    


# Load data

adj_s = np.load('data/breastTCGA/adj_mirTCGA.npy', allow_pickle=True)
adj_s_sp = sp.csr_matrix(adj_s, dtype=np.float64)
feat_s = np.load('data/breastTCGA/feat_mirTCGA.npy', allow_pickle=True)

adj_t = np.load('data/breastTCGA/adj_geneTCGA.npy', allow_pickle=True)
adj_t_sp = sp.csr_matrix(adj_t, dtype=np.int64)
feat_t = np.load('data/breastTCGA/feat_geneTCGA.npy', allow_pickle=True)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig_s = adj_s_sp
adj_orig_s = adj_orig_s - sp.dia_matrix((adj_orig_s.diagonal()[np.newaxis, :], [0]), shape=adj_orig_s.shape)
adj_orig_s.eliminate_zeros()
adj_orig_s = adj_orig_s.tolil()

adj_orig_t = adj_t_sp
adj_orig_t = adj_orig_t - sp.dia_matrix((adj_orig_t.diagonal()[np.newaxis, :], [0]), shape=adj_orig_t.shape)
adj_orig_t.eliminate_zeros()
adj_orig_t = adj_orig_t.tolil()

# Some preprocessing
adj_norm_s = preprocess_graph(adj_orig_s)
adj_norm_t = preprocess_graph(adj_orig_t)

num_nodes_s = adj_s_sp.shape[0]
num_nodes_t = adj_t_sp.shape[0]

features_s = feat_s
features_t = feat_t

num_features_s = features_s.shape[1]
num_features_t = features_t.shape[1]

nfeat_u = int(num_features_s/10)
nfeat_s_recon = num_features_s - nfeat_u
nfeat_t_recon = num_features_t - nfeat_u

num_nodes_dict = {'source': num_nodes_s,'target': num_nodes_t}
num_features_dict = {'source': num_features_s,'target': num_features_t}

# Define placeholders
placeholders = {'features_source': tf.placeholder(tf.float32, shape=[num_nodes_s, num_features_s])
                ,'features_target': tf.placeholder(tf.float32, shape=[num_nodes_t, num_features_t])
                ,'adj_source': tf.sparse_placeholder(tf.float32, shape=[num_nodes_s, num_nodes_s])
                ,'adj_target': tf.sparse_placeholder(tf.float32, shape=[num_nodes_t, num_nodes_t])
                ,'adj_orig_source': tf.sparse_placeholder(tf.float32, shape=[num_nodes_s, num_nodes_s])
                ,'adj_orig_target': tf.sparse_placeholder(tf.float32, shape=[num_nodes_t, num_nodes_t])
                ,'dropout': tf.placeholder_with_default(0., shape=())
                ,'WU': tf.placeholder(tf.float32, shape=())
                ,'training': tf.placeholder(tf.bool, shape=())}

# Create model
model = BAYREL(placeholders, num_features_dict, num_nodes_dict)

pos_weight_s = float(adj_s_sp.shape[0] * adj_s_sp.shape[0] - adj_s_sp.sum()) / adj_s_sp.sum()
pos_weight_t = float(adj_t_sp.shape[0] * adj_t_sp.shape[0] - adj_t_sp.sum()) / adj_t_sp.sum()
pos_weight_dict = {'source': pos_weight_s, 'target': pos_weight_t}

norm_s = adj_s_sp.shape[0] * adj_s_sp.shape[0] / float((adj_s_sp.shape[0] * adj_s_sp.shape[0] - adj_s_sp.sum()) * 2)
norm_t = adj_t_sp.shape[0] * adj_t_sp.shape[0] / float((adj_t_sp.shape[0] * adj_t_sp.shape[0] - adj_t_sp.sum()) * 2)
norm_dict = {'source': norm_s, 'target': norm_t}


# validation data load

target_id_dict = np.load('data/breastTCGA/dict_geneTCGA.npy', allow_pickle=True).tolist()
source_id_dict = np.load('data/breastTCGA/dict_mirTCGA.npy', allow_pickle=True).tolist()
val_intracts_df = pd.read_csv('data/breastTCGA/valIntTCGA.txt', delimiter='\t')

val_interacts = []
for i in range(val_intracts_df.shape[0]):
    val_interacts.append([source_id_dict[val_intracts_df.iloc[i,0]], target_id_dict[val_intracts_df.iloc[i,1]]])

    
def pos_val(biadj, val_interacts):
    counter = 0.0
    for i,e in enumerate(val_interacts):
        if biadj[e[0], e[1]]:
            counter += 1.0
    
    return counter/(1.0*len(val_interacts))


# training

with tf.name_scope('optimizer'):
    preds_dict = {'features_source': model.reconf_s, 'features_target': model.reconf_t
                  ,'adj_source': model.recong_s, 'adj_target': model.recong_t
                 }
    
    labels_dict = {'features_source': tf.slice(placeholders['features_source'], [0, 0]
                                               , [num_nodes_s, nfeat_s_recon])
                   , 'features_target': tf.slice(placeholders['features_target'], [0, 0]
                                               , [num_nodes_t, nfeat_s_recon])
                   ,'adj_source': tf.reshape(tf.eye(num_nodes_s) 
                                            + tf.sparse_tensor_to_dense(placeholders['adj_orig_source']
                                                                      , validate_indices=False), [-1])
                   , 'adj_target': tf.reshape(tf.eye(num_nodes_t) 
                                              + tf.sparse_tensor_to_dense(placeholders['adj_orig_target']
                                                                      , validate_indices=False), [-1])
                  }
    
    opt = OptimizerVAE(preds_dict=preds_dict
                       ,labels_dict=labels_dict
                       ,model=model
                       ,num_nodes_dict=num_nodes_dict
                       ,pos_weight_dict=pos_weight_dict
                       ,norm_dict=norm_dict
                      )

# Initialize session
sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

adj_label_s = sparse_to_tuple(adj_orig_s)
adj_label_t = sparse_to_tuple(adj_orig_t)
for epoch in range(epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm_s, adj_norm_t, adj_label_s, adj_label_t, features_s
                                    , features_t, placeholders)
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['WU']: np.min([epoch/100.0, 0.])})
    feed_dict.update({placeholders['training']: True})
    
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.nll_g_s , opt.nll_g_t, opt.nll_f_s , opt.nll_f_t
                     , opt.kl_u_s, opt.kl_u_t, opt.qpzf_s, opt.qpzf_t
                     , opt.acc_s, opt.acc_t, opt.reconf_s_rmse, opt.reconf_t_rmse
                     , model.recong_s, model.recong_t]
                    , feed_dict=feed_dict)

    feed_dict.update({placeholders['training']: False})
    samp_adj = []
    for i in range(101):
        samp_adj.append(sess.run(model.A, feed_dict=feed_dict))
    
    best_vals = []
    for threshold in range(30, 90):
        post_adj = 1*(np.sum(np.stack(samp_adj), axis=0)>threshold)
        pv = pos_val(post_adj, val_interacts)
        deg = np.mean(post_adj)
        if deg <= 0.5 and pv >= 0.2:
            best_vals.append([epoch, threshold, pv, deg])
    
    print("Epoch:", '%04d' % (epoch + 1)
          ,"cost=", "{:.5f}".format(outs[1])
          ,"nll_g_s=", "{:.5f}".format(outs[2])
          ,"nll_g_t=", "{:.5f}".format(outs[3])
          ,"nll_f_s=", "{:.5f}".format(outs[4])
          ,"nll_f_t=", "{:.5f}".format(outs[5])
          ,"kl_u_s=", "{:.5f}".format(outs[6])
          ,"kl_u_t=", "{:.5f}".format(outs[7])
          ,"qpzf_s=", "{:.5f}".format(outs[8])
          ,"qpzf_t=", "{:.5f}".format(outs[9])
          ,"recong_s_acc=", "{:.5f}".format(outs[10])
          ,"recong_t_acc=", "{:.5f}".format(outs[11])
          ,"reconf_s_rmse=", "{:.5f}".format(outs[12])
          ,"reconf_t_rmse=", "{:.5f}".format(outs[13])
          ,"time=", "{:.5f}".format(time.time() - t))
    print('---------------------------------')
    print('---------------------------------')
    for i in range(len(best_vals)):
        print("Epoch:", '%04d' % (best_vals[i][0] + 1)
              ,"threshold=", '%04d' % (best_vals[i][1])
              ,"pv=", "{:.3f}".format(best_vals[i][2])
              ,"deg=", "{:.3f}".format(best_vals[i][3]))
        print('---------------------------------')
    print('---------------------------------')
    print('---------------------------------')
print("Optimization Finished!")
