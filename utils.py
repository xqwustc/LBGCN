#!/usr/local/bin/python
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
from scipy.sparse import csr_matrix


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _convert_sp_mat_to_sp_tensor_(X):
    coo = sp.coo_matrix(X).astype(np.float64)
    indices = np.mat([coo.row, coo.col]).transpose()  # transpose()函数将数组转置  mat() 函数用于构造矩阵
    return tf.SparseTensor(indices, coo.data, coo.shape)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# def load_data(dataset_str):
#     """
#     Loads input data from gcn/data directory
#
#     ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
#         (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
#     ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
#     ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
#     ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
#         object;
#     ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
#
#     All objects above must be saved using python pickle module.
#
#     :param dataset_str: Dataset name
#     :return: All data input files loaded (as well the training/test data).
#     """
#     datafile = '../../../data/'
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open(datafile+"ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file(datafile+"ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y)+500)
#
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#
#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]
#
#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)  # 矩阵 a*b
    return res


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation.将稀疏矩阵转换为元组表示"""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix.对称规范化邻接矩阵"""
#     print('adj:', adj, type(adj))  # <class 'scipy.sparse.csr.csr_matrix'>
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def preprocess_adj(adj):
#     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation
#     简单GCN模型的邻接矩阵预处理及元组表示的转换."""
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # 给邻阶矩阵加上单位矩阵在归一化
#     return sparse_to_tuple(adj_normalized)

def normalize_adj(adj):  # 把不带自环的邻接矩阵加上自环再求平均归一化后的邻接矩阵
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()

    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj)
    norm_adj = norm_adj.dot(d_mat_inv)
    pre_adj_loop = norm_adj.tocsr()

    return pre_adj_loop


def Side_BILinear_pooling(adj_fold_hat, X, n_fold):
    B_fold_hat = adj_fold_hat
    # step1 sum_squared
    # sum = dot(adj_, XW, True)
    # sum = dot(adj_, X, True)
    temp_embed_1 = []

    for f in range(n_fold):
        temp_embed_1.append(tf.sparse_tensor_dense_matmul(B_fold_hat[f], X))

        sum = tf.concat(temp_embed_1, 0)

        sum_squared = tf.multiply(sum, sum)

        # step2 squared_sum
        # squared = tf.multiply(XW, XW)
        squared = tf.multiply(X, X)

    temp_embed_2 = []

    for f in range(n_fold):
        temp_embed_2.append(tf.sparse_tensor_dense_matmul(B_fold_hat[f], squared))

    squared_sum = tf.concat(temp_embed_2, 0)
    # squared_sum = dot(adj_, squared, True)

    # step3
    new_embedding = 0.5 * (sum_squared - squared_sum)
    return new_embedding


def split_u_i(X, n_users, n_items):
    user_embedding, item_embedding = tf.split(X, [n_users, n_items], 0)
    return user_embedding, item_embedding


def get_user(adj, n_users, n_items):
    i = 1
    row_list = adj.getrow(0)[:, n_users:]
    user_adj = row_list
    while i < n_users:
        temp = adj.getrow(i)
        li = temp[:, n_users:]
        user_adj = sp.vstack((user_adj, li), format='csr')
        i += 1
    return user_adj


def get_item(adj, n_users, n_items):
    i = n_users + 1
    row_list = adj.getrow(n_users)[:, 0:n_users]
    user_adj = row_list
    while i < n_items + n_users:
        temp = adj.getrow(i)
        li = temp[:, 0:n_users]
        user_adj = sp.vstack((user_adj, li), format='csr')
        i += 1
    return user_adj


def preprocess_bilinear(adj):
    # adj = adj + sp.eye(adj.shape[0]) # 可能需要添加自环
    # adj_loop = adj.toarray() + np.eye(adj.toarray().shape[0])
    adj_loop = adj.toarray()

    # D_all = adj_loop.sum(1) + np.ones((adj_loop.shape[0],1))  # 添加自环需要再加一个1
    D_all = adj_loop.sum(1)   # 添加自环需要再加一个1
    N_all = 0.5 * np.multiply(D_all, D_all - np.ones((adj.shape[0])))
    N_all = np.diag(1. / N_all)
    N_all[np.isinf(N_all)] = 0.
    # N_all = sp.coo_matrix(N_all)
    # adj_loop = sp.coo_matrix(adj_loop)
    # return sparse_to_tuple(N_all)
    return N_all


def get_N_all(adj, n_users, n_items):
    # 得到user和item的邻接矩阵
    user_adj = get_user(adj, n_users, n_items)
    item_adj = get_item(adj, n_users, n_items)

    # 得到user和item的度矩阵
    user_N_all = preprocess_bilinear(user_adj)
    item_N_all = preprocess_bilinear(item_adj)

    # return sparse_to_tuple(user_N_all), sparse_to_tuple(item_N_all)
    return user_N_all, item_N_all


def BILinear_pooling(adj_fold_hat, X, n_fold, n_users, n_items, adj):
    embedding = Side_BILinear_pooling(adj_fold_hat, X, n_fold)

    u_embedding, i_embedding = split_u_i(embedding, n_users, n_items)

    user_N_all, item_N_all = get_N_all(adj, n_users, n_items)

    user_embedding = dot(tf.cast(_convert_sp_mat_to_sp_tensor_(user_N_all), tf.float32), u_embedding, True)

    item_embedding = dot(tf.cast(_convert_sp_mat_to_sp_tensor_(item_N_all), tf.float32), i_embedding, True)

    all_embedding = tf.concat([user_embedding, item_embedding], axis=0)

    return all_embedding
