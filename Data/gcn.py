import numpy as np
import cobra
from numpy.linalg import svd, norm
import lasagne

def _check_reaction_and_metabolite_ids(model):
    if set(r.id for r in model.reactions) & set(m.id for m in model.metabolites):
        raise ValueError(
            "The following ID's are both in reactions and metabolites: " +
            ", ".join(set(r.id for r in model.reactions) & set(m.id for m in model.metabolites))
        )


def model_to_A(model, metabolite_indices, reaction_indices, connect_self=False, ):
    """
    Makes an adjecency matrix of the connectivity of a stoichiometric model. The model
    is formulated as a bipartite graph with each node corresponding to either a metabolite
    or a reaction.
    The nodes are ordered as metabolites first followed by reactions.

    The stoiciometric connections are represented in the off-diagonal blocks of the adjacency
    matrix (edges between a metabolite and a reaction).

    :param model: A cobra model
    :param connect_self: Boolean. If True the diagonal elements will be 1, otherwise 0.
    :return:
    """
    n_mets = len(model.metabolites)
    n_reacs = len(model.reactions)
    size = n_mets + n_reacs

    _check_reaction_and_metabolite_ids(model)

    # metabolite_indices = {met.id: i for i, met in enumerate(model.metabolites)}
    # reaction_indices = {reac.id: n_mets + i for i, reac in enumerate(model.reactions)}

    if connect_self:
        A = np.diag([1 for _ in range(size)])
    else:
        A = np.zeros([size, size])

    for reac in model.reactions:
        reac_index = reaction_indices[reac.id]
        for met, coef in reac.metabolites.items():
            met_index = metabolite_indices[met.id]
            A[reac_index, met_index] = A[met_index, reac_index] = coef

    return A


def degree_matrix(A):
    A_sum = np.abs(A).sum(1)
    inv_A_sum = 1 / A_sum
    return np.diag(inv_A_sum)


def normalized_adjacency_matrix(A):
    D = degree_matrix(A)
    A_hat = np.sqrt(D).dot(A).dot(np.sqrt(D))
    return A_hat


def model_to_sif_file(model):
    """
    Return cytoscape sif-formatted data for the metabolic network
    :param model:
    :return:
    """
    _check_reaction_and_metabolite_ids(model)

    lines = []

    for reac in model.reactions:
        for met, coef in reac.metabolites.items():
            if coef > 0:
                fields = (reac.id, "stoichiometry", met.id)
            else:
                fields = (met.id, "stoichiometry", reac.id)

            lines.append(" ".join(fields))

    return "\n".join(lines) + "\n"


# Taken from http://wiki.scipy.org/Cookbook/RankNullspace
def nullspace(matrix, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    matrix : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    matrix = np.atleast_2d(matrix)
    u, s, vh = svd(matrix)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def calculate_cosine_similarity(a):
    """
    Pairwise cosine similarity calculations of rows/columns in a matrix
    :param a: Array of dimension 2

    :return: Square array with shape (len(a), len(a))
    """
    out = np.diag([1. for _ in range(len(a))])
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            out[i, j] = out[j, i] = np.dot(a[i], a[j]) / (norm(a[i]) * norm(a[j]))
    return out


def calculate_flux_coupling(model):
    s_model = cobra.util.create_stoichiometric_matrix(model)
    ns = nullspace(s_model)
    ns = ns / ns.sum(0)
    coupling_matrix = calculate_cosine_similarity(ns)
    return coupling_matrix


class GCNLayer(lasagne.layers.Layer):
    def __init__(
            self, incoming, A_hat, num_out_features=3, num_graphs=2,
            W=lasagne.init.GlorotUniform(),
            nonlinearity=lasagne.nonlinearities.sigmoid,
            **kwargs
    ):
        super(GCNLayer, self).__init__(incoming, **kwargs)
        self.A_hat = A_hat
        self.num_graphs = num_graphs
        self.num_out_features = num_out_features
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.linear
        self.nonlinearity = nonlinearity

        print([self.num_graphs, incoming.output_shape[2], num_out_features])
        self.W = self.add_param(W, [self.num_graphs, incoming.output_shape[2], num_out_features], name="W")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_out_features)

    def get_output_for(self, input, **kwargs):
        A_X = T.tensordot(self.A_hat, input, axes=[1, 1]).transpose(2, 1, 0, 3)
        Z = T.tensordot(A_X, self.W, axes=2)
        return self.nonlinearity(Z)
