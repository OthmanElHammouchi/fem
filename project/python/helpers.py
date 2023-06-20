import numpy as np
import scipy.sparse as sp
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from tqdm import tqdm

def fem_solve(nodes, elems, h, bc, show_progress=True):
    n_nodes = nodes.size
    n_elems = elems.shape[0]

    stiff_mat = sp.lil_array((n_nodes, n_nodes))
    load_vec = np.zeros(n_nodes)

    disable = not show_progress
    for e in tqdm(range(n_elems), disable=disable):
        k = elems[e, 2]
        q = elems[e, 3]

        element_matrix = (k / h) * np.array([[1, -1], [-1, 1]])
        element_vec = ((q * h) / 2 ) * np.array([1, 1])

        for r in range(2):
            load_vec[int(elems[e, r])] += element_vec[r]
            for s in range(2):
                stiff_mat[int(elems[e, r]), int(elems[e, s])] += element_matrix[r, s]


    node_idx = int(bc[0])
    val = bc[1]
    load_vec -= val * stiff_mat[:, [node_idx]].toarray().flatten()
    load_vec[node_idx] = val

    stiff_mat[[node_idx], :] = 0
    stiff_mat[:, [node_idx]] = 0
    stiff_mat[node_idx, node_idx] = 1

    sol = sp.linalg.spsolve(stiff_mat.tocsc(), load_vec)
    res = interp.interp1d(nodes, sol)
    return res


def exact_solve(x_part, k_vals, q_vals, bc):
    n_pts = len(x_part)
    n_parts = n_pts - 1

    lhs = np.zeros((2 * n_parts, 2 * n_parts))
    lhs[0, 0] = x_part[0]
    lhs[0, 1] = 1
    lhs[-1, -2] = 1

    rhs = np.zeros(2 * n_parts)
    rhs[0] = bc[1] + (q_vals[0] / (2 * k_vals[0])) * x_part[0] ** 2
    rhs[-1] = (q_vals[-1] / k_vals[-1]) * x_part[-1]

    for k in range(1, n_parts):
        lhs[2 * k - 1, 2 * k] = k_vals[k]
        lhs[2 * k - 1, 2 * (k - 1)] = -k_vals[k - 1]

        lhs[2 * k, 2 * k] = x_part[k]
        lhs[2 * k, 2 * (k - 1)] = -x_part[k]
        lhs[2 * k, 2 * k + 1] = 1
        lhs[2 * k, 2 * (k - 1) + 1] = -1

        rhs[2 * k] = (
            q_vals[k] / (2 * k_vals[k]) - q_vals[k - 1] / (2 * k_vals[k - 1])
        ) * x_part[k] ** 2
        rhs[2 * k - 1] = (
            q_vals[k] - q_vals[k - 1]
        ) * x_part[k]

    coeffs = np.linalg.solve(lhs, rhs)

    def res(x):
        idx = np.asarray(x >= x_part).nonzero()[0][-1]
        if idx == n_pts - 1:  # edge case where x is right endpoint
            idx -= 1
        c1 = coeffs[2 * idx]
        c2 = coeffs[2 * idx + 1]
        out = -(q_vals[idx] / (2 * k_vals[idx])) * x**2 + c1 * x + c2
        return out

    return res

def gen_mesh(h, x_part, k_vals, q_vals, Ta, randomise = True):
  a = x_part[0]
  b = x_part[-1]
  n_elems = int((b - a) / h)
  n_nodes = n_elems + 1

  if randomise:
    rng = np.random.default_rng()
    perm = rng.permutation(np.arange(n_nodes)).astype(int)

    nodes = np.linspace(a, b, n_nodes)
    nodes = nodes[perm]
    elems = np.zeros((n_elems, 4))

    indices = np.empty(n_nodes, dtype=int)
    indices[perm] = np.arange(n_nodes)
    elems[:, :2] = np.transpose(np.vstack([indices[:-1], indices[1:]]))
    bc = [indices[0], Ta]

    for e in range(n_elems):
        l_node = nodes[int(elems[e, 0])]
        elems[e, 2] = k_vals[np.asarray(l_node >= x_part).nonzero()[0][-1]]
        elems[e, 3] = q_vals[np.asarray(l_node >= x_part).nonzero()[0][-1]]

    elems = elems[rng.permutation(np.arange(n_elems)), :]
    
  else:
    bc = [0, Ta]
    nodes = np.linspace(a, b, n_nodes)
    elems = np.zeros((n_elems, 4))

    for e in range(n_elems):
        elems[e, 0] = e
        elems[e, 1] = e + 1
        l_node = nodes[int(elems[e, 0])]
        elems[e, 2] = k_vals[np.asarray(l_node >= x_part).nonzero()[0][-1]]
        elems[e, 3] = q_vals[np.asarray(l_node >= x_part).nonzero()[0][-1]]
  
  return nodes, elems, bc