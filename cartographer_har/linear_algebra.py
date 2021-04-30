import numpy as np
import math

# Construct the block matrix
#   A B
#   C D
def blocks(A, B, C, D):
    return np.vstack([
        np.hstack([A, B]),
        np.hstack([C, D])
    ])

def project(i, n, dtype=int):
    r = np.zeros(n, dtype=dtype)
    r[i] = 1
    return r.reshape((1, n))

# (a x b) → (b x a)
def twist(a, b, dtype=int):
    za = np.zeros((b, a), dtype)
    zd = np.zeros((a, b), dtype)
    return blocks(za, np.identity(b, dtype), np.identity(a, dtype), zd)

# Reflexive transitive closure of an adjacency matrix
# FIXME: should safely add the identity matrix- only set if values are zero
def star(X):
    # if X.dtype != bool:
        # raise ValueError("X.dtype must be bool")

    (n, m) = X.shape
    if n != m:
        raise ValueError("X must be square")

    # reflexive
    # TODO: use fill_diagonal + min(1, X.diagonal()) here ?
    X += np.identity(n, X.dtype)

    # transitive: compute log(n) squarings, stopping when X doesn't change anymore
    # NOTE: this exits immediately when X is the adjacency matrix for a discrete graph
    for _ in range(0, math.ceil(math.log2(n))):
        Xprime = X @ X
        if np.all((Xprime == X).flatten()):
            break

    return X

# FIXME: allow passing callback to compute reflexive, transitive closure of D.
# This is useful when we know in advance that D will be the zero matrix, for example
def trace(M, u):
    # Decompose M into the block matrix
    #   A B
    #   C D
    n = M.shape[0] - u
    A = M[:n, :n]
    B = M[:n, n:]
    C = M[n:, :n]
    D = M[n:, n:]

    return A + B @ star(D) @ C


def direct_sum(A, B):
    Am, An = A.shape
    Bm, Bn = B.shape

    assert A.dtype == B.dtype

    return np.vstack([
        np.hstack([A, np.zeros((Am, Bn), dtype=A.dtype)]),
        np.hstack([np.zeros((Bm, An), dtype=A.dtype), B])
    ])


def direct_sums(xs):
    if xs == []:
        raise ValueError("direct_sums must take an array of matrices")

    if len(xs) == 1:
        return xs[0]

    # NOTE: this is not very efficient- we should preallocate the whole array.
    acc = xs[0]
    for x in xs[1:]:
        acc = direct_sum(acc, x)

    return acc

# The permutation matrix representing (id x twist x id)
def exchange(a, b, c, d, dtype=int):
    return direct_sums([
        np.identity(a, dtype),
        twist(b, c, dtype),
        np.identity(d, dtype)
    ])

# "offset" compose two matrices along a shared "boundary" of n wires
def offset_compose(F, G, n):
    return direct_sum(np.identity(F.shape[1] - n, F.dtype), G) @ \
           direct_sum(F, np.identity(G.shape[0] - n, G.dtype))
