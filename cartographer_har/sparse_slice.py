import scipy.sparse as sp
from cartographer_har.linear_algebra import *

class Mat:
    @classmethod
    def twist(cls, a: int, b: int, dtype):
        """
        Build the block matrix
            0 I
            I 0
        """
        za = cls.zeros((b, a), dtype)
        zd = cls.zeros((a, b), dtype)
        return cls.blocks(za, cls.identity(b, dtype), cls.identity(a, dtype), zd)

    @classmethod
    def exchange(cls, a, b, c, d, dtype=int):
        return cls.direct_sums([
            cls.identity(a, dtype=dtype),
            cls.twist(b, c, dtype=dtype),
            cls.identity(d, dtype=dtype)
        ])

    @classmethod
    def direct_sum(cls, A, B):
        return cls.direct_sums([A, B])

class Sparse(Mat):
    @staticmethod
    def zeros(shape, dtype, format='csr'):
        if format == 'csr':
            return sp.csr_matrix(shape, dtype=dtype)
        elif format == 'lil':
            return sp.lil_matrix(shape, dtype=dtype)
        else:
            raise ValueError(f"unsupported format {format}")

    @staticmethod
    def identity(n: int, dtype):
        return sp.identity(n, dtype)

    @staticmethod
    def blocks(a, b, c, d):
        """ Build the matrix
                A B
                C D
        """
        return sp.bmat([[a, b], [c, d]])

    @staticmethod
    def direct_sums(mats, format='csr', dtype=None):
        return sp.block_diag(mats, format=format, dtype=dtype)

class HAR:
    def __init__(self, M, L, R, N, arity):
        size = M.shape[0]
        # M must be a square matrix
        assert M.shape[0] == M.shape[1]
        self.M = M

        # Number of node labels must be equal to matrix dimensions
        assert N.shape[0] == M.shape[0]
        self.N = N

        assert L.shape == (size, size)
        self.L = L

        assert R.shape == (size, size)
        self.R = R

        self.arity = np.array(arity)

    @property
    def size(self):
        return self.M.shape[0]

    @property
    def dom(self):
        return self.arity[0]

    @property
    def cod(self):
        return self.arity[1]

    @staticmethod
    def identity(n, dtype=int):
        return HAR(
            M=Sparse.zeros((n, n), dtype),
            L=Sparse.identity(n, dtype),
            R=Sparse.identity(n, dtype),
            # NOTE: scipy sparse needs explicit dimensions for SpGEMV
            N=Sparse.zeros((n, 1), dtype),
            arity=(n, n))

    @staticmethod
    def twist(a, b, dtype=int):
        n = a + b
        return HAR(
            M=Sparse.zeros((n, n), dtype),
            L=Sparse.identity(n, dtype),
            R=Sparse.twist(a, b, dtype),
            N=Sparse.zeros((n, 1), dtype),
            arity=(n, n),
        )

    def left_boundary_order(self):
        M = self.L @ self.M @ self.L.T # isomorphic graph
        L = Sparse.identity(self.size, self.L.dtype)
        R = self.R @ self.L.T
        N = self.L @ self.N
        return HAR(M, L, R, N, self.arity)

    def right_boundary_order(self):
        M = self.R @ self.M @ self.R.T # isomorphic graph
        R = Sparse.identity(self.size, self.L.dtype)
        L = self.L @ self.R.T
        N = self.R @ self.N
        return HAR(M, L, R, N, self.arity)

    # NOTE: we assume the label is encoded as an int
    @staticmethod
    def singleton(label: int, arity, dtype=int):
        a = arity[0]
        b = arity[1]
        n = a + b + 1

        # NOTE: we construct as a dense matrix and then convert to sparse
        M = np.zeros((n, n), dtype=dtype)
        M[a, :a] = np.arange(0, a) + 1
        M[a+1:, a] = np.arange(0, b) + 1
        M = sp.csr_matrix(M)

        N = Sparse.zeros((n, 1), dtype, format='lil')
        N[a] = label
        N = sp.csr_matrix(N)

        L = Sparse.identity(n, int)
        R = Sparse.identity(n, int)
        return HAR(M, L, R, N, np.array(arity))

    def tensor(f, g):
        a1, b1 = f.arity
        a2, b2 = g.arity

        M = Sparse.direct_sums([f.M, g.M], format='csr')
        L = Sparse.exchange(a1, f.size - a1, a2, g.size - a2, f.L.dtype) @ Sparse.direct_sum(f.L, g.L)
        R = Sparse.exchange(f.size - b1, b1, g.size - b2, b2, g.L.dtype) @ Sparse.direct_sum(f.R, g.R)
        N = sp.vstack([f.N, g.N])

        return HAR(M, L, R, N, (a1 + a2, b1 + b2))

    def compose(f, g):
        f = f.right_boundary_order()
        g = g.left_boundary_order()
        a, b = f.arity
        b, c = g.arity

        M = Sparse.direct_sums([f.M[:, :f.size-b], g.M[b:, :]])
        L = Sparse.direct_sums([f.L, Sparse.identity(g.size - b, g.L.dtype)])
        R = Sparse.direct_sums([Sparse.identity(f.size - b, f.R.dtype), g.R])
        N = sp.vstack([f.N, g.N[b:]])

        return HAR(M, L, R, N, (a, c))

    # infix for tensor
    def __matmul__(f, g):
        return f.tensor(g)

    # infix for composition
    def __rshift__(f, g):
        return f.compose(g)
