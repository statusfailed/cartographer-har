import numpy as np
from cartographer_har.linear_algebra import *

class HAR:
    def __init__(self, M, L, R, arity):
        size = M.shape[0]
        # M must be a square matrix
        assert M.shape[0] == M.shape[1]
        self.M = M
        
        # TODO: assert L and R must be permutations (A (resp. B) unique values in [0, self.size])
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
        return HAR(np.zeros((n, n), dtype), np.identity(n, dtype), np.identity(n, dtype), (n, n))
    
    # TODO: general permutations instead of twists
    @staticmethod
    def twist(a, b, dtype=int):
        n = a + b
        return HAR(
            M=np.zeros((n, n), dtype),
            L=np.identity(n, dtype),
            R=twist(a, b, dtype),
            arity=(n, n),
        )
    
    def left_boundary_order(self):
        M = self.L @ self.M @ self.L.T # isomorphic graph
        L = np.identity(self.size, self.L.dtype)
        R = self.R @ self.L.T
        return HAR(M, L, R, self.arity)
    
    def right_boundary_order(self):
        M = self.R @ self.M @ self.R.T # isomorphic graph
        R = np.identity(self.size, self.L.dtype)
        L = self.L @ self.R.T
        return HAR(M, L, R, self.arity)
    
    @staticmethod
    def singleton(label, arity, dtype=int):
        # E.g., for a 2 -> 1 operation, we have the matrix
        #     0 0 0 0
        #     0 0 0 0
        #     1 1 0 0
        #     0 0 1 0
        a = arity[0]
        b = arity[1]
        n = a + b + 1
        
        M = np.zeros((n, n), dtype=dtype)
        M[a, :a] = np.arange(0, a) + 1 # TODO: np.arange! to get 1, 2, 3 ...
        M[a+1:, a] = np.arange(0, b) + 1
        
        L = np.identity(n, int) # NOTE: this means that in "left-boundary order", left nodes come first.
        R = np.identity(n, int)
        return HAR(M, L, R, np.array(arity))
    
    def tensor(f, g):
        a1, b1 = f.arity
        a2, b2 = g.arity

        M = direct_sum(f.M, g.M)
        L = exchange(a1, f.size - a1, a2, g.size - a2, f.L.dtype) @ direct_sum(f.L, g.L)
        R = exchange(f.size - b1, b1, g.size - b2, b2, g.L.dtype) @ direct_sum(f.R, g.R)

        return HAR(M, L, R, (a1 + a2, b1 + b2))
    
    def compose(f, g):
        f = f.right_boundary_order()
        g = g.left_boundary_order()
        a, b = f.arity
        b, c = g.arity

        M = direct_sum(f.M[:, :f.size-b], g.M[b:, :])
        L = direct_sum(f.L, np.identity(g.size - b, g.L.dtype))
        R = offset_compose(f.R, g.R, b)
        return HAR(M, L, R, (a, c))
