# Example circuits for use with benchmarks

from enum import Enum
from cartographer_har.sparse_slice import HAR

class Gate(Enum):
    """ Encoding of circuit generators as integers """
    COPY    = 1 # Δ
    DISCARD = 2 # !
    AND     = 3
    XOR     = 4

    @property
    def type(self):
        arities = {
            Gate.COPY: (1, 2),
            Gate.DISCARD: (1, 0),
            Gate.AND: (2, 1),
            Gate.XOR: (2, 1)
        }

        return arities[self]

    @property
    def HAR(self):
        return HAR.singleton(self.value, self.type)

################################################################################
# Utilities

def ex():
    return HAR.identity(1) @ HAR.twist(1, 1) @ HAR.identity(1)

def copy2():
    return Gate.COPY.HAR >> ex()

# copy_2 ; (xor × and)
def add_mul():
    return copy2() >> (Gate.XOR.HAR @ Gate.AND.HAR)

def full_adder():
    f = add_mul()
    id = HAR.identity(1)
    return (f @ id) >> (id @ HAR.twist(1, 1)) >> (f @ id) >> (id @ f) >> (id @ Gate.XOR.HAR)

def adder(n):
    """ adder(n) constructs a 2^n bit adder
    """
    circuit = full_adder()
    for i in range(0, power):
        pass

class BasicSuite:
    def time_identity(self):
        HAR.identity(1)

    def time_identity_tensor_identity(self):
        HAR.identity(1) @ HAR.identity(1) @ HAR.identity(1) @ HAR.identity(1)

    def time_identity_tensor_identity_v2(self):
        (HAR.identity(1) @ HAR.identity(1)) @ (HAR.identity(1) @ HAR.identity(1))

class TensorTime:
    # log_2(size) of circuits to tensor together
    params = list(range(0, 21))
    # params = list(range(0, 6)) # dense- 14.9s @ 5(!!!)
    param_names = ['log_2(n)']

    def setup(self, n):
        self.lhs = full_adder()
        self.rhs = full_adder()
        for i in range(0, n):
            self.lhs = self.lhs @ self.rhs
            self.rhs = self.rhs @ self.rhs

    def time_tensor(self, n):
        self.lhs @ self.rhs

class CircuitSuite:
    # params = [ 2**i for i in range(0, 8) ]
    params = list(range(0, 12))

    def setup(self, n):
        # compute the basic building block before running the benchmark
        self.full_adder = full_adder()

    def time_full_adder_chain(self, n):
        # Compute 2^n tensors of a full adder block
        a = self.full_adder
        for i in range(0, n):
            a = a @ a
