# Example circuits for use with benchmarks

from enum import Enum
from cartographer_har.sparse_slice import HAR

class Gate(Enum):
    """ Encoding of circuit generators as integers """
    COPY    = 1 # Δ
    DISCARD = 2 # !
    AND     = 3
    XOR     = 4
    ZERO    = 5
    ONE     = 6
    NOT     = 7

    @property
    def type(self):
        arities = {
            Gate.COPY: (1, 2),
            Gate.DISCARD: (1, 0),
            Gate.AND: (2, 1),
            Gate.XOR: (2, 1),
            Gate.ZERO: (0, 1),
            Gate.ONE: (0, 1),
            Gate.NOT: (1, 1)
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
    """ full_adder() constructs a 2-bit full adder:

            C_in ---\         /--- Result
                     \-|---|-/
            A    ------| + |
                     /-|---|-\
            B    ---/         \--- Cout

    """
    f = add_mul()
    id = HAR.identity(1)
    return (f @ id) >> (id @ HAR.twist(1, 1)) >> (f @ id) >> (id @ f) >> (id @ Gate.XOR.HAR)

def adder(n):
    """ adder(n) constructs a 2^n-bit adder """
    # NOTE: strictly speaking this assumes the input data is in a particular
    # order which is not how you'd normally implement things- but one can
    # definitely consider this an n-bit adder.
    if n < 0:
        raise ValueError("Can't construct an n-bit adder for n < 0")
    if n == 0:
        # If an n-bit full adder is (Cin × n × n) → (n × Cout) then this makes sense :-)
        return HAR.identity(1)
    elif n == 1:
        return full_adder()
    else:
        id = HAR.identity(2**(n - 1))
        a = adder(n - 1) # recurse
        return (a @ id @ id) >> (id @ a)
    
    return circuit

###############################
# Benchmarks

MAX_LOG2_SIZE = 20

class HarBenchmark:
    params = list(range(0, MAX_LOG2_SIZE + 1))
    param_names = ['n']

class TensorTime(HarBenchmark):
    def setup(self, n):
        acc = Gate.AND.HAR
        for i in range(0, n):
            acc = acc @ acc
        self.lhs = acc
        self.rhs = acc

    def time_tensor(self, n):
        self.lhs @ self.rhs


class ComposeSmallBoundary(HarBenchmark):
    def setup(self, n):
        acc = Gate.NOT.HAR
        for i in range(0, n):
            acc = acc >> acc
        self.lhs = acc
        self.rhs = acc

    def time_small_boundary(self, n):
        self.lhs >> self.rhs

class ComposeLargeBoundary:
    def setup(self, n):
        acc = Gate.NOT.HAR
        for i in range(0, n):
            acc = acc @ acc
        self.lhs = acc
        self.rhs = acc

    def time_tensor(self, n):
        self.lhs >> self.rhs

class AdderCircuit:
    def setup(self, n):
        pass

    # def time_

class SyntheticBenchmark(HarBenchmark):
    def setup(self, n):
        # compute the basic building block before running the benchmark
        self.adder = adder(n)

    # Construct a 2^(n+1)-bit adder from two 2^n-1 bit adders.
    def time_n_bit_adder(self, n):
        id = HAR.identity(2**n)
        return (self.adder @ id @ id) >> (id @ self.adder)
