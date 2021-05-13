using Printf
using BenchmarkTools
using Catlab.WiringDiagrams
using Catlab.Theories

function mk_generators()
  # Set up a theory
  A = Ob(FreeSymmetricMonoidalCategory, :A)

  # "primitive" wiring diagrams
  idw   = to_wiring_diagram(id(A))
  twist = to_wiring_diagram(braid(A, A))
  copy  = to_wiring_diagram(Hom(:copy, A, A⊗A))
  xor   = to_wiring_diagram(Hom(:xor, A⊗A, A))
  and   = to_wiring_diagram(Hom(:and, A⊗A, A))
  not   = to_wiring_diagram(Hom(:not, A, A))
  
  return idw, twist, copy, xor, and, not
end

# NOTE: we pass the primitive generators into full_adder because BenchmarkTools
# advises that using global variables is likely to cause a slowdown.
function full_adder(primitives)
  idw, twist, copy, xor, and, not = primitives
  # some bits we need
  ex = idw ⊗ twist ⊗ idw
  copy2 = compose(copy ⊗ copy, ex)
  add_mul = compose(copy2, xor ⊗ and)

  # the circuit
  return compose(
    (add_mul ⊗ idw),
    (idw     ⊗ twist),
    (add_mul ⊗ idw),
    (idw     ⊗ add_mul),
    (idw     ⊗ xor)
  )
end

################################
## Tensor Benchmark

function setup_tensor(n)
  idw, twist, copy, xor, and, not = mk_generators()
  acc = and
  for i in 0:n
    acc = acc ⊗ acc
  end

  return acc, acc
end

function time_tensor(lhs_rhs)
  lhs, rhs = lhs_rhs
  return lhs ⊗ rhs
end

################################
## Composition (Small Boundary)

function setup_compose_small(n)
  idw, twist, copy, xor, and, not = mk_generators()
  acc = not
  for i in 0:n
    acc = compose(acc, acc)
  end

  return acc, acc
end

function time_compose_small(lhs_rhs)
  lhs, rhs = lhs_rhs
  return compose(lhs, rhs)
end

################################
## Composition (Large Boundary)

function setup_compose_large(n)
  idw, twist, copy, xor, and, not = mk_generators()
  acc = not
  for i in 0:n
    acc = acc ⊗ acc
  end

  return acc, acc
end

function time_compose_large(lhs_rhs)
  lhs, rhs = lhs_rhs
  return compose(lhs, rhs)
end

################################
## Run

# This code is a bit ugly, but it'll do.
function verbose_run_benchmark(name :: String, setup, bench, time_per_sample :: Int, num_samples :: Int)
  for n in 1:20
    @printf "\n-------------------------\n"
    @printf "%s   n = %d\n" name n
    b = @benchmarkable $(bench)($(setup(n)))
    r = run(b, seconds=time_per_sample*num_samples, samples=num_samples)
    display(r)
    if length(r.times) < 10
      @printf "\nDNF n=%d\n" n
      break
    end
  end
end

function run_benchmark(name :: String, setup, bench, time_per_sample :: Int, num_samples :: Int)
  for n in 1:20
    b = @benchmarkable $(bench)($(setup(n)))
    r = run(b, seconds=time_per_sample*num_samples, samples=num_samples)

    if length(r.times) >= 10
      for (i, t) in enumerate(r.times)
        # Poor man's CSV
        @printf "%s,%d,%d,%f\n" name n (i - 1) t
      end
    else
      # Did not finish
      break
    end

  end
end

@printf "benchmark,n,sample,time\n"
time_per_sample = 60 # seconds
num_samples = 10
run_benchmark("tensor",        setup_tensor,        time_tensor,        time_per_sample, num_samples)
run_benchmark("compose_small", setup_compose_small, time_compose_small, time_per_sample, num_samples)
run_benchmark("compose_large", setup_compose_large, time_compose_large, time_per_sample, num_samples)
