using Test
using Chmy
import Chmy: StaticCoef, makeop, ncomponents, linear_index, dimensions
import Chmy: NoKind, SymKind, AltKind, DiagKind

# Test naming convention:
# Reuse the same symbolic field names across the test suite whenever possible,
# because each new symbolic name tends to create extra compilation work.
# When adding or editing tests, prefer these names unless there is a strong
# reason not to:
# - Scalars: `a`, `b`, `c`, `d`
# - Uniform scalars: reuse `a`, `b`, `c`, `d` under `@uniform`
# - Vectors: `u`, `v`, `w`
# - Uniform vectors: reuse `u`, `v`, `w` under `@uniform`
# - General tensors: `T`
# - Symmetric tensors: `S`
# - Alternating tensors: `A`
# - Diagonal tensors: `D`
# - Identity tensors: `I`
# - Zero tensors: `O`
# If a test needs more than one tensor of the same kind, extend the same small
# vocabulary consistently, for example `T, R`, `S, Q`, `A, B`, `D, E`, or
# `I, J`, instead of inventing unrelated names.
# Prefer this convention for both regular and uniform declarations, and for both
# tensor-core tests and other symbolic-expression tests.

include("test_staticcoef.jl")
include("test_tensors.jl")
include("test_calculus.jl")
include("test_node.jl")
include("test_macros.jl")
include("test_rewriters.jl")
include("test_isless_lex.jl")
include("test_canonicalize.jl")
include("test_show.jl")
