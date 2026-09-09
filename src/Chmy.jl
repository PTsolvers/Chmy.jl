module Chmy

using KernelAbstractions

using Moshi: Data.@data, Data.isa_variant, Match.@match

using PrettyTables: pretty_table, TextTableFormat
import Adapt

import LinearAlgebra: ⋅, ×, tr, det, diag, transpose
import Base: broadcasted

# re-export from LinearAlgebra
export ⋅, ×, tr, det, diag, transpose

export Operator

"""
Abstract supertype for all Chmy operators.
"""
abstract type Operator end

export 𝓅, 𝓈
include("locations.jl")

module Kind
struct Sym end
struct Alt end
struct Diag end
struct None end
end
const TensorKind = Union{Kind.Sym,Kind.Alt,Kind.Diag,Kind.None}

ncomponents(::Kind.None, dims, rank) = dims^rank
ncomponents(::Kind.Sym, dims, rank) = binomial(dims + rank - 1, rank)
ncomponents(::Kind.Alt, dims, rank) = binomial(dims, rank)
ncomponents(::Kind.Diag, dims, rank) = iszero(rank) ? 1 : dims

export isexpr, iscall
export isliteral, isindex, istensor, iscomp, islocs, isinds, isuniform
export head, args, operation, arguments
export value, argument, arity, isunary, isbinary
export components, locations, indices
export tensorrank, tensorname, tensorkind

export DTerm, ==ₛ
include("dterm.jl")

const ZERO = Literal(0)
const ONE  = Literal(1)

export 𝑖, 𝑗, 𝑘

const 𝑖 = Index(1)
const 𝑗 = Index(2)
const 𝑘 = Index(3)

export ⊡, ⊗, sym, asym, adj, gram, cogram
include("operators.jl")

include("utils.jl")

export AbstractRule, Passthrough, Chain, Prewalk, Postwalk, Fixpoint
include("rewriters.jl")

include("tensors.jl")

export @scalars, @vectors, @tensors, @uniform, @sym, @diag, @alt
include("macros.jl")

export AbstractDerivative, CentralDifference, StaggeredCentralDifference
include("derivatives.jl")

export DifferentialOperator
export Gradient, Divergence, Curl
include("calculus.jl")

export stencil_rule, lower
include("lowering.jl")

export <ₛ
include("isless_lex.jl")

export canonicalize, simplify
include("simplify.jl")

export Lifted
include("lift.jl")

export subs
include("subs.jl")

include("show.jl")

end # module Chmy
