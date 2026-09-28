module Chmy

using KernelAbstractions

using Moshi: Data.@data, Data.isa_variant, Match.@match

using PrettyTables: pretty_table, TextTableFormat
import Adapt

import LinearAlgebra: ⋅, ×, tr, det, diag, transpose
import Base: broadcasted

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# re-export from LinearAlgebra
export ⋅, ×, tr, det, diag, transpose

export DTerm, Literal, Index, Operator, Fun
include("types.jl")

export isexpr, iscall
export isliteral, isindex, istensor, iscomp, islocs, isinds, isuniform
export head, children, operation, arguments
export value, argument, arity, isunary, isbinary
export components, locations, indices
export tensorrank, tensorname, tensorkind
export ==ₛ, !=ₛ, <ₛ, isnegof, isstaticzero, isstaticone, isunaryminus
export 𝒪, ℐ
include("dterm.jl")

const ZERO = Literal(0)
const ONE = Literal(1)

export 𝑖, 𝑗, 𝑘
const 𝑖 = Index(1)
const 𝑗 = Index(2)
const 𝑘 = Index(3)

export 𝓅, 𝓈
const 𝓅 = Point()
const 𝓈 = Segment()

export ⊡, ⊗, sym, asym, cof, adj, gram, cogram
include("operators.jl")

include("utils.jl")

export AbstractRule, Passthrough, Chain, Prewalk, Postwalk, Fixpoint
include("rewriters.jl")

include("tensors.jl")

export @scalars, @vectors, @tensors, @uniform, @sym, @diag, @alt
include("macros.jl")

export AbstractDifference, Difference, StaggeredDifference
export CentralDifference, ForwardDifference, BackwardDifference
export StaggeredCentralDifference, StaggeredForwardDifference, StaggeredBackwardDifference
export stencil_nodes, stencil_weights
include("differences.jl")

export DifferentialOperator
export Gradient, Divergence, Curl
include("calculus.jl")

export stencil_rule, lower
include("lowering.jl")

export canonicalize, simplify
include("simplify.jl")

export Lifted
include("lift.jl")

export subs
include("subs.jl")

export Binding
export findkey, push, pairstuple
include("binding.jl")

export compile
include("compile.jl")

include("show.jl")

end # module Chmy
