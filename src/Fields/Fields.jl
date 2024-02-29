module Fields

export AbstractField, Field, VectorField, TensorField
export ConstantField, ZeroField, OneField, ValueField
export FunctionField
export location, halo, interior, set!

export divg

using Chmy
using Chmy.Grids
using Chmy.Architectures
import Chmy.GridOperators
import Chmy: @add_cartesian

import Base.@propagate_inbounds
import LinearAlgebra

using KernelAbstractions
using Adapt

include("abstract_field.jl")
include("field.jl")
include("function_field.jl")
include("constant_field.jl")

end # module Fields
