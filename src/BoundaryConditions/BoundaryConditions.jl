module BoundaryConditions

export FieldBoundaryCondition, Dirichlet, Neumann, bc!, batch
export BoundaryFunction

using Chmy
using Chmy.Grids
using Chmy.Fields

import Chmy: @add_cartesian

using KernelAbstractions

import Base.@propagate_inbounds

include("field_boundary_condition.jl")
include("boundary_function.jl")

end
