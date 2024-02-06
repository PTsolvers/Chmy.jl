module BoundaryConditions

export FieldBoundaryCondition, Dirichlet, Neumann, bc!
export BoundaryFunction

export AbstractBatch, FieldBatch, EmptyBatch, BatchSet, batch

using Chmy
using Chmy.Grids
using Chmy.Fields
using Chmy.Architectures

import Chmy: @add_cartesian

using KernelAbstractions

import Base.@propagate_inbounds

const SDA = SingleDeviceArchitecture
const SG  = StructuredGrid

include("field_boundary_condition.jl")
include("batch.jl")
include("boundary_function.jl")

end
