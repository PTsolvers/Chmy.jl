abstract type AbstractDerivative <: Operator end

Base.getindex(d::AbstractDerivative, i::Integer) = Lifted(d, i)

struct CentralDifference <: AbstractDerivative end

stencil_rule(::CentralDifference, f::DTerm, l::Location, i::DTerm) = 1 // 2 * (f[l][i+1] - f[l][i-1])
stencil_rule(::CentralDifference, f::DTerm, i::DTerm) = 1 // 2 * (f[i+1] - f[i-1])

struct StaggeredCentralDifference <: AbstractDerivative end

stencil_rule(::StaggeredCentralDifference, f::DTerm, ::typeof(𝓅), i::DTerm) = f[𝓈][i] - f[𝓈][i-1]
stencil_rule(::StaggeredCentralDifference, f::DTerm, ::typeof(𝓈), i::DTerm) = f[𝓅][i+1] - f[𝓅][i]
