struct Gradient{N,P}
    ∂::P
end

Gradient(∂::Vararg{AbstractPartialDerivative,N}) where {N} = Gradient{N,typeof(∂)}(∂)
function Gradient{N}(∂::STerm) where {N}
    ∂s = ntuple(i -> LiftedPartialDerivative{i}(∂), Val(N))
    Gradient{N,typeof(∂s)}(∂s)
end

function (g::Gradient{N})(f::STerm) where {N}
    return Vec{N}(ntuple(i -> g.∂[i](f), Val(N))...)
end

struct Divergence{N,P}
    ∂::P
end

Divergence(∂::Vararg{AbstractPartialDerivative,N}) where {N} = Divergence{N,typeof(∂)}(∂)
function Divergence{N}(∂::STerm) where {N}
    ∂s = ntuple(i -> LiftedPartialDerivative{i}(∂), Val(N))
    Divergence{N,typeof(∂s)}(∂s)
end

function (d::Divergence{N})(v::Vec{N}) where {N}
    return +(ntuple(i -> d.∂[i](v[i]), Val(N))...)
end
