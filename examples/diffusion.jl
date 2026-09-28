using Chmy
using CairoMakie

function diffusion(dims)
    # numerics
    nt = 100
    # fields
    @scalars u
    # operators
    divg = Divergence(StaggeredCentralDifference())
    grad = Gradient(StaggeredCentralDifference())
    # equation
    r_u = divg(grad(u))
    # lower and simplify
    locs = ntuple(_ -> 𝓈, length(dims))
    r_l = lower(r_u, locs) |> simplify
    # arrays
    u1 = rand(dims...)
    u2 = copy(u1)
    # bind data
    bnd = Binding(u[locs...] => u1)
    # compile
    r_c = compile(r_l, bnd; inbounds = true)
    return run(r_c, u1, u2, nt)
end

function run(r_c, u1, u2, nt)
    N = ndims(u1)
    f, l = extrema(CartesianIndices(u2))
    for it in 1:nt
        for I in (f + oneunit(f)):(l - oneunit(l))
            u2[I] = u1[I] + (0.4 / N) * r_c((u1,), Tuple(I))
        end
        bc!(u2)
        u1, u2 = u2, u1
    end
    return visme(u1)
end

function bc!(u::Array{<:Any, N}) where {N}
    f, l = extrema(CartesianIndices(u))
    for D in 1:N
        # unit offset along dimension D
        δ = CartesianIndex(ntuple(i -> i == D ? 1 : 0, Val(N)))
        n = l[D] - f[D]
        # lower face: copy from the first interior layer
        for I in f:(l - n * δ)
            u[I] = u[I + δ]
        end
        # upper face: copy from the last interior layer
        for I in (f + n * δ):l
            u[I] = u[I - δ]
        end
    end
    return
end

visme(u::Vector) = lines(u)
visme(u::Matrix) = heatmap(u; axis = (aspect = DataAspect(),))
visme(u::Array{<:Any, 3}) = heatmap(u[:, :, end ÷ 2]; axis = (aspect = DataAspect(),))

diffusion((64,))        # 1D
diffusion((64, 64))     # 2D
diffusion((64, 64, 64)) # 3D
