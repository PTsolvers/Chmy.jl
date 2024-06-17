# Grid Operators

The gist of the finite difference relies on replacing derivatives by difference quotients anchored on structured grids. We currently support various finite difference operators for fields defined in Cartesian coordinates. The table below summarizes the most common usage of grid operators, with the grid `g::StructuredGrid` and index `I = @index(Global, Cartesian)` defined and `P = Field(backend, grid, location)` is some field defined on the grid `g`.

| Mathematical Formulation | Code |
|:-------|:------------|
| $\frac{\partial}{\partial x} P$ | ` ∂x(P, g, I)` |
| $\frac{\partial}{\partial y} P$ | ` ∂y(P, g, I)` |
| $\frac{\partial}{\partial z} P$ | ` ∂z(P, g, I)` |
| $\nabla P$ | ` divg(P, g, I)` |

## Computing the Divergence of a Vector Field

To illustrate the usage of grid operators, we compute the divergence of an vector field $V$ using the `divg` function. We first allocate memory for required fields.

```julia
V     = VectorField(backend, grid)
∇V    = Field(backend, grid, Center())
# use set! to set up the initial vector field...
```

The kernel that computes the divergence needs to have the grid information passed as for other finite difference operators.

```julia
@kernel inbounds = true function update_∇!(V, ∇V, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    ∇V[I...] = divg(V, g, I...)
end
```

The kernel can then be launched when required as we detailed in section [Kernels](./kernels.md).

```julia
launch(arch, grid, update_∇! => (V, ∇V, grid))
```



## Masking

Masking is particularly important when performing finite differences on GPUs, as it allows for efficient and accurate computations by selectively applying operations only where needed, allowing more flexible control over the grid operators and improving performance. Thus, by providing masked grid operators, we enable more flexible control over the domain on which the grid operators should be applied for advanced users.

In the following example, we first define a mask `ω` on the 2D `StructuredGrid`. Then we specify to **not** mask the center area of all Vx, Vy nodes (accessible through `ω.vc`, `ω.cv`) on the staggered grid.

```julia
# define the mask
ω = FieldMask2D(arch, grid) # with backend and grid geometry defined

# define the initial inclusion
r = 2.0
init_inclusion = (x,y) -> ifelse(x^2 + y^2 < r^2, 1.0, 0.0)

# mask all other entries other than the initial inclusion 
set!(ω.vc, grid, init_inclusion)
set!(ω.cv, grid, init_inclusion)
```

We can then pass the mask to other grid operators when applying them within the kernel. When computing masked derivatives, a mask being the subtype of `AbstractMask` is premultiplied at the corresponding grid location for each operand:

```julia
@kernel function update_strain_rate!(ε̇, V, ω::AbstractMask, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    # with masks ω
    ε̇.xx[I...] = ∂x(V.x, ω, g, I...)
    ε̇.yy[I...] = ∂y(V.y, ω, g, I...)
    ε̇.xy[I...] = 0.5 * (∂y(V.x, ω, g, I...) + ∂x(V.y, ω, g, I...))
end
```

The kernel can be launched as follows, with some launcher defined using `launch = Launcher(arch, grid)`:

```julia
# define fields
ε̇ = TensorField(backend, grid)
V = VectorField(backend, grid)

# launch kernel
launch(arch, grid, update_strain_rate! => (ε̇, V, ω, grid))
```


## Interpolation

Interpolating physical parameters such as permeability and density between various types of nodes is frequently necessary on a staggered grid. Chmy.jl provides conveninent functions for performing interpolations of field values between different types of nodes.

In the following example, we specify to use the linear interpolation rule `lerp` when interpolating nodal values of the density field `ρ`, defined on pressure nodes (with location `(Center(), Center())`) to `ρvx` and `ρvy`, defined on Vx and Vy nodes respectively.

```julia
# define density ρ on pressure nodes
ρ   = Field(backend, grid, Center())
ρ0  = 3.0; set!(ρ, ρ0)

# allocate memory for density on Vx, Vy nodes
ρvx = Field(backend, grid, (Vertex(), Center()))
ρvy = Field(backend, grid, (Center(), Vertex()))
```

The kernel `interpolate_ρ!` performs the actual interpolation of nodal values and requires the grid information passed by `g`.

```julia
@kernel function interpolate_ρ!(ρ, ρvx, ρvy, g::StructuredGrid, O)
    I = @index(Global,  NTuple)
    I = I + O
    # Interpolate from pressure nodes to Vx, Vy nodes
    lerp(ρvx, location(ρ), g, I...)
    lerp(ρvy, location(ρ), g, I...)
end
```