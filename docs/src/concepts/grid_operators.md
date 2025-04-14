# Grid Operators

Chmy.jl currently supports various finite difference operators for fields defined in Cartesian coordinates. The table below summarizes the most common usage of grid operators, with the grid `g::StructuredGrid` and index `I = @index(Global, Cartesian)` defined and `P = Field(backend, grid, location)` is some field defined on the grid `g`.

| Mathematical Formulation | Code |
|:-------|:------------|
| $\frac{\partial}{\partial x} P$ | ` ∂x(P, g, I)` |
| $\frac{\partial}{\partial y} P$ | ` ∂y(P, g, I)` |
| $\frac{\partial}{\partial z} P$ | ` ∂z(P, g, I)` |
| $\frac{\partial^2}{\partial x^2} P$ | ` ∂²x(P, g, I)` |
| $\frac{\partial^2}{\partial y^2} P$ | ` ∂²y(P, g, I)` |
| $\frac{\partial^2}{\partial z^2} P$ | ` ∂²z(P, g, I)` |
| $\nabla P$ | `divg(P, g, I)` |
| $\Delta P$ | `lapl(P, g, I)` |
| $\nabla \cdot χ \nabla P$ | `divg_grad(P, χ, g, I)` |

## Computing the Divergence of a Vector Field

To illustrate the usage of grid operators, we compute the divergence of a vector field $V$, the Laplacian of a scalar field $P$ and the divergence-gradient of a scalar field $C$ weighted by the coefficient $χ$ using the `divg`, `lapl` and `divg_grad` functions, respectively. We first allocate memory for required fields.

```julia
P  = Field(backend, grid, Center())
C  = Field(backend, grid, Center())
χ  = Field(backend, grid, Center()) # works also for Vertex()
∇V = Field(backend, grid, Center())
V  = VectorField(backend, grid)
# use set! to set up the initial vector field...
```

The kernel that computes the divergence needs to have the grid information passed as for other finite difference operators.

```julia
@kernel inbounds = true function update!(∇V, V, P, C, χ, g::StructuredGrid, O)
    I = @index(Global, Cartesian)
    I = I + O
    ∇V[I] = divg(V, g, I)
    P[I]  = lapl(P, g, I)
    C[I]  = divg_grad(C, χ, g, I)
end
```

The kernel can then be launched when required as we detailed in section [Kernels](./kernels.md).

```julia
launch(arch, grid, update! => (∇V, V, P, C, χ, grid))
```

## Masking

Masking allows selectively applying operations only where needed, allowing more flexible control over the grid operators and improving performance. Thus, by providing masked grid operators, we enable more flexible control over the domain on which the grid operators should be applied.

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
    I = @index(Global, Cartesian)
    I = I + O
    # with masks ω
    ε̇.xx[I] = ∂x(V.x, ω, g, I)
    ε̇.yy[I] = ∂y(V.y, ω, g, I)
    ε̇.xy[I] = 0.5 * (∂y(V.x, ω, g, I) + ∂x(V.y, ω, g, I))
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

Chmy.jl provides an interface `itp` which interpolates the field `f` from its location to the specified location `to` using the given interpolation rule `r`. The indices specify the position within the grid at location `to`:

```julia
itp(f, to, r, grid, I...)
```

Currently implemented interpolation rules are:
- `Linear()` which implements `rule(t, v0, v1) = v0 + t * (v1 - v0)`;
- `HarmonicLinear()` which implements `rule(t, v0, v1) = 1/(1/v0 + t * (1/v1 - 1/v0))`.

Both rules are exposed as convenience wrapper functions `lerp` and `hlerp`, using `Linear()` and `HarmonicLinear()` rules, respectively:

```julia
lerp(f, to, grid, I...)  # implements itp(f, to, Linear(), grid, I...)
hlerp(f, to, grid, I...) # implements itp(f, to, HarmonicLinear(), grid, I...)
```

In the following example, we use the linear interpolation wrapper `lerp` when interpolating nodal values of the density field `ρ`, defined on cell centres, i.e. having the location `(Center(), Center())` to `ρx` and `ρy`, defined on cell interfaces in the x- and y- direction, respectively.

```julia
# define density ρ on cell centres
ρ   = Field(backend, grid, Center())
ρ0  = 3.0; set!(ρ, ρ0)

# allocate memory for density on cell interfaces
ρx = Field(backend, grid, (Vertex(), Center()))
ρy = Field(backend, grid, (Center(), Vertex()))
```

The kernel `interpolate_ρ!` performs the actual interpolation and requires the grid information passed by `g`,

```julia
@kernel function interpolate_ρ!(ρ, ρx, ρy, g::StructuredGrid, O)
    I = @index(Global, Cartesian)
    I = I + O
    # interpolate from cell centres to cell interfaces
    ρx[I] = lerp(ρ, location(ρx), g, I)
    ρy[I] = lerp(ρ, location(ρy), g, I)
end
```

and can be launched with some launcher defined using `launch = Launcher(arch, grid)`:
```julia
launch(arch, grid, interpolate_ρ! => (ρ, ρx, ρy, grid))
```
