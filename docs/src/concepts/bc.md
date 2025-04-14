# Boundary Conditions

Using [Chmy.jl](https://github.com/PTsolvers/Chmy.jl), we aim to study partial differential equations (PDEs) arising from physical or engineering problems. Additional initial and/or boundary conditions are necessary for the model problem to be well-posed, ensuring the existence and uniqueness of a stable solution. The strategy used in this package to apply boundary conditions is based on ghost points.

We provide a small overview for boundary conditions that one often encounters. In the following, we consider the unknown function $u : \Omega \mapsto \mathbb{R}$ defined on some bounded computational domain $\Omega \subset \mathbb{R}^d$ in a $d$-dimensional space. With the domain boundary denoted by $\partial \Omega$, we have some function $g : \partial \Omega \mapsto \mathbb{R}$ prescribed on the boundary.

| Type    | Form | Example |
|:------------|:------------|:---------|
| Dirichlet | $u = g$ on $\partial \Omega$ | In fluid dynamics, the no-slip condition for viscous fluids states that at a solid boundary the fluid has zero velocity relative to the boundary. |
| Neumann | $\partial_{\boldsymbol{n}} u = g$ on $\partial \Omega$, where $\boldsymbol{n}$ is the outer normal vector to $\Omega$ | It specifies the values in which the derivative of a solution is applied within the boundary of the domain. An application in thermodynamics is a prescribed heat flux through the boundary |
| Robin  |  $u + \alpha \partial_\nu u = g$ on $\partial \Omega$, where $\alpha \in \mathbb{R}$.  | Also called impedance boundary conditions from their application in electromagnetic problems |

## Applying Boundary Conditions with `bc!()`

In the following, we describe the syntax in [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) for launching kernels that impose boundary conditions on some `field` that is well-defined on a `grid` with backend specified through `arch`.

For Dirichlet and Neumann boundary conditions, they are referred to as homogeneous if $g = 0$, otherwise they are non-homogeneous if $g = v$ holds, for some $v\in \mathbb{R}$.

|     | Homogeneous | Non-homogeneous |
|:------------|:------------|:------------|
| Dirichlet on $\partial \Omega$ | `bc!(arch, grid, field => Dirichlet())` | `bc!(arch, grid, field => Dirichlet(v))` |
| Neumann on $\partial \Omega$ | `bc!(arch, grid, field => Neumann())` | `bc!(arch, grid, field => Neumann(v))` |

Note that the syntax shown in the table above is a **fused expression** of both _specifying_ and _applying_ the boundary conditions.

!!! warning "$\partial \Omega$ Refers to the Entire Domain Boundary!"
    By specifying `field` to a single boundary condition, we impose the boundary condition on the entire domain boundary by default. See the section for "Mixed Boundary Conditions" below for specifying different BC on different parts of the domain boundary.

Alternatively, one could also define the boundary conditions beforehand using `batch()` provided the `grid` information as well as the `field` variable. This way the boundary condition to be prescribed is **precomputed**.

```julia
# pre-compute batch
bt = batch(grid, field => Neumann()) # specify Neumann BC for the variable `field`
bc!(arch, grid, bt)                  # apply the boundary condition
```

In the script [batcher.jl](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/batcher.jl), we provide a MWE using both **fused** and **precomputed** expressions for BC update.

## Specifying BC within a `launch`

When using `launch` to specify the execution of a kernel (more see section [Kernels](./kernels.md)), one can pass the specified boundary condition(s) as an optional parameter using `batch`, provided the grid information of the discretized space. This way we can gain efficiency from making good use of already cached values.

In the 2D diffusion example as introduced in the tutorial ["Getting Started with Chmy.jl"](../getting_started/introduction.md), we need to update the temperature field `C` at k-th iteration using the values of heat flux `q` and physical time step size `Δt` from (k-1)-th iteration. When launching the kernel `update_C!` with `launch`, we simultaneously launch the kernel for the BC update using:

```julia
launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
```

### Mixed Boundary Conditions

In the code example above, by specifying boundary conditions using syntax such as `field => Neumann()`, we essentially launch a kernel that impose the Neumann boundary condition on the entire domain boundary $\partial \Omega$. More often, one may be interested in prescribing different boundary conditions on different parts of $\partial \Omega$.

The following figure showcases a 2D square domain $\Omega$ with different boundary conditions applied on each side:

- The top boundary (red) is a Dirichlet boundary condition where $u = a$.
- The bottom boundary (blue) is also a Dirichlet boundary condition where $u = b$.
- The left and right boundaries (green) are Neumann boundary conditions where $\frac{\partial u}{\partial y} = 0$.

```@raw html
<img src="https://raw.githubusercontent.com/PTsolvers/Chmy.jl/main/docs/src/assets/mixed_bc_example.png" width="60%"/>
```

To launch a kernel that satisfies these boundary conditions in Chmy.jl, you can use the following code:

```julia
bc!(arch, grid, field => (x = Neumann(), y = (Dirichlet(b), Dirichlet(a))))
```

### More Complex Boundary Conditions

In some cases, one may need to apply more complex boundary conditions, with a dependency in space or using conditions for instance. This can be done using `BoundaryFunction`. We provide in this subsection a usage example of this feature.

Taking the previous example, let's define two subdomains at the bottom boundary, with two different Dirichlet boundary conditions.

In this case, the boundary will have the value $u = b$ for $x < 0$, and $u = c$ for $x > 0$.

First, let's call Chmy and KernelAbstractions, and initialize our variables:

```julia
using Chmy
using KernelAbstractions

# load backend
backend = CPU()
arch = Arch(backend)

# define grid
grid   = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(100,100))
launch = Launcher(arch, grid)

# empty field for the example
field = Field(backend, grid, Center())
```

We can then define `subdomain`, which would be the function applying the boundary condition at the bottom of the model on our variable `field`.

```julia
function subdomain(x, b, c, half_domain)
    if x < half_domain
        return b
    else
        return c
    end
end
```

The function `subdomain` will be the input of `BoundaryFunction`. `BoundaryFunction` can then be passed to the `bc!()` function, which will apply the boundary condition on the field.

```julia
# parameters for the boundaries
# values of the Dirichlet BC
a = 0
b = 10
c = 20
half_domain = 0

# define BoundaryFunction
boundary_x = BoundaryFunction(subdomain; parameters = (b, c, half_domain))

# apply the boundary condition
bc = (field  => (x = (Dirichlet(boundary_x), Dirichlet(a)), y=Neumann()))
bc!(arch, grid, bc)
```

The `parameters` argument is used to pass the parameters of the function `subdomain`.
