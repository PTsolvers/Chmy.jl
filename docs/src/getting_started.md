# Getting Started with Chmy.jl

[Chmy.jl](https://github.com/PTsolvers/Chmy.jl) is powered by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and it is a backend-agnostic toolkit for finite difference computations on multi-dimensional computational staggered grids. In this introductory tutorial, we will showcase the essence of Chmy.jl in which we resolve a simple explicit 2D diffusion problem. The full code of the tutorial material is available under [diffusion_2d.jl](https://github.com/PTsolvers/Chmy.jl/blob/main/examples/diffusion_2d.jl).

## Basic Diffusion

The diffusion equation is a second order parabolic PDE, here for a multivariable function $C(x,t)$ that represents the temperature field showing derivatives in both temporal $\partial t$ and spatial $\partial x$ dimensions, where $\chi$ is the diffusion coefficient. In 2D we have the following formulation for the diffusion process:

```math
\begin{equation}
\frac{\partial C}{\partial t} = \chi \left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right).
\end{equation}
```

Using the Fourier's law of heat conduction, which relates the heat flux $q$, $(W/m^2)$ to the temperature gradient $\frac{\partial C}{\partial x_i}$ $(K/m)$, we can rewrite equation `(1)` as a system of two PDEs, consisting of equations `(2)` and `(3)`.

```math
\begin{equation}
q := -\chi \nabla C,
\end{equation}
```
```math
\begin{equation}
\frac{\partial C}{\partial t} = - \nabla \cdot q.
\end{equation}
```

### Boundary Condition

Generally, partial differential equations (PDEs) require initial or [boundary conditions](./concepts/bc.md) to ensure a unique and stable solution. For the temperature field `C`, a Neumann boundary condition is given by:

```math
\begin{equation}
\frac{\partial C}{\partial n} = g(x, t)
\end{equation}
```
where $\frac{\partial C}{\partial n}$ is the derivative of `C` normal to the boundary, and $g(x, t)$ is a given function. In this tutorial example, we consider a homogeneous Neumann boundary condition, $g(x, t) = 0$, which implies that there is no flux across the boundary.


## Using Chmy.jl for Backend Portable Implementation

As the first step, we need to load the main module and any necessary submodules of [Chmy.jl](https://github.com/PTsolvers/Chmy.jl). Moreover, we use [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) for writting backend-agnostic kernels that are compatible with Chmy.jl. One may also load additional modules for results analysis or plotting, etc...

```julia
using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.Fields, Chmy.BoundaryConditions, Chmy.GridOperators, Chmy.KernelLaunch
using KernelAbstractions # for CPU or various GPU backend
using Printf, CairoMakie # for I/O and plotting
# using AMDGPU
```

In this introductory tutorial, we will use the CPU backend for simplicity:

```julia
arch = Arch(CPU())
```

if a different backend is desired, one needs to load the relevant package accordingly. For example, if AMD GPUs are available, one can comment out `using AMDGPU` and make sure to use `arch = Arch(ROCBackend())` when selecting architecture. For more about specifying if executing on a single-device or multi-device architecture, see the dedicated documentation section for [Architectures](./concepts/architectures.md)

## Writing & Launch Compute Kernels

We want to resolve the system of equations `(2)` & `(3)` numerically, for this we will use the explicit forward [Euler method](https://en.wikipedia.org/wiki/Euler_method) for temporal discretization and [finite-differences](https://en.wikipedia.org/wiki/Finite_difference) for spatial discretization. Accordingly, the kernels for performing the arithmetic operations for each time step can be defined as follows:

```julia
@kernel inbounds = true function compute_q!(q, C, χ, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    q.x[I...] = -χ * ∂x(C, g, I...)
    q.y[I...] = -χ * ∂y(C, g, I...)
end
```

```julia
@kernel inbounds = true function update_C!(C, q, Δt, g::StructuredGrid, O)
    I = @index(Global, NTuple)
    I = I + O
    C[I...] -= Δt * divg(q, g, I...)
end
```

## Model Setup

The diffusion model that we resolve for should contain the following model setup

```julia
# geometry
grid   = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(128, 128))
launch = Launcher(arch, grid; outer_width=(16, 8))
# physics
χ = 1.0
# numerics
Δt = minimum(spacing(grid))^2 / χ / ndims(grid) / 2.1
```

In the problem only two physical fields, the temperature field `C` and the heat flux `q` are evolving with time. For better accuracy of the solution, we opted for defining physical properties on different nodes on the staggered grid (more see [Grids](./concepts/grids.md)).

```julia
# allocate fields
C = Field(backend, grid, Center())
q = VectorField(backend, grid)
```

We randomly initialized the entries of `C` field and finished the initial model setup. One can refer to the section [Fields](./concepts/fields.md) for setting up more complex initial conditions.

```julia
# initial conditions
set!(C, grid, (_, _) -> rand())
bc!(arch, grid, C => Neumann(); exchange=C)
```

```@raw html
<div style="text-align: center;">
    <img src="../assets/field_set_ic_random.png" width="50%"/>
</div>
```

## Solving Time-dependent Problem

We are resolving a time-dependent problem, so we advance our solutions using a time loop, specifying the number of iterations we desire to perform. The true action that takes place within the time loop is the variable update that is performed by the compute kernels `compute_q!` and `update_C!`, accompanied by the imposing the Neumann boundary condition on the `C` field.

```julia
# action
nt = 100
for it in 1:nt
    @printf("it = %d/%d \n", it, nt)
    launch(arch, grid, compute_q! => (q, C, χ, grid))
    launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann(); exchange=C))
end
KernelAbstractions.synchronize(backend)
```

If you follow up the tutorial with the correct implementation, you should see something like this, here the final result at `it = 100` for the temperature field `C` is plotted.

```@raw html
<div style="text-align: center;">
    <img src="../assets/diffusion_2d_it_100.png" width="50%"/>
</div>
```