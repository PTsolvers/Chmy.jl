# Getting Started with Chmy.jl

[Chmy.jl](https://github.com/PTsolvers/Chmy.jl) is powered by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and it is a backend-agnostic toolkit for finite difference computations on multi-dimensional computational staggered grids. In this introductory tutorial, we will show the essence of Chmy.jl in which we resolved a simple 2D diffusion problem.


## Basic Diffusion

The diffusion equation is a second order parabolic PDE, here for a multivariable function $C(x,t)$ that represents the temperature field showing derivatives in both temporal $\partial t$ and spatial $\partial x$ dimensions, where $D$ is the diffusion coefficient.

## TODO: this is yet 1D! Change it to general case

```math
\begin{equation}
\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2},
\end{equation}
```

Using the Fourier's law of heat conduction, which relates the heat flux $q$, $(W/m^2)$ to the temperature gradient $\frac{\partial C}{\partial x}$ $(K/m)$, we can rewrite equation (1) as a system of two PDEs, consisting of equations (2) and (3).

```math
\begin{equation}
q := -D \frac{\partial C}{\partial x},
\end{equation}
```
```math
\begin{equation}
\frac{\partial C}{\partial t} = -\frac{\partial q}{\partial x}.
\end{equation}
```

!!! info "Derivation of Equation (3)"
    ```math
    \begin{equation}
    \frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2} = D \frac{\partial}{\partial x}(\frac{\partial}{\partial x}C) = - \frac{\partial q}{\partial x}
    \end{equation}
    ```


## Finite-difference Discretization

TODO:

```math
\begin{equation}
\frac{\partial C}{\partial x} \approx \frac{C_{x+\Delta x} - C_x}{\Delta x}.
\end{equation}
```


## Writing & Launch Compute Kernels

Accordingly, the kernels for performing the arithmetic operations for each time step can be defined as follows:

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






## Results