### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 34237eee-ca2e-11f0-9a35-39b494c55d7b
# ╠═╡ show_logs = false
begin
    import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["Revise", "PlutoLinks", "PlutoUI", "CairoMakie"])
	Pkg.add(url="https://github.com/PTsolvers/Chmy.jl", rev="iu/v0.2")
	using PlutoLinks, PlutoUI, Revise, Chmy
	import CairoMakie: Figure, Axis, Colorbar, DataAspect, heatmap!
end


# ╔═╡ e7168060-8cc7-4965-8224-3f3c844289a9
using LinearAlgebra

# ╔═╡ 1e0cd8df-528c-4868-985a-393a10e539cd
TableOfContents()

# ╔═╡ a80768e2-0f6b-4d7f-8863-3528fa092291
md"""
# Intro to Chmy.jl

Hi, thank you for your interest in Chmy.jl, the Julia library for implementing finite-difference operators on regular rectangular grids. In this notebook, we introduce main concepts of Chmy.jl and implement a small example: steady diffusion in 2D.

!!! warning
	Chmy.jl is a work in progress, many features are missing, documentation and tests are non-existent. This is just a demonstration of capabilities of the package. I'm happy to take feedback and feature requests.

## Philosophy

It all starts with one simple question:
"""

# ╔═╡ 2eec34ba-d46a-4687-bb80-3194e1179aef
details("How to derivative?", md"![Thinking kitty](https://i.imgur.com/ikl0uW4.png)")

# ╔═╡ c06ba65e-5e63-418e-82de-08801ff0f3c8
md"""
Let's make an array:
"""

# ╔═╡ 5645e4ae-20e6-4dc4-9113-2aa4e2fc998b
A = rand(4)

# ╔═╡ b2597873-92cb-41bb-8a15-6db882ade747
md"""
Obvious solution is just to make a function for a derivative at index `i`:
"""

# ╔═╡ 8b1e7194-953a-4a1b-9ca8-9e486bcb612f
let d(x, i) = x[i + 1] - x[i]
	d(A, 1) ≈ A[2] - A[1]
end

# ╔═╡ 22697e58-064c-479c-b5fe-7f8d1cdf4d1d
md"""
Ok, problem solved!

Well, no. What about second derivative $\partial^2(A)/\partial x^2$? Or what if we want to compute $\partial(A^2)/\partial x$? In other words, we want our derivative function to be **composable**.

Next attempt: we can pass a callback:
"""

# ╔═╡ 53f06a87-6a56-4eef-9e00-5628b5825ac7
let d(x, i) = x[i + 1] - x[i]
	d(f, x, i) = f(x, i + 1) - f(x, i)
	@assert d(d, A, 1) ≈ A[1] - 2A[2] + A[3]
	@assert d((x, i) -> d(d, x, i), A, 1) ≈ -A[1] + 3A[2] - 3A[3] + A[4]
end

# ╔═╡ d4b972a4-de54-4a8c-864e-ed231acb7704
md"""
A bit better, we have now one layer of composition. This is how derivative operators are implemented in [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/blob/6b1c59bad589311774d1e9aeb8fb1fbbb4e3f126/src/Operators/difference_operators.jl#L7-L27). But what if we want to go **deeper**? Then the code becomes ugly.

---

Here's an idea: consider fields not as arrays, but as functions mapping indices to values. Therefore, functions operating on fields should return... *Other functions!* At the bottom level we still want to access data stored in arrays, or compute the data on the fly.
"""

# ╔═╡ 597ae26a-0d2f-4941-8c92-039368081786
get(f) = i -> f[i]

# ╔═╡ c29ac9d1-9c1c-4337-bc46-0f93b032186e
md"""
Derivative of a field is also a field, which means it also returns a function:
"""

# ╔═╡ 5bc99c52-684d-4c00-b67c-e581965fc9a8
d(f) = i -> f(i + 1) - f(i)

# ╔═╡ 208d1fe5-2553-4a69-b1c1-e44e6f33be3d
d(A)

# ╔═╡ 458df925-0d1f-4b0b-9e6d-240b3f7ecc6b
get(A)(2) ≈ A[2]

# ╔═╡ 40de0823-2af2-4207-851b-a437a8bc7820
d(get(A))(1) ≈ A[2] - A[1]

# ╔═╡ 5ca01538-b4ce-424f-8fdf-729e3a377828
d(d(get(A)))(1) ≈ A[1] - 2A[2] + A[3]

# ╔═╡ 64ea4639-90fb-4c25-942e-e461e2edbad7
d(d(d(get(A))))(1) ≈ -A[1] + 3A[2] - 3A[3] + A[4]

# ╔═╡ 6a175155-a76f-476b-b6af-7c9f30499c00
md"""
Cool, infinite composability! And now calling `d(f)` doesn't actually perform any computations, but is a *lazy object* representing the computation. Only when applied to an index, it actually performs the computation.

This purely functional approach is great, but we are limited by the Julia's anonymous function syntax. Many things are difficult to implement: for example, what if we want to compute the sparsity pattern of a finite-difference operator? Also, anonymous functions have weird compiler-generated names that are hard to read.

Chmy.jl is created as a package for representing the finite-difference, or, more generally, stencil operations as lazy objects, with a functional approach in mind, but with more features.

---
"""

# ╔═╡ 8faa14c9-6bdf-411c-829c-c5e3645b4311
md"""
## Chmy DSL

Chmy.jl v0.2 features a domain-specific language (DSL) for representing stencil operations as symbolic objects.

!!! hmmm
    What is the difference between Chmy DSL and [Symbolics.jl](https://symbolics.juliasymbolics.org/stable/)? Symbolic expressions in Chmy are *completely static*, and all symbolic manipulations happen *at compilation time*. This has drawbacks, such as increased compilation time and more potential for type instabilities. On the other hand, constructing and compiling Chmy expressions **has no runtime cost**, so it can be done in hot loops. 
	

The main object in Julia DSL is a tagged variable:
"""

# ╔═╡ 96b8658f-6d51-420f-86aa-9c017cecbb97
f = Tag(:f)

# ╔═╡ 12c7b7e4-fffc-4143-a790-d52e275b2a3b
md"""
With tags, we can construct **very** complex expressions:
"""

# ╔═╡ 5a1c972e-20cd-45d7-a77b-584724d21c9f
expr = f + 1

# ╔═╡ 03edaa61-3855-4ca2-a246-6cef14b56f02
md"""
Chmy expressions are lazy, and are represented as a special type `SExpr`, which mimics standard Julia expressions, but is static:
"""

# ╔═╡ 098fa57e-37ff-474b-bac4-953389b44470
head(expr), children(expr)

# ╔═╡ 8ce4d6c3-b234-40eb-986b-f5b85dc36c32
md"""
Next important object is a static index:
"""

# ╔═╡ 79bbbd1a-1b83-44f1-aafa-4e1846d5c4a6
i, j, k = SIndex(1), SIndex(2), SIndex(3)

# ╔═╡ b78692db-e0b0-45b0-b830-ed32b50e5dd1
md"""
We can attach one or more indices to a Chmy expression by using array-like indexing:
"""

# ╔═╡ 22ff2325-67de-4e2a-8d4e-c777729d37f8
stencil_expr = expr[i]

# ╔═╡ a044abce-5208-4d9e-9179-00ae808257d4
md"""
### Lowering stencil operations

When we construct a Chmy expression, and attach indices to it, we need to propagate the indices recursively until only the tagged variables are indexed:
"""

# ╔═╡ 8a9d6e44-7ed3-43f0-9ad2-cf190d4df008
lowered_expr = lower_stencil(stencil_expr)

# ╔═╡ 3a909d0d-0b7a-431c-822f-f476b0c25ddb
md"""
The lowering is a compile-time operation:
"""

# ╔═╡ 65083f09-b49d-4148-9812-e51473e0bc64
@code_llvm lower_stencil(stencil_expr)

# ╔═╡ 71c76795-4c5c-4c30-9e88-16461a360072
md"""
### Binding the data to Chmy expressions

Ok, we can construct expressions, but how do we do actual computations? We need to **bind** the actual data to the computations.
"""

# ╔═╡ 41194bd1-aaa4-4a34-8196-1607508f019d
g = Tag(:g)

# ╔═╡ 1a153714-8de0-4bbc-ab9c-aed49477dffa
binding = Binding(f => rand(2), g => 1)

# ╔═╡ a7c625b4-7994-4b2b-80c4-440e1884ca44
md"""
The binding is an immutable object that can be sent to the computational kernel. To update the binding, we need to construct a new one:
"""

# ╔═╡ 8b3cf604-4b9b-4a51-8f6e-38125a46909a
push(binding, g => rand(2))

# ╔═╡ 5fbfe498-1d93-4bfa-a2d7-4c81ee36165e
md"""
We can use the binding as a dictionary to access data bound to an expression:
"""

# ╔═╡ bcd7be0b-9a5d-4540-a29b-9aa13eafde6d
binding[f]

# ╔═╡ d4cbfb3e-fdef-43d2-95f4-f4653121d9e1
md"""
### Compiling and executing Chmy expressions

Let's see how the expression is actually compiled to a standard Julia code:
"""

# ╔═╡ 3bd61dc2-0215-4cbd-abe1-183edd12b4e7
btypes = binding_types(binding)

# ╔═╡ 2b75fa91-2d0a-4077-9e34-e927152c86ec
to_expr(lowered_expr, btypes)

# ╔═╡ fced73c1-c15e-4f59-a668-edb6b4297e40
md"""
To compute the expression, use the function `compute(expr, binding, inds...)`:
"""

# ╔═╡ d2db2103-d2f1-4c92-9ae8-7d1e096b27ae
compute_inbounds(expr, bnd, inds...) = @inbounds compute(expr, bnd, inds...)

# ╔═╡ e4245dae-aeba-448e-bdc3-2091e4cea752
@code_llvm compute_inbounds(lowered_expr, binding, 1)

# ╔═╡ 07b8416b-73f9-4f02-86e3-bb09cb499fbc
md"""
## Grid spaces

In many applications, it is convenient to explicitly associate the discrete field with a particular topological element on a grid, such as a node, edge, face, or a cell.

In Chmy, the topological element is defined as a direct product of 0D and 1D spaces, called `Point` and `Segment`, respectively:
"""

# ╔═╡ 3ff47ee9-f16a-4e20-bab6-50cbd9f4e09d
p, s = Point(), Segment()

# ╔═╡ 5857337c-05d6-46e7-aac2-1b3c9fa07956
md"""
We can attach spaces in the same way as indices:
"""

# ╔═╡ 6a38f026-a732-4ebb-a4de-fbf6ad949e34
loc_expr = expr[p, s][i, j]

# ╔═╡ 9118a3fc-317d-408e-90d7-3e4c176f0452
lowered_loc_expr = lower_stencil(loc_expr)

# ╔═╡ d36d8b52-ccf8-4be8-bae6-0d30d4ffc436
binding_loc = Binding(f[p, s] => rand(1, 1))

# ╔═╡ 7e2836b0-342f-42c2-ac17-1a01f564b51b
compute(lowered_loc_expr, binding_loc, 1, 1)

# ╔═╡ 381c451c-12b4-4bf6-8e21-fdfffb4bd5bc
md"""
## Derivatives
"""

# ╔═╡ 7d8d5820-e437-4d97-98f7-9372f199857b
md"""
Derivatives in Chmy are special type of symbolic terms. These terms have rules defined on them to expand the derivative into primitive stencil operations. The derivative types are subtypes of a `AbstractDerivative` type:
"""

# ╔═╡ 07b21203-f561-4ea5-821d-49db291c5a0c
D = StaggeredCentralDifference()

# ╔═╡ e592b9dc-920a-4b22-9266-3b8ad94b1279
D(f)

# ╔═╡ 80b5a14b-f690-4730-bc07-7f0782934b07
lower_stencil(D(f)[s][i])

# ╔═╡ 955d22ed-938a-4a2d-9a67-35aa8a8b82b0
md"""
Partial derivatives are subtypes of the `AbstractPartialDerivative{I}` type, and they include the information of the direction `I` of derivation.

However, in most applications, we want to define the derivative rule in 1D, and then "lift" the derivation to nD. For that, Chmy provides the type `PartialDerivative` which wraps a regular 1D derivative and lifts the derivation to nD:
"""

# ╔═╡ 394557dd-6e71-4ba5-80ab-320fad844edf
∂ = PartialDerivative(D)

# ╔═╡ ee075fbe-13b3-4255-993b-d93982836e83
∂(f, 1)

# ╔═╡ c7a4aa0c-2377-429b-a4be-7169419b031e
lower_stencil(∂(f, 2)[p, s, s][i, j, k])

# ╔═╡ 069e01eb-cb9a-4138-af9a-0b6c69f81774
md"""
### Implementing new derivative rules
"""

# ╔═╡ 1628ac5c-7a32-4297-86cc-c11d4ac7881f
md"""
To implement your custom derivative rules in 1D (e.g. high-order WENO, upwind etc.), create a subtype of `AbstractDerivative`:
"""

# ╔═╡ ccecaea0-250e-4034-bc57-088368aa6768
begin
struct ForwardDifference <: AbstractDerivative end

function Chmy.stencil_rule(::ForwardDifference,
						   args::Tuple{STerm},
						   inds::Tuple{STerm})
    f, i = only(args), only(inds)
    return 0.5 * (f[i+1] - f[i])
end

function Chmy.stencil_rule(::ForwardDifference,
						   args::Tuple{STerm},
						   loc::Tuple{Space},
						   inds::Tuple{STerm})
    f, l, i = only(args), only(loc), only(inds)
    return 0.5 * (f[l][i+1] - f[l][i])
end
end

# ╔═╡ cfc4ecd1-246b-4772-b2d4-ba11b66574de
𝒟ᶠ = ForwardDifference()

# ╔═╡ d6451119-0a1f-4420-aaf8-f44372f11ae7
𝒟ᶠ(f)

# ╔═╡ 7f11e3e0-f2ac-44d1-863e-af867344ea96
lower_stencil(𝒟ᶠ(f)[i])

# ╔═╡ 6f887d81-78c1-4d87-bc2c-c64c7e7dce2f
md"""
In Chmy.jl, it is recommended to define derivatives in a reference Cartesian coordinate system with unit spacing. To map the scaled, shifted, or even curvilinear coordinate systems to the reference, use vector and tensor calculus.
"""

# ╔═╡ 2e151b7e-63b4-4b2b-a09b-7b9e27a1494f
md"""
## Rewriting expressions

The true power of Chmy.jl expressions is in symbolic manipulation. Chmy.jl provides basic building blocks for combining rewriting rules. All high-level operations such as lowering and lifting are implemented using these blocks.

Rewriting rules in Chmy.jl are subtypes of `AbstractRule`. Let's implement a custom rule for symbolic substitution:
"""

# ╔═╡ 56e19c0b-c592-4528-8e23-0ed3b9ca0e0b
begin
struct SubsRule{Lhs,Rhs} <: AbstractRule
	lhs::Lhs
	rhs::Rhs
end

SubsRule(kv::Pair) = SubsRule(kv.first, kv.second)

(rule::SubsRule{Lhs})(term::Lhs) where {Lhs <: STerm} = rule.rhs

Base.show(io::IO, rule::SubsRule) = print(io, rule.lhs, " => ", rule.rhs)
	
function Base.show(io::IO, ::MIME"text/plain", rule::SubsRule)
	print(io, "SubsRule:\n ")
	show(io, rule)
end
end

# ╔═╡ a0b468ca-2487-45ae-ba1a-297f2a8d01f3
rule = SubsRule(f + 1 => sin(g))

# ╔═╡ 0fa0cd54-125f-40e0-8ce9-9939ac0c3f5d
rule(f + 1)

# ╔═╡ 715ded47-e2b4-4bfa-ad59-05b2db4d3cbc
md"""
When the rule doesn't match, it returns `nothing`:
"""

# ╔═╡ 51847573-8105-4d0a-985e-db0b5e63618a
nested_expr = sin(f + 1) + (f + 1)^2

# ╔═╡ 76c7c3e7-3927-41d0-b88c-1b96ce57f004
rule(nested_expr)

# ╔═╡ 8722c712-a3cb-472f-b7e9-466d73d36c3d
md"""
Wait, but there is clearly $(string(rule.lhs)) in the `nested_expr`! The problem is that the matching subexpression is not at the top level in the expression:
"""

# ╔═╡ f3e15626-53eb-402f-ab07-3f5ff17e7206
head(nested_expr), children(nested_expr)

# ╔═╡ 99747341-4b83-479a-af5e-f0b6ecb7a9f4
md"""
Chmy.jl provides a few convenient "wrapper" rules, inspired by [SymbolicUtils.jl](https://docs.sciml.ai/SymbolicUtils/stable/manual/rewrite/#Composing-rewriters). One of these rules is `Passthrough`, which returns the unmodified expression if there was no match:
"""

# ╔═╡ ce83a2df-2570-407b-887e-de1fab697f73
Passthrough(rule)(nested_expr)

# ╔═╡ 28da8c88-67b8-4060-8b90-926cf2674bf1
md"""
To recursively traverse the expression tree, applying the rule at each level, Chmy.jl provides `Prewalk` and `Postwalk` rules, for traversing the expression tree in pre- and post-order, respectively:
"""

# ╔═╡ 48e10e6d-b1d5-4ce6-85b2-2e23172e0af7
Postwalk(rule)(nested_expr)

# ╔═╡ a70000a3-6dbd-4ca0-abfa-fa366669fd2d
md"""
Finally, we can implement the Maple-like subs function by combining the rules:
"""

# ╔═╡ 06597ebe-b4dd-4947-8c34-db3945887b51
subs(expr::STerm, kv::Pair) = Postwalk(SubsRule(kv))(expr)

# ╔═╡ 00fca560-cedd-4345-9aa7-edc4b01ce8cf
subs(nested_expr, f + 1 => cos(g))

# ╔═╡ 1155a511-d693-4aca-b191-7d29cff46ce3
md"""
## Grids

There is preliminary support for grids in Chmy.jl, which helps with writing dimensionally-independent codes.
"""

# ╔═╡ 3d6d5a0a-9ada-486e-93de-90445bd9203d
nx, ny = 51, 51

# ╔═╡ eea542c9-ec9d-4e79-a3e5-53414c1e8246
grid = Grid(nx, ny)

# ╔═╡ eaafb6d2-3d29-4c85-b010-73ea8c94b09f
dims(grid)

# ╔═╡ 35cc09f7-1757-478c-b1e8-815e175db107
dims(grid, s, s)

# ╔═╡ fd58c068-4246-4401-8566-4cf72cde205e
inds = indices(grid)

# ╔═╡ c1a82c0a-0bc7-41e1-b615-a2ec43e0c78b
lower_stencil(expr[inds...])

# ╔═╡ 365d4078-7e9e-4d20-914c-0bdceb62d8b4
md"""
## Vector and tensor calculus
"""

# ╔═╡ 6809d195-228b-4144-a55e-217d2ccb3a93
md"""
The best way to develop numerical codes is two write the governing PDEs in coordinate-independent form using covariant derivatives. Then stencils for 1D, 2D and 3D can be generated from a single expression by attaching the corresponding number of indices.
"""

# ╔═╡ 28eab159-715d-48ca-8c67-b3f0da51bc62
V = Vec(Tag(:V, 1), Tag(:V, 2))

# ╔═╡ 090ba3e9-599b-4225-9ad5-f92c69ec5537
V + V

# ╔═╡ eb6b63de-2071-4d03-97e7-723fd39f4336
2V

# ╔═╡ ba1b6cfd-a6a6-42f7-98c5-becf323eb3db
V ⋅ V

# ╔═╡ bdac230e-cf50-4107-b425-9cd26fc942e6
τ = SymmetricTensor{2,2}(Tag(:τ, 1, 1), Tag(:τ, 1, 2), Tag(:τ, 2, 2))

# ╔═╡ afbaed9a-e631-472b-9eee-9ab6889b21be
J₂(τ) = sqrt(0.5 * (τ ⊡ τ))

# ╔═╡ 61b76c59-4ea7-4be3-9610-0435339a5a6c
J₂(τ)

# ╔═╡ eecd228b-bcfe-438b-913f-722680493fac
md"""
### Gradient and divergence

Chmy provides gradient and divergence operators to represent fully coordinate-independent differential operators:
"""

# ╔═╡ 498dbc25-cfdf-43ed-9212-8640c9dbc7f5
grad = Gradient{ndims(grid)}(∂.op)

# ╔═╡ ae6bb862-ec26-4af4-98ea-882ce9ccfb08
divg = Divergence{ndims(grid)}(∂.op)

# ╔═╡ bbb22d3e-8843-473b-8be5-c32fefb69423
divg(V)

# ╔═╡ dda215e4-6b3d-4a47-ade1-a9820db00b87
lower_stencil(divg(V)[s, s][i, j])

# ╔═╡ bddba274-8d67-4c32-97d8-beb860705e3b
md"""
## Calling arbitrary Julia functions from Chmy expressions

Chmy DSL is quite expressive, but unfortunately limited to the static computations, without complex control flow such as if statements or loops.

It is possible to use your custom functions in Chmy expressions by using the `SFun` wrapper:
"""

# ╔═╡ 17b67dfe-47fb-47fe-8498-2fb2fd739a71
function foo(x, y)
	if x > y
		return 1.0
	else
		return x * y
	end
end

# ╔═╡ 0219769b-f730-4d4e-b4c1-0317119bc385
SFun(foo)(f, g) + 1

# ╔═╡ b4fa6af4-0b84-49ee-a989-3a05cda4950c
md"""
## Example: 2D diffusion

To conclude this short intro, we will develop a steady diffusion solver in 2D.

Define the diffusion flux:
"""

# ╔═╡ 79660990-bf7b-4f78-8a35-60b0bdf7e72d
q = -grad(f)

# ╔═╡ d90c7a81-7cfc-4cdc-800c-9ff1b2534828
md"""
Then residual:
"""

# ╔═╡ 454d189c-5abd-4214-a3f9-01ad9f18b218
r = -divg(q)

# ╔═╡ 7549f0c1-1187-4eb5-a4a1-c9fcb70090b8
r_c = lower_stencil(r[p, p][inds...])

# ╔═╡ 7cd812e4-90f9-495c-9aef-5d925518acad
md"""
### Boundary conditions

Chmy doesn't provide any special functionality for imposing boundary conditions yet. In the future, there might be high-level API for common cases, but it is possible to implement boundary conditions of arbitrary complexity just by utilising the symbolic manipulation.

In this setup, we specify homogeneous Neumann boundary conditions at $i = 1$ and $i = N_x$, and Dirichlet boundary conditions at $j = 1$ and $j = N_y$:
"""

# ╔═╡ 29c066ff-be81-415b-ad36-54a38190a7e1
begin
f_b = f[p, p][inds[1], inds[2] - 1]
f_t = f[p, p][inds[1], inds[2] + 1]
end;

# ╔═╡ cda956c8-6864-40aa-8010-ae27e955c0d0
begin
q_l = lower_stencil(q[1][s, p][inds[1] - 1, inds[2]])
q_r = lower_stencil(q[1][s, p][inds[1]    , inds[2]])
end;

# ╔═╡ b8b145ca-6326-4d35-b85c-d1ef34ac5f28
begin
bc_l = q_l => SUniform(0)
bc_r = q_r => SUniform(0)
bc_b = f_b => SUniform(+1)
bc_t = f_t => SUniform(-1)
end;

# ╔═╡ fff22af4-e499-4312-af07-d94368807cc7
md"""
Using the just implemented `subs` function, we can "inject" the boundary conditions by directly substituting the fluxes and values into the residual!
"""

# ╔═╡ 0f1f522b-8ed5-41e0-890e-43f1614241bf
begin
r_l = subs(r_c, bc_l)
r_r = subs(r_c, bc_r)
r_b = subs(r_c, bc_b)
r_t = subs(r_c, bc_t)
end;

# ╔═╡ 4119eab1-5740-408c-a241-e6a2db40f017
begin
r_bl = subs(r_b, bc_l)
r_br = subs(r_b, bc_r)
r_tl = subs(r_t, bc_l)
r_tr = subs(r_t, bc_r)
end;

# ╔═╡ d55fd13c-a8ff-4906-9f41-e75b82342137
md"""
Now, the residual function just evaluates numerically the residual at all grid points:
"""

# ╔═╡ 46b52b49-6f12-4a29-af89-81a277fd8133
function compute_residual!(R, r_c, bc, bnd)
	nx, ny = size(R)
	@inbounds begin
		# inner points
		for j in 2:ny - 1, i in 2:nx - 1
			R[i, j] = compute(r_c, bnd, i, j)
		end
		# x sides
		for j in 2:ny - 1
			R[1, j]  = compute(bc.l, bnd, 1, j)
			R[nx, j] = compute(bc.r, bnd, nx, j)
		end
		# y sides
		for i in 2:nx - 1
			R[i, 1]  = compute(bc.b, bnd, i, 1)
			R[i, ny] = compute(bc.t, bnd, i, ny)
		end
		# corners
		R[1, 1]   = compute(bc.bl, bnd, 1, 1)
		R[nx, 1]  = compute(bc.br, bnd, nx, 1)
		R[1, ny]  = compute(bc.tl, bnd, 1, ny)
		R[nx, ny] = compute(bc.tr, bnd, nx, ny)
	end
	return
end

# ╔═╡ 016cdd41-c10c-4ccd-8646-5761a097c684
md"""
Finally, let's write our solver:
"""

# ╔═╡ 217808ea-584e-479a-826d-a0aafc5f872e
bc = (l=r_l, r=r_r, b=r_b, t=r_t, bl=r_bl, br=r_br, tl=r_tl, tr=r_tr);

# ╔═╡ 4c67cc1c-bf5a-4d21-b7b2-d13bdaa43152
begin
# arrays
R = zeros(dims(grid, p, p));
diff_bnd = Binding(
	f[p, p] => rand(dims(grid, p, p)...)
)
f_a = diff_bnd[f[p, p]];
# visualisation
fig = Figure(size=(650, 270))
ax  = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="initial guess"),
	   Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="final solution"))
plt = (heatmap!(ax[1], f_a; colorrange=(-1, 1), colormap=:roma),
	   heatmap!(ax[2], f_a; colorrange=(-1, 1), colormap=:roma))
cb  = (Colorbar(fig[1, 1][1, 2], plt[1]),
	   Colorbar(fig[1, 2][1, 2], plt[2]))
# iterative loop
for iter in 1:100_000
	compute_residual!(R, r_c, bc, diff_bnd)
	@. f_a += 0.25 * R
end
# update plots
plt[2][1] = f_a
fig
end

# ╔═╡ Cell order:
# ╟─34237eee-ca2e-11f0-9a35-39b494c55d7b
# ╟─1e0cd8df-528c-4868-985a-393a10e539cd
# ╟─a80768e2-0f6b-4d7f-8863-3528fa092291
# ╟─2eec34ba-d46a-4687-bb80-3194e1179aef
# ╟─c06ba65e-5e63-418e-82de-08801ff0f3c8
# ╠═5645e4ae-20e6-4dc4-9113-2aa4e2fc998b
# ╟─b2597873-92cb-41bb-8a15-6db882ade747
# ╠═8b1e7194-953a-4a1b-9ca8-9e486bcb612f
# ╟─22697e58-064c-479c-b5fe-7f8d1cdf4d1d
# ╠═53f06a87-6a56-4eef-9e00-5628b5825ac7
# ╟─d4b972a4-de54-4a8c-864e-ed231acb7704
# ╠═597ae26a-0d2f-4941-8c92-039368081786
# ╟─c29ac9d1-9c1c-4337-bc46-0f93b032186e
# ╠═5bc99c52-684d-4c00-b67c-e581965fc9a8
# ╠═208d1fe5-2553-4a69-b1c1-e44e6f33be3d
# ╠═458df925-0d1f-4b0b-9e6d-240b3f7ecc6b
# ╠═40de0823-2af2-4207-851b-a437a8bc7820
# ╠═5ca01538-b4ce-424f-8fdf-729e3a377828
# ╠═64ea4639-90fb-4c25-942e-e461e2edbad7
# ╟─6a175155-a76f-476b-b6af-7c9f30499c00
# ╟─8faa14c9-6bdf-411c-829c-c5e3645b4311
# ╠═96b8658f-6d51-420f-86aa-9c017cecbb97
# ╟─12c7b7e4-fffc-4143-a790-d52e275b2a3b
# ╠═5a1c972e-20cd-45d7-a77b-584724d21c9f
# ╟─03edaa61-3855-4ca2-a246-6cef14b56f02
# ╠═098fa57e-37ff-474b-bac4-953389b44470
# ╟─8ce4d6c3-b234-40eb-986b-f5b85dc36c32
# ╠═79bbbd1a-1b83-44f1-aafa-4e1846d5c4a6
# ╟─b78692db-e0b0-45b0-b830-ed32b50e5dd1
# ╠═22ff2325-67de-4e2a-8d4e-c777729d37f8
# ╟─a044abce-5208-4d9e-9179-00ae808257d4
# ╠═8a9d6e44-7ed3-43f0-9ad2-cf190d4df008
# ╟─3a909d0d-0b7a-431c-822f-f476b0c25ddb
# ╠═65083f09-b49d-4148-9812-e51473e0bc64
# ╟─71c76795-4c5c-4c30-9e88-16461a360072
# ╠═41194bd1-aaa4-4a34-8196-1607508f019d
# ╠═1a153714-8de0-4bbc-ab9c-aed49477dffa
# ╟─a7c625b4-7994-4b2b-80c4-440e1884ca44
# ╠═8b3cf604-4b9b-4a51-8f6e-38125a46909a
# ╟─5fbfe498-1d93-4bfa-a2d7-4c81ee36165e
# ╠═bcd7be0b-9a5d-4540-a29b-9aa13eafde6d
# ╟─d4cbfb3e-fdef-43d2-95f4-f4653121d9e1
# ╠═3bd61dc2-0215-4cbd-abe1-183edd12b4e7
# ╠═2b75fa91-2d0a-4077-9e34-e927152c86ec
# ╟─fced73c1-c15e-4f59-a668-edb6b4297e40
# ╠═d2db2103-d2f1-4c92-9ae8-7d1e096b27ae
# ╠═e4245dae-aeba-448e-bdc3-2091e4cea752
# ╟─07b8416b-73f9-4f02-86e3-bb09cb499fbc
# ╠═3ff47ee9-f16a-4e20-bab6-50cbd9f4e09d
# ╟─5857337c-05d6-46e7-aac2-1b3c9fa07956
# ╠═6a38f026-a732-4ebb-a4de-fbf6ad949e34
# ╠═9118a3fc-317d-408e-90d7-3e4c176f0452
# ╠═d36d8b52-ccf8-4be8-bae6-0d30d4ffc436
# ╠═7e2836b0-342f-42c2-ac17-1a01f564b51b
# ╟─381c451c-12b4-4bf6-8e21-fdfffb4bd5bc
# ╟─7d8d5820-e437-4d97-98f7-9372f199857b
# ╠═07b21203-f561-4ea5-821d-49db291c5a0c
# ╠═e592b9dc-920a-4b22-9266-3b8ad94b1279
# ╠═80b5a14b-f690-4730-bc07-7f0782934b07
# ╟─955d22ed-938a-4a2d-9a67-35aa8a8b82b0
# ╠═394557dd-6e71-4ba5-80ab-320fad844edf
# ╠═ee075fbe-13b3-4255-993b-d93982836e83
# ╠═c7a4aa0c-2377-429b-a4be-7169419b031e
# ╟─069e01eb-cb9a-4138-af9a-0b6c69f81774
# ╟─1628ac5c-7a32-4297-86cc-c11d4ac7881f
# ╠═ccecaea0-250e-4034-bc57-088368aa6768
# ╠═cfc4ecd1-246b-4772-b2d4-ba11b66574de
# ╠═d6451119-0a1f-4420-aaf8-f44372f11ae7
# ╠═7f11e3e0-f2ac-44d1-863e-af867344ea96
# ╟─6f887d81-78c1-4d87-bc2c-c64c7e7dce2f
# ╟─2e151b7e-63b4-4b2b-a09b-7b9e27a1494f
# ╠═56e19c0b-c592-4528-8e23-0ed3b9ca0e0b
# ╠═a0b468ca-2487-45ae-ba1a-297f2a8d01f3
# ╠═0fa0cd54-125f-40e0-8ce9-9939ac0c3f5d
# ╟─715ded47-e2b4-4bfa-ad59-05b2db4d3cbc
# ╠═51847573-8105-4d0a-985e-db0b5e63618a
# ╠═76c7c3e7-3927-41d0-b88c-1b96ce57f004
# ╟─8722c712-a3cb-472f-b7e9-466d73d36c3d
# ╠═f3e15626-53eb-402f-ab07-3f5ff17e7206
# ╟─99747341-4b83-479a-af5e-f0b6ecb7a9f4
# ╠═ce83a2df-2570-407b-887e-de1fab697f73
# ╟─28da8c88-67b8-4060-8b90-926cf2674bf1
# ╠═48e10e6d-b1d5-4ce6-85b2-2e23172e0af7
# ╟─a70000a3-6dbd-4ca0-abfa-fa366669fd2d
# ╠═06597ebe-b4dd-4947-8c34-db3945887b51
# ╠═00fca560-cedd-4345-9aa7-edc4b01ce8cf
# ╟─1155a511-d693-4aca-b191-7d29cff46ce3
# ╠═3d6d5a0a-9ada-486e-93de-90445bd9203d
# ╠═eea542c9-ec9d-4e79-a3e5-53414c1e8246
# ╠═eaafb6d2-3d29-4c85-b010-73ea8c94b09f
# ╠═35cc09f7-1757-478c-b1e8-815e175db107
# ╠═fd58c068-4246-4401-8566-4cf72cde205e
# ╠═c1a82c0a-0bc7-41e1-b615-a2ec43e0c78b
# ╟─365d4078-7e9e-4d20-914c-0bdceb62d8b4
# ╟─6809d195-228b-4144-a55e-217d2ccb3a93
# ╠═28eab159-715d-48ca-8c67-b3f0da51bc62
# ╠═090ba3e9-599b-4225-9ad5-f92c69ec5537
# ╠═eb6b63de-2071-4d03-97e7-723fd39f4336
# ╠═e7168060-8cc7-4965-8224-3f3c844289a9
# ╠═ba1b6cfd-a6a6-42f7-98c5-becf323eb3db
# ╠═bdac230e-cf50-4107-b425-9cd26fc942e6
# ╠═afbaed9a-e631-472b-9eee-9ab6889b21be
# ╠═61b76c59-4ea7-4be3-9610-0435339a5a6c
# ╟─eecd228b-bcfe-438b-913f-722680493fac
# ╠═498dbc25-cfdf-43ed-9212-8640c9dbc7f5
# ╠═ae6bb862-ec26-4af4-98ea-882ce9ccfb08
# ╠═bbb22d3e-8843-473b-8be5-c32fefb69423
# ╠═dda215e4-6b3d-4a47-ade1-a9820db00b87
# ╟─bddba274-8d67-4c32-97d8-beb860705e3b
# ╠═17b67dfe-47fb-47fe-8498-2fb2fd739a71
# ╠═0219769b-f730-4d4e-b4c1-0317119bc385
# ╟─b4fa6af4-0b84-49ee-a989-3a05cda4950c
# ╠═79660990-bf7b-4f78-8a35-60b0bdf7e72d
# ╟─d90c7a81-7cfc-4cdc-800c-9ff1b2534828
# ╠═454d189c-5abd-4214-a3f9-01ad9f18b218
# ╠═7549f0c1-1187-4eb5-a4a1-c9fcb70090b8
# ╟─7cd812e4-90f9-495c-9aef-5d925518acad
# ╠═29c066ff-be81-415b-ad36-54a38190a7e1
# ╠═cda956c8-6864-40aa-8010-ae27e955c0d0
# ╠═b8b145ca-6326-4d35-b85c-d1ef34ac5f28
# ╟─fff22af4-e499-4312-af07-d94368807cc7
# ╠═0f1f522b-8ed5-41e0-890e-43f1614241bf
# ╠═4119eab1-5740-408c-a241-e6a2db40f017
# ╟─d55fd13c-a8ff-4906-9f41-e75b82342137
# ╠═46b52b49-6f12-4a29-af89-81a277fd8133
# ╟─016cdd41-c10c-4ccd-8646-5761a097c684
# ╠═217808ea-584e-479a-826d-a0aafc5f872e
# ╠═4c67cc1c-bf5a-4d21-b7b2-d13bdaa43152
