### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 34237eee-ca2e-11f0-9a35-39b494c55d7b
# ╠═╡ show_logs = false
begin
    import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["Revise", "PlutoLinks"])
	Pkg.develop(path="..")
	using PlutoLinks, Revise
end


# ╔═╡ 1b2341be-2d5b-428e-9e99-2f3d69313d32
# ╠═╡ show_logs = false
@revise using Chmy

# ╔═╡ a80768e2-0f6b-4d7f-8863-3528fa092291
md"""
# Intro to Chmy.jl

Hi everyone!

## Fields
"""

# ╔═╡ 96b8658f-6d51-420f-86aa-9c017cecbb97
f = Tag(:f)

# ╔═╡ 5a1c972e-20cd-45d7-a77b-584724d21c9f
expr = cos(f + 1)

# ╔═╡ 098fa57e-37ff-474b-bac4-953389b44470
head(expr), children(expr)

# ╔═╡ 79bbbd1a-1b83-44f1-aafa-4e1846d5c4a6
i, j, k = SIndex(1), SIndex(2), SIndex(3)

# ╔═╡ 22ff2325-67de-4e2a-8d4e-c777729d37f8
stencil_expr = expr[i]

# ╔═╡ e28d25fe-add7-443b-ab3e-cd40dc44f416


# ╔═╡ 8a9d6e44-7ed3-43f0-9ad2-cf190d4df008
lowered_expr = lower_stencil(stencil_expr)

# ╔═╡ 65083f09-b49d-4148-9812-e51473e0bc64
@code_llvm lower_stencil(stencil_expr)

# ╔═╡ 41194bd1-aaa4-4a34-8196-1607508f019d
g = Tag(:g)

# ╔═╡ 1a153714-8de0-4bbc-ab9c-aed49477dffa
binding = Binding(f => rand(2), g => 1)

# ╔═╡ 8b3cf604-4b9b-4a51-8f6e-38125a46909a
push(binding, g => rand(2))

# ╔═╡ bcd7be0b-9a5d-4540-a29b-9aa13eafde6d
binding[f]

# ╔═╡ 3bd61dc2-0215-4cbd-abe1-183edd12b4e7
btypes = binding_types(binding)

# ╔═╡ 2b75fa91-2d0a-4077-9e34-e927152c86ec
to_expr(lowered_expr, btypes)

# ╔═╡ d2db2103-d2f1-4c92-9ae8-7d1e096b27ae
compute_inbounds(expr, bnd, inds...) = @inbounds compute(expr, bnd, inds...)

# ╔═╡ e4245dae-aeba-448e-bdc3-2091e4cea752
@code_llvm compute_inbounds(lowered_expr, binding, 1)

# ╔═╡ 3ff47ee9-f16a-4e20-bab6-50cbd9f4e09d
p, s = Point(), Segment()

# ╔═╡ 9118a3fc-317d-408e-90d7-3e4c176f0452
lowered_loc_expr = lower_stencil(expr[p, s][i, j])

# ╔═╡ d36d8b52-ccf8-4be8-bae6-0d30d4ffc436
binding_loc = Binding(f[p, s] => rand(1, 1))

# ╔═╡ 7e2836b0-342f-42c2-ac17-1a01f564b51b
compute(lowered_loc_expr, binding_loc, 1, 1)

# ╔═╡ 381c451c-12b4-4bf6-8e21-fdfffb4bd5bc
md"""
## Derivatives
"""

# ╔═╡ 07b21203-f561-4ea5-821d-49db291c5a0c
D = StaggeredCentralDifference()

# ╔═╡ e592b9dc-920a-4b22-9266-3b8ad94b1279
D(f)

# ╔═╡ 80b5a14b-f690-4730-bc07-7f0782934b07
lower_stencil(D(f)[s][i])

# ╔═╡ 394557dd-6e71-4ba5-80ab-320fad844edf
∂ = PartialDerivative(D)

# ╔═╡ ee075fbe-13b3-4255-993b-d93982836e83
∂(f, 1)

# ╔═╡ c7a4aa0c-2377-429b-a4be-7169419b031e
lower_stencil(∂(f, 1)[p, s, s][i, j, k])

# ╔═╡ 3d6d5a0a-9ada-486e-93de-90445bd9203d
nx, ny, nz = 11, 11, 11

# ╔═╡ eea542c9-ec9d-4e79-a3e5-53414c1e8246
grid = Grid(nx, ny, nz)

# ╔═╡ eaafb6d2-3d29-4c85-b010-73ea8c94b09f
dims(grid)

# ╔═╡ 35cc09f7-1757-478c-b1e8-815e175db107
dims(grid, s, s, s)

# ╔═╡ 28eab159-715d-48ca-8c67-b3f0da51bc62
V = Vec(Tag(:V, 1), Tag(:V, 2))

# ╔═╡ bdac230e-cf50-4107-b425-9cd26fc942e6
τ = SymmetricTensor{2,2}(Tag(:τ, 1, 1), Tag(:τ, 1, 2), Tag(:τ, 2, 2))

# ╔═╡ 498dbc25-cfdf-43ed-9212-8640c9dbc7f5


# ╔═╡ Cell order:
# ╟─34237eee-ca2e-11f0-9a35-39b494c55d7b
# ╟─1b2341be-2d5b-428e-9e99-2f3d69313d32
# ╟─a80768e2-0f6b-4d7f-8863-3528fa092291
# ╠═96b8658f-6d51-420f-86aa-9c017cecbb97
# ╠═5a1c972e-20cd-45d7-a77b-584724d21c9f
# ╠═098fa57e-37ff-474b-bac4-953389b44470
# ╠═79bbbd1a-1b83-44f1-aafa-4e1846d5c4a6
# ╠═22ff2325-67de-4e2a-8d4e-c777729d37f8
# ╠═e28d25fe-add7-443b-ab3e-cd40dc44f416
# ╠═8a9d6e44-7ed3-43f0-9ad2-cf190d4df008
# ╠═65083f09-b49d-4148-9812-e51473e0bc64
# ╠═41194bd1-aaa4-4a34-8196-1607508f019d
# ╠═1a153714-8de0-4bbc-ab9c-aed49477dffa
# ╠═8b3cf604-4b9b-4a51-8f6e-38125a46909a
# ╠═bcd7be0b-9a5d-4540-a29b-9aa13eafde6d
# ╠═3bd61dc2-0215-4cbd-abe1-183edd12b4e7
# ╠═2b75fa91-2d0a-4077-9e34-e927152c86ec
# ╠═d2db2103-d2f1-4c92-9ae8-7d1e096b27ae
# ╠═e4245dae-aeba-448e-bdc3-2091e4cea752
# ╠═3ff47ee9-f16a-4e20-bab6-50cbd9f4e09d
# ╠═9118a3fc-317d-408e-90d7-3e4c176f0452
# ╠═d36d8b52-ccf8-4be8-bae6-0d30d4ffc436
# ╠═7e2836b0-342f-42c2-ac17-1a01f564b51b
# ╟─381c451c-12b4-4bf6-8e21-fdfffb4bd5bc
# ╠═07b21203-f561-4ea5-821d-49db291c5a0c
# ╠═e592b9dc-920a-4b22-9266-3b8ad94b1279
# ╠═80b5a14b-f690-4730-bc07-7f0782934b07
# ╠═394557dd-6e71-4ba5-80ab-320fad844edf
# ╠═ee075fbe-13b3-4255-993b-d93982836e83
# ╠═c7a4aa0c-2377-429b-a4be-7169419b031e
# ╠═3d6d5a0a-9ada-486e-93de-90445bd9203d
# ╠═eea542c9-ec9d-4e79-a3e5-53414c1e8246
# ╠═eaafb6d2-3d29-4c85-b010-73ea8c94b09f
# ╠═35cc09f7-1757-478c-b1e8-815e175db107
# ╠═28eab159-715d-48ca-8c67-b3f0da51bc62
# ╠═bdac230e-cf50-4107-b425-9cd26fc942e6
# ╠═498dbc25-cfdf-43ed-9212-8640c9dbc7f5
