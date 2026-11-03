# face patterns

"""
Abstract supertype for the patterns of the faces of the domain, matched with `matches(pattern, face)`.
"""
abstract type FacePattern end

"""
Abstract supertype for the patterns of the walls of the domain, i.e. of its faces of codimension 1.
"""
abstract type WallPattern <: FacePattern end

struct AnyWallPattern <: WallPattern end

"""
    AlongAxis(axis)

Matches both walls normal to `axis`.
"""
struct AlongAxis <: WallPattern
    axis::Int
end

"""
    Wall(axis, side)

Matches the wall normal to `axis` on the `side`, `↓` or `↑`.
"""
struct Wall <: WallPattern
    axis::Int
    side::Side
end

# single walls, with the last axis pointing up
struct LeftWallPattern <: WallPattern end
struct RightWallPattern <: WallPattern end
struct BottomWallPattern <: WallPattern end
struct TopWallPattern <: WallPattern end
struct BackWallPattern <: WallPattern end
struct FrontWallPattern <: WallPattern end

"Matches every wall."
const AnyWall = AnyWallPattern()
"Matches the lower wall of the first axis."
const LeftWall = LeftWallPattern()
"Matches the upper wall of the first axis."
const RightWall = RightWallPattern()
"Matches the lower wall of the last axis in 2D and 3D."
const BottomWall = BottomWallPattern()
"Matches the upper wall of the last axis in 2D and 3D."
const TopWall = TopWallPattern()
"Matches the lower wall of the second axis in 3D."
const BackWall = BackWallPattern()
"Matches the upper wall of the second axis in 3D."
const FrontWall = FrontWallPattern()

struct AnyCornerPattern <: FacePattern end

"Matches every corner, i.e. every face of dimension 0."
const AnyCorner = AnyCornerPattern()

dim(p::FacePattern, n) = n - codim(p, n)
codim(p::FacePattern, n) = n - dim(p, n)

codim(::WallPattern, n) = 1
dim(::AnyCornerPattern, n) = 0

matches(::AnyCornerPattern, face::Face) = iscorner(face)

function matches(p::WallPattern, face::Face)
    iswall(face) || return false
    axis = findfirst(!=(:), face)::Int
    return matches(p, axis, face[axis], length(face))
end

matches(::AnyWallPattern, axis, side, n) = true
matches(p::AlongAxis, axis, side, n) = axis == p.axis
matches(p::Wall, axis, side, n) = axis == p.axis && side == p.side
matches(::LeftWallPattern, axis, side, n) = matches(Wall(1, ↓), axis, side, n)
matches(::RightWallPattern, axis, side, n) = matches(Wall(1, ↑), axis, side, n)
matches(::BottomWallPattern, axis, side, n) = n >= 2 && matches(Wall(n, ↓), axis, side, n)
matches(::TopWallPattern, axis, side, n) = n >= 2 && matches(Wall(n, ↑), axis, side, n)
matches(::BackWallPattern, axis, side, n) = n >= 3 && matches(Wall(n - 1, ↓), axis, side, n)
matches(::FrontWallPattern, axis, side, n) = n >= 3 && matches(Wall(n - 1, ↑), axis, side, n)

struct EitherWall{P <: WallPattern, Q <: WallPattern} <: WallPattern
    p::P
    q::Q
end
struct BothWalls{P <: WallPattern, Q <: WallPattern} <: WallPattern
    p::P
    q::Q
end
struct NotWall{P <: WallPattern} <: WallPattern
    p::P
end

"""
    |(p::WallPattern, q::WallPattern)

Matches the walls matched by `p` or `q`.
"""
Base.:|(p::WallPattern, q::WallPattern) = EitherWall(p, q)

"""
    &(p::WallPattern, q::WallPattern)

Matches the walls matched by both `p` and `q`.
"""
Base.:&(p::WallPattern, q::WallPattern) = BothWalls(p, q)

"""
    !(p::WallPattern)

Matches the walls not matched by `p`.
"""
Base.:!(p::WallPattern) = NotWall(p)

matches(e::EitherWall, axis, side, n) = matches(e.p, axis, side, n) || matches(e.q, axis, side, n)
matches(b::BothWalls, axis, side, n) = matches(b.p, axis, side, n) && matches(b.q, axis, side, n)
matches(neg::NotWall, axis, side, n) = !matches(neg.p, axis, side, n)

# region patterns of the outward distance ξ from the wall, positive in the ghost region

"""
Abstract supertype for the patterns of the outward distance from a wall, matched with `matches(pattern, ξ)`.
"""
abstract type RegionPattern end

struct GhostPattern <: RegionPattern end
struct BoundaryPattern <: RegionPattern end
struct InteriorPattern <: RegionPattern end

"""
    Within(width)

Matches the samples at most `width` away from the wall, on either side.
"""
struct Within <: RegionPattern
    width::Rational{Int}
end

"Matches the samples outside the domain."
const Ghost = GhostPattern()
"Matches the samples on the wall."
const Boundary = BoundaryPattern()
"Matches the samples inside the domain."
const Interior = InteriorPattern()

matches(::GhostPattern, ξ::Rational) = ξ > zero(ξ)
matches(::BoundaryPattern, ξ::Rational) = iszero(ξ)
matches(::InteriorPattern, ξ::Rational) = ξ < zero(ξ)
matches(p::Within, ξ::Rational) = abs(ξ) <= p.width

struct EitherRegion{P <: RegionPattern, Q <: RegionPattern} <: RegionPattern
    p::P
    q::Q
end
struct BothRegions{P <: RegionPattern, Q <: RegionPattern} <: RegionPattern
    p::P
    q::Q
end
struct NotRegion{P <: RegionPattern} <: RegionPattern
    p::P
end

"""
    |(p::RegionPattern, q::RegionPattern)

Matches the samples matched by `p` or `q`.
"""
Base.:|(p::RegionPattern, q::RegionPattern) = EitherRegion(p, q)

"""
    &(p::RegionPattern, q::RegionPattern)

Matches the samples matched by both `p` and `q`.
"""
Base.:&(p::RegionPattern, q::RegionPattern) = BothRegions(p, q)

"""
    !(p::RegionPattern)

Matches the samples not matched by `p`.
"""
Base.:!(p::RegionPattern) = NotRegion(p)

matches(e::EitherRegion, ξ::Rational) = matches(e.p, ξ) || matches(e.q, ξ)
matches(b::BothRegions, ξ::Rational) = matches(b.p, ξ) && matches(b.q, ξ)
matches(neg::NotRegion, ξ::Rational) = !matches(neg.p, ξ)

const DefaultRegion = Ghost | Boundary

# tensor patterns hold one index pattern per tensor index: `:` matches every component, `𝐧` the component normal to
# the wall, and `𝐭` the tangential ones
const IndexPattern = Union{Colon, AnyNormal, AnyTangent}
const TensorPattern = Tuple{Vararg{IndexPattern}}

matches(::Colon, comp, normal_axis) = true
matches(::AnyNormal, comp, normal_axis) = comp == normal_axis
matches(::AnyTangent, comp, normal_axis) = comp != normal_axis

matches(tp::TensorPattern, comps, normal_axis)::Bool = all(map((p, c) -> matches(p, c, normal_axis), tp, comps))

# the free indices of a tensor pattern index the tensor data of the rules
isfree(::Colon) = true
isfree(::Union{AnyNormal, AnyTangent}) = false

istangential(::AnyTangent) = true
istangential(::Union{Colon, AnyNormal}) = false

# the wall vectors in the data of rules
tensorrank(::Union{AnyNormal, AnyTangent}, args) = 1

printop(io, ::AnyNormal, args) = print(io, '𝐧')
function printop(io, t::AnyTangent, args)
    print(io, '𝐭')
    t.slot == 1 || print_subscript(io, t.slot)
    return
end

# predicates and matches

"""
Abstract supertype for the predicates of overrides. A predicate implements `Chmy.match(predicate, face, comps, coord,
ctx)`, which returns `nothing`, or an [`OverrideMatch`](@ref) if the component `comps` of the sample at `coord` in the
lowering context `ctx` is overridden at the `face`, a face adjacent to the face of the context. A predicate selecting
tensor components also implements `Chmy.freecomps(predicate, comps)`, the components indexing the tensor data of the
rule, and `Chmy.tangentcomps(predicate, comps)`, the components resolving `𝐭`.
"""
abstract type OverridePredicate end

"""
Abstract supertype for the results of matching an [`OverridePredicate`](@ref). Every match implements
[`signed_axes`](@ref) and [`signed_distances`](@ref), so that rules can be applied with any predicate.
"""
abstract type OverrideMatch end

"""
    signed_axes(match)

The outward normal axes of the matched face, negative on the lower sides.
"""
function signed_axes end

"""
    signed_distances(match)

The outward distances of the sample from the walls of the matched face, in the order of [`signed_axes`](@ref).
"""
function signed_distances end

"""
    FaceMatch(signed_axes, signed_distances)

The minimal [`OverrideMatch`](@ref), storing the signed axes of the matched face and the outward distances of the
sample from its walls.
"""
struct FaceMatch{K} <: OverrideMatch
    signed_axes::NTuple{K, Int}
    signed_distances::NTuple{K, Rational{Int}}
end

signed_axes(m::FaceMatch) = m.signed_axes
signed_distances(m::FaceMatch) = m.signed_distances

signed_axis(axis, side) = flipsign(axis, outward(side))

freecomps(::OverridePredicate, comps) = comps
tangentcomps(::OverridePredicate, comps) = ()

# the length of the tensor pattern of the predicate matches the rank of the overridden term
checkpattern(::OverridePredicate, rank) = nothing

"""
    WallPredicate(wall, region, comps)

Matches the walls matched by the face pattern `wall`, when the outward distance of the sample from the wall matches
the `region` and the tensor component matches the tensor pattern `comps`. The match is a [`FaceMatch`](@ref) of the
wall.
"""
struct WallPredicate{W <: WallPattern, R <: RegionPattern, T <: TensorPattern} <: OverridePredicate
    wall::W
    region::R
    comps::T
end

function match(p::WallPredicate, face, comps, coord, ctx)
    matches(p.wall, face) || return nothing
    axis = findfirst(!=(:), face)::Int
    ξ = distance(ctx, coord, axis)
    (matches(p.region, ξ) && matches(p.comps, comps, axis)) || return nothing
    return FaceMatch((signed_axis(axis, face[axis]),), (ξ,))
end

freecomps(p::WallPredicate, comps) = Tuple(c for (s, c) in zip(p.comps, comps) if isfree(s))
tangentcomps(p::WallPredicate, comps) = Tuple(c for (s, c) in zip(p.comps, comps) if istangential(s))

function checkpattern(p::WallPredicate, rank)
    length(p.comps) == rank ||
        throw(ArgumentError(lazy"tensor pattern $(p.comps) does not match the rank $rank of the overridden term"))
    return
end

"""
    FacePredicate(face, region)

Matches the faces matched by the face pattern `face`, when the outward distances of the sample from all walls of the
face match the `region`. The match is a [`FaceMatch`](@ref) of the face.
"""
struct FacePredicate{F <: FacePattern, R <: RegionPattern} <: OverridePredicate
    face::F
    region::R
end

function match(p::FacePredicate, face, comps, coord, ctx)
    matches(p.face, face) || return nothing
    axes = Tuple(a for a in eachindex(face) if face[a] != (:))
    ξs = map(a -> distance(ctx, coord, a), axes)
    all(ξ -> matches(p.region, ξ), ξs) || return nothing
    return FaceMatch(map(a -> signed_axis(a, face[a]), axes), ξs)
end

# overrides

"""
    Override(predicates)

Overrides a term next to the boundary, called as `Override(predicates)(u, rules...)` with one rule call per predicate.
At every face adjacent to the face of the lowering context, the first matching predicate selects its rule, and the
rules of the matched faces are applied in the order of the predicates. See [`override`](@ref).
"""
struct Override <: Operator
    predicates::Tuple{Vararg{OverridePredicate}}
end

# an override of the component `comps` sampled at `coord`, with the matches of the faces that are still to be applied
struct OverrideSample <: Operator
    override::Override
    comps::Tuple{Vararg{Int}}
    coord::Tuple{Vararg{Rational{Int}}}
    candidates::Tuple{Vararg{Pair{Int, OverrideMatch}}}
end

tensorrank(::Union{Override, OverrideSample}, args) = tensorrank(first(args))

function checkranks(o::Override, args::Tuple{Vararg{DTerm}})
    length(args) == 1 + length(o.predicates) || throw(ArgumentError("an override requires one rule per predicate"))
    rank = tensorrank(first(args))
    foreach(o.predicates, Base.tail(args)) do p, rule
        checkpattern(p, rank)
        nfree = length(freecomps(p, ntuple(identity, rank)))
        all(g -> tensorrank(g) ∈ (0, nfree), arguments(rule)) ||
            throw(ArgumentError(lazy"the data of a rule must be scalars or tensors of rank $nfree"))
    end
    return
end

# the arguments of an override sample are the ones of its override, checked when the override call was built
checkranks(::OverrideSample, ::Tuple{Vararg{DTerm}}) = nothing

# an override prints as the overridden term, highlighted in red
printop(io, ::Union{Override, OverrideSample}, args) = printstyled(io, first(args); color = :red)
printprec(::Union{Override, OverrideSample}, args) = printprec(first(args))

# the first matching predicate of every face adjacent to the face of the context, faces of lower codimension and lower
# axes first, ordered by the predicate
function candidates(predicates, comps, coord, ctx::LoweringContext{N}) where {N}
    sides = Tuple(a for a in 1:N if ctx.face[a] != (:))
    result = Pair{Int, OverrideMatch}[]
    # bit k of the mask selects the k-th side
    for mask in sort!(collect(1:(2^length(sides) - 1)); by = m -> (count_ones(m), m))
        face = ntuple(Val(N)) do a
            k = findfirst(==(a), sides)
            isnothing(k) || iszero(mask & 1 << (k - 1)) ? (:) : ctx.face[a]
        end
        for (i, p) in enumerate(predicates)
            m = match(p, face, comps, coord, ctx)
            isnothing(m) && continue
            push!(result, i => m)
            break
        end
    end
    return Tuple(sort!(result; by = first))
end

function lower_rule(o::Override, args, comps, coord, ctx)
    sample = OverrideSample(o, comps, coord, candidates(o.predicates, comps, coord, ctx))
    return lower_rule(sample, args, comps, coord, ctx)
end

function lower_rule(s::OverrideSample, args, comps, coord, ctx)
    # a sample read elsewhere, e.g. by a mirror rule, is matched again
    (comps == s.comps && coord == s.coord) || return lower_rule(s.override, args, comps, coord, ctx)
    isempty(s.candidates) && return here(first(args)[comps...], coord)
    i, m = first(s.candidates)
    rest = OverrideSample(s.override, s.comps, s.coord, Base.tail(s.candidates))(args...)[comps...]
    rule = args[1 + i]
    pred = s.override.predicates[i]
    axes = signed_axes(m)
    face = (; axes, tangents = tangentcomps(pred, comps), dims = length(coord))
    data = map(g -> slicedata(resolve_face(g, face), pred, comps, axes), arguments(rule))
    # rules are written in the frame of the outward normal
    flip(x) = orient(x, comps, axes)
    return here(flip(boundary_rule(operation(rule), flip(rest), data, m)), coord)
end

# a component changes sign with every index along a mirrored axis
orient(x, comps, signed_axes) = isodd(count(c -> -c ∈ signed_axes, comps)) ? -x : x

# scalar data apply to every component, while tensor data are given in the global frame and their free components are
# taken in the frame of the outward normal
function slicedata(g::DTerm, pred, comps, signed_axes)
    tensorrank(g) == 0 && return g
    free = freecomps(pred, comps)
    return orient(g[free...], free, signed_axes)
end

unitvector(a, s, dims) = pack([i == a ? s : 0 for i in 1:dims])

# the data of a rule with the wall vectors and the boundary data resolved at the matched face
resolve_face(term::DTerm, face) = iscall(term) ? resolve_face(operation(term), term, face) : walkargs(x -> resolve_face(x, face), term)
resolve_face(::Operator, term, face) = walkargs(x -> resolve_face(x, face), term)

# nested overrides resolve the data at their own faces
resolve_face(::Override, term, face) = term

function resolve_face(::AnyNormal, term, face)
    length(face.axes) == 1 ||
        throw(ArgumentError(lazy"the normal is only defined on walls, got a face of codimension $(length(face.axes))"))
    a = only(face.axes)
    return unitvector(abs(a), sign(a), face.dims)
end

function resolve_face(t::AnyTangent, term, face)
    t.slot <= length(face.tangents) ||
        throw(ArgumentError(lazy"the tangent $(t.slot) requires as many tangential indices, got $(length(face.tangents))"))
    # tangential axes are never mirrored, so the tangent is the same in the global and the local frame
    return unitvector(face.tangents[t.slot], 1, face.dims)
end

# boundary data

struct OnFace <: Operator
    span::Bool
end

"""
    𝒷(g; span = false)

The boundary data `g` stored at the face matched by an override: the stored field is located at the walls of the
face, e.g. `g[↓, 𝓈][0]` by the left wall, and along the other axes at the sample, or nowhere with `span = true`, e.g.
`g[↓, :]`, a single value for the left wall. Can only appear in the data of override rules.
"""
𝒷(g; span::Bool = false) = OnFace(span)(asterm(g))

tensorrank(::OnFace, args) = tensorrank(only(args))
checkranks(::OnFace, ::Tuple{Vararg{DTerm}}) = nothing
print_opname(io, ::OnFace) = print(io, '𝒷')

function lower_rule(::OnFace, args, comps, coord, ctx)
    throw(ArgumentError("boundary data `𝒷` can only appear in the data of override rules"))
end

# a stored field located at the global locations `frames`, or at the sample where the frame is `nothing`
struct Stored <: Operator
    frames::Tuple
end

tensorrank(::Stored, args) = tensorrank(only(args))
checkranks(::Stored, ::Tuple{Vararg{DTerm}}) = nothing
print_opname(io, ::Stored) = print(io, '𝒷')

function resolve_face(b::OnFace, term, face)
    frames = ntuple(a -> face_frame(b, a, face.axes), face.dims)
    return Stored(frames)(resolve_face(only(arguments(term)), face))
end

function face_frame(b::OnFace, a, signed_axes)
    k = findfirst(s -> abs(s) == a, signed_axes)
    isnothing(k) || return signed_axes[k] > 0 ? Upper() : Lower()
    return b.span ? (:) : nothing
end

function lower_rule(s::Stored, (x,)::Tuple{DTerm}, comps, coord, ctx)
    y = x[comps...]
    isuniform(y) || isfield(y) || throw(ArgumentError("boundary data `𝒷` can only be given to fields"))
    return leaf(y, s.frames, coord, ctx)
end

# a stored scalar field, or a component of a stored tensor field
function isfield(term::DTerm)
    return @match term begin
        Tensor(_, _, _, _) => true
        DExpr(Comp(_), (Tensor(_, _, _, _),)) => true
        _ => false
    end
end

# rules

"""
Abstract supertype for the rules of overrides, operators without data whose calls hold the data of the rule, e.g.
`Add()(p)`. A rule implements [`boundary_rule`](@ref).
"""
abstract type BoundaryRule <: Operator end

# the data of rules may be tensors
checkranks(::BoundaryRule, ::Tuple{Vararg{DTerm}}) = nothing

"""
    boundary_rule(rule, u, data, match)

The value of the overridden component `u` at the sample, as an expression evaluated at the sample, given the `data` of
the `rule` call and the [`OverrideMatch`](@ref). `u` is the component with the rules of the faces that are applied
later, and `u` and the data are in the frame of the outward normals of the match.
"""
function boundary_rule end

"""
    Add(p)

Adds `p` to the overridden term.
"""
struct Add <: BoundaryRule end
Add(p) = Add()(asterm(p))

"""
    Scale(w)

Multiplies the overridden term by `w`.
"""
struct Scale <: BoundaryRule end
Scale(w) = Scale()(asterm(w))

# a term that is not a rule call replaces the overridden term
struct Replace <: BoundaryRule end

boundary_rule(::Add, u, (p,), match) = u + p
boundary_rule(::Scale, u, (w,), match) = w * u
boundary_rule(::Replace, u, (v,), match) = v

"""
Abstract supertype for the rules applied along the normal of a wall, which implement [`wall_rule`](@ref).
"""
abstract type WallRule <: BoundaryRule end

"""
    wall_rule(rule, u, data, ξ)

The 1D expression of the `rule` of an upper wall, sampled relative to the sample at the outward distance `ξ` from the
wall, e.g. `2g[-ξ] - u[-2ξ]`. The rule is lifted along the normal of the matched wall, mirrored at lower walls.
"""
function wall_rule end

"""
    WallValue(g)

Imposes the value `g` on the wall: on the wall the term is `g`, and in the ghost region the mirror image through the
wall makes the linear interpolation equal to `g`.
"""
struct WallValue <: WallRule end
WallValue(g) = WallValue()(asterm(g))

"""
    WallDerivative(g)

Imposes the outward normal derivative `g` on the wall, through the mirror image of the ghost samples. The values on
the wall are not changed.
"""
struct WallDerivative <: WallRule end
WallDerivative(g) = WallDerivative()(asterm(g))

function boundary_rule(op::WallRule, u, data, match)
    axes = signed_axes(match)
    length(axes) == 1 ||
        throw(ArgumentError(lazy"wall rules are only defined on walls, got a face of codimension $(length(axes))"))
    return Lifted(op, axes)(u, data...)
end

function lower_rule(op::WallRule, (u, data...)::Tuple{DTerm, Vararg{DTerm}}, coord::NTuple{1}, ctx)
    return wall_rule(op, u, data, distance(ctx, coord, 1))
end

function wall_rule(::WallValue, u, (g,), ξ)
    iszero(ξ) && return g[0]
    ξ > 0 && return 2g[-ξ] - u[-2ξ]
    throw(ArgumentError(lazy"WallValue can only be applied on the wall or in the ghost region, got ξ = $ξ"))
end

function wall_rule(::WallDerivative, u, (g,), ξ)
    iszero(ξ) && return u[0]
    ξ > 0 && return u[-2ξ] + 2ξ * g[-ξ]
    throw(ArgumentError(lazy"WallDerivative can only be applied on the wall or in the ghost region, got ξ = $ξ"))
end

# the user interface

asrule(r::TermLike) = Replace()(asterm(r))
asrule(r::DTerm) = iscall(r) ? asrule(operation(r), r) : Replace()(r)
asrule(::BoundaryRule, r) = r
asrule(::Operator, r) = Replace()(r)

splititem(item::Pair) = (first(item), last(item))
splititem(rule) = (nothing, rule)

colons(rank) = ntuple(_ -> (:), rank)

# a predicate given as a pattern, or a tuple of a face pattern, a region pattern, and a tensor or index pattern,
# where the missing parts default to every wall, the ghost region and the wall, and every component
aspredicate(p::OverridePredicate, rank) = p
aspredicate(::Nothing, rank) = aspredicate((), rank)
aspredicate(t::Tuple, rank) = predicate(foldl(setpattern, t; init = (AnyWall, DefaultRegion, nothing))..., rank)
aspredicate(t::TensorPattern, rank) = aspredicate((t,), rank)
aspredicate(x, rank) = aspredicate((x,), rank)

setpattern((f, r, c), x::FacePattern) = (x, r, c)
setpattern((f, r, c), x::RegionPattern) = (f, x, c)
setpattern((f, r, c), x::IndexPattern) = (f, r, (x,))
setpattern((f, r, c), x::TensorPattern) = (f, r, x)
setpattern(spec, x) = throw(ArgumentError(lazy"$x is not a face, region, or tensor pattern"))

predicate(f::WallPattern, r, ::Nothing, rank) = WallPredicate(f, r, colons(rank))
predicate(f::WallPattern, r, c, rank) = WallPredicate(f, r, c)
predicate(f::FacePattern, r, ::Nothing, rank) = FacePredicate(f, r)
predicate(f::FacePattern, r, c, rank) = throw(ArgumentError("tensor patterns can only be matched at walls"))

"""
    override(u, items...)

Override the term `u` next to the boundary. Every item is a rule, or a pair `predicate => rule`, where the predicate is
an [`OverridePredicate`](@ref), a face, region, or tensor pattern, or a tuple of these patterns. Missing patterns
default to `AnyWall`, `Ghost | Boundary`, and every component. A wall pattern, or a tensor pattern, gives a
[`WallPredicate`](@ref), another face pattern a [`FacePredicate`](@ref). A rule is a rule call, e.g. `WallValue(g)`,
or a term or a number, which replaces `u`. See [`Override`](@ref).

```julia
override(u, LeftWall => WallValue(1), WallDerivative(0))
override(V, (𝐧, 𝐭) => WallValue(0))
override(r, (AnyCorner, Boundary) => Scale(1 // 2))
```
"""
function override(u, items::Vararg{Any, N}) where {N}
    term = asterm(u)
    rank = tensorrank(term)
    parts = map(splititem, items)
    predicates = map(p -> aspredicate(first(p), rank), parts)
    return Override(predicates)(term, map(p -> asrule(last(p)), parts)...)
end
