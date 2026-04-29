"""
    HaloArray(parent, halowidths)
    HaloArray(parent, halowidths...)

Wrap a padded array and expose only its interior through standard one-based
axes. Integer indices outside those axes are accepted when they land inside the
stored halo region.
"""
struct HaloArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    halowidths::NTuple{N,Tuple{Int,Int}}

    function HaloArray{T,N,A}(parent::A, halowidths::NTuple{N,Tuple{Int,Int}}) where {T,N,A<:AbstractArray{T,N}}
        _check_parent_halowidths(parent, halowidths)
        return new{T,N,A}(parent, halowidths)
    end
end

function HaloArray(parent::A, halowidths...) where {T,N,A<:AbstractArray{T,N}}
    widths = _normalize_constructor_halowidths(halowidths, Val(N))
    return HaloArray{T,N,A}(parent, widths)
end

function HaloArray{T}(::UndefInitializer, dims::NTuple{N,<:Integer}, halowidths...) where {T,N}
    interior_dims = _normalize_dims(dims, Val(N))
    widths = _normalize_constructor_halowidths(halowidths, Val(N))
    padded_dims = ntuple(i -> interior_dims[i] + widths[i][1] + widths[i][2], Val(N))
    return HaloArray(Array{T}(undef, padded_dims), widths)
end

Base.parent(A::HaloArray) = A.parent
halowidths(A::HaloArray) = A.halowidths

Base.IndexStyle(::Type{<:HaloArray}) = IndexCartesian()

Base.size(A::HaloArray{T,N}) where {T,N} = ntuple(i -> size(parent(A), i) - halowidths(A)[i][1] - halowidths(A)[i][2], Val(N))
Base.axes(A::HaloArray) = map(Base.OneTo, size(A))

Base.@propagate_inbounds function Base.getindex(A::HaloArray{T,N}, I::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds parent(A)[_parent_indices(A, I)...]
end

Base.@propagate_inbounds function Base.setindex!(A::HaloArray{T,N}, v, I::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds parent(A)[_parent_indices(A, I)...] = v
    return A
end

function Base.checkbounds(::Type{Bool}, A::HaloArray{T,N}, I::Vararg{Integer,N}) where {T,N}
    sz = size(A)
    widths = halowidths(A)
    return all(ntuple(i -> 1 - widths[i][1] <= I[i] <= sz[i] + widths[i][2], Val(N)))
end
Base.checkbounds(::Type{Bool}, ::HaloArray, ::Vararg{Integer}) = false

"""
    interior(A::HaloArray)

Return a view of the interior portion of the padded parent array.
"""
interior(A::HaloArray) = view(parent(A), _interior_ranges(A)...)

"""
    halo(A::HaloArray, face::Face)

Return a view of the halo region identified by `face`. Spanning dimensions use
the interior range, so side halos do not include corner halo regions.
"""
function halo(A::HaloArray{T,N}, face::Face) where {T,N}
    ndims(face) == N || throw(ArgumentError("face dimension $(ndims(face)) does not match array dimension $N"))
    ranges = ntuple(i -> _halo_range(A, face.axes[i], i), Val(N))
    return view(parent(A), ranges...)
end

Adapt.adapt_structure(to, A::HaloArray) = HaloArray(Adapt.adapt(to, parent(A)), halowidths(A))

function _normalize_constructor_halowidths(args::Tuple, ::Val{N}) where {N}
    if length(args) == 1
        width_arg = args[1]
        if N == 1 && _is_halowidth_pair(width_arg)
            return (_normalize_halowidth(width_arg),)
        end
        return _normalize_halowidths(width_arg, Val(N))
    elseif length(args) == N
        return ntuple(i -> _normalize_halowidth(args[i]), Val(N))
    else
        throw(ArgumentError("expected either one halo width collection or $N halo width entries, got $(length(args)) entries"))
    end
end

function _normalize_halowidths(halowidths, ::Val{N}) where {N}
    length(halowidths) == N || throw(ArgumentError("expected $N halo width entries, got $(length(halowidths))"))
    return ntuple(i -> _normalize_halowidth(halowidths[i]), Val(N))
end

_is_halowidth_pair(_) = false
_is_halowidth_pair(::Tuple{<:Integer,<:Integer}) = true

function _normalize_halowidth(width::Tuple{<:Integer,<:Integer})
    lower, upper = Int(width[1]), Int(width[2])
    lower >= 0 && upper >= 0 || throw(ArgumentError("halo widths must be nonnegative, got $width"))
    return (lower, upper)
end
_normalize_halowidth(width) = throw(ArgumentError("halo width must be a pair of integers, got $width"))

function _normalize_dims(dims, ::Val{N}) where {N}
    return ntuple(Val(N)) do i
        dim = Int(dims[i])
        dim >= 0 || throw(ArgumentError("interior dimensions must be nonnegative, got $dims"))
        dim
    end
end

function _check_parent_halowidths(parent, widths::NTuple{N,Tuple{Int,Int}}) where {N}
    for i in 1:N
        halo_extent = widths[i][1] + widths[i][2]
        size(parent, i) >= halo_extent ||
            throw(ArgumentError("parent size in dimension $i must be at least $halo_extent, got $(size(parent, i))"))
    end
    return nothing
end

_parent_indices(A::HaloArray{T,N}, I) where {T,N} = ntuple(i -> I[i] + halowidths(A)[i][1], Val(N))

_interior_ranges(A::HaloArray{T,N}) where {T,N} = ntuple(i -> _interior_range(A, i), Val(N))

function _interior_range(A::HaloArray, i)
    lower = halowidths(A)[i][1]
    return lower+1:lower+size(A, i)
end

function _halo_range(A::HaloArray, ::Lower, i)
    lower = halowidths(A)[i][1]
    return 1:lower
end

function _halo_range(A::HaloArray, ::Upper, i)
    lower = halowidths(A)[i][1]
    interior_size = size(A, i)
    upper = halowidths(A)[i][2]
    return lower+interior_size+1:lower+interior_size+upper
end

_halo_range(A::HaloArray, ::Span, i) = _interior_range(A, i)
