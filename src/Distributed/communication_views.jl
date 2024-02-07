get_recv_view(side::Val{S}, dim::Val{D}, f::Field) where {S,D} = get_recv_view(side, dim, parent(f), halo(f))

function get_send_view(side::Val{S}, dim::Val{D}, f::Field) where {S,D}
    get_send_view(side, dim, parent(f), halo(f), overlap(location(f, dim)))
end

function get_recv_view(::Val{1}, ::Val{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (halo_width+1):(2halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_recv_view(::Val{2}, ::Val{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (size(array, D)-2halo_width+1):(size(array, D)-halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Val{1}, ::Val{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (overlap+2halo_width+1):(overlap+3halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Val{2}, ::Val{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (size(array, D)-overlap-3halo_width+1):(size(array, D)-overlap-2halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end
