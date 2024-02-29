overlap(::Vertex) = 1
overlap(::Center) = 0

function get_recv_view(side::Side, dim::Dim, f::Field)
    return get_recv_view(side, dim, parent(f), halo(f))
end

function get_send_view(side::Side, dim::Dim, f::Field)
    return get_send_view(side, dim, parent(f), halo(f), overlap(location(f, dim)))
end

function get_recv_view(::Side{1}, ::Dim{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (halo_width+1):(2halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_recv_view(::Side{2}, ::Dim{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (size(array, D)-2halo_width+1):(size(array, D)-halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Side{1}, ::Dim{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (overlap+2halo_width+1):(overlap+3halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Side{2}, ::Dim{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (size(array, D)-overlap-3halo_width+1):(size(array, D)-overlap-2halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end
