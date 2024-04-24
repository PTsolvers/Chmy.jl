struct FieldMask1D{T,CF,VF} <: AbstractMask{T,1}
    c::CF
    v::VF
    function FieldMask1D{T}(c::AbstractField{T,1}, v::AbstractField{T,1}) where {T}
        return new{T,typeof(c),typeof(v)}(c, v)
    end
end

function FieldMask1D(arch::Architecture, grid::StructuredGrid{1}, type=eltype(grid); kwargs...)
    c = Field(arch, grid, Center(), type; kwargs...)
    v = Field(arch, grid, Vertex(), type; kwargs...)
    return FieldMask1D{type}(c, v)
end

@propagate_inbounds at(ω::FieldMask1D, ::Tuple{Center}, ix) = ω.c[ix]
@propagate_inbounds at(ω::FieldMask1D, ::Tuple{Vertex}, ix) = ω.v[ix]

struct FieldMask2D{T,CCF,VVF,VCF,CVF} <: AbstractMask{T,2}
    cc::CCF
    vv::VVF
    vc::VCF
    cv::CVF
    function FieldMask2D{T}(cc::AbstractField{T,2}, vv::AbstractField{T,2}, vc::AbstractField{T,2}, cv::AbstractField{T,2}) where {T}
        return new{T,typeof(cc),typeof(vv),typeof(vc),typeof(cv)}(cc, vv, vc, cv)
    end
end

function FieldMask2D(arch::Architecture, grid::StructuredGrid{2}, type=eltype(grid); kwargs...)
    cc = Field(arch, grid, Center(), type; kwargs...)
    vv = Field(arch, grid, Vertex(), type; kwargs...)
    vc = Field(arch, grid, (Vertex(), Center()), type; kwargs...)
    cv = Field(arch, grid, (Center(), Vertex()), type; kwargs...)
    return FieldMask2D{type}(cc, vv, vc, cv)
end

@propagate_inbounds at(ω::FieldMask2D, ::Tuple{Center,Center}, ix, iy) = ω.cc[ix, iy]
@propagate_inbounds at(ω::FieldMask2D, ::Tuple{Vertex,Vertex}, ix, iy) = ω.vv[ix, iy]
@propagate_inbounds at(ω::FieldMask2D, ::Tuple{Vertex,Center}, ix, iy) = ω.vc[ix, iy]
@propagate_inbounds at(ω::FieldMask2D, ::Tuple{Center,Vertex}, ix, iy) = ω.cv[ix, iy]

struct FieldMask3D{T,CCCF,VVVF,VCCF,CVCF,CCVF,VVCF,VCVF,CVVF} <: AbstractMask{T,3}
    ccc::CCCF
    vvv::VVVF
    vcc::VCCF
    cvc::CVCF
    ccv::CCVF
    vvc::VVCF
    vcv::VCVF
    cvv::CVVF
    function FieldMask3D{T}(ccc::AbstractField{T,3},
                            vvv::AbstractField{T,3},
                            vcc::AbstractField{T,3},
                            cvc::AbstractField{T,3},
                            ccv::AbstractField{T,3},
                            vvc::AbstractField{T,3},
                            vcv::AbstractField{T,3},
                            cvv::AbstractField{T,3}) where {T}
        return new{T,
                   typeof(ccc),
                   typeof(vvv),
                   typeof(vcc),
                   typeof(cvc),
                   typeof(ccv),
                   typeof(vvc),
                   typeof(vcv),
                   typeof(cvv)}(ccc, vvv, vcc, cvc, ccv, vvc, vcv, cvv)
    end
end

function FieldMask3D(arch::Architecture, grid::StructuredGrid{3}, type=eltype(grid); kwargs...)
    ccc = Field(arch, grid, Center(), type; kwargs...)
    vvv = Field(arch, grid, Vertex(), type; kwargs...)
    vcc = Field(arch, grid, (Vertex(), Center(), Center()), type; kwargs...)
    cvc = Field(arch, grid, (Center(), Vertex(), Center()), type; kwargs...)
    ccv = Field(arch, grid, (Center(), Center(), Vertex()), type; kwargs...)
    vvc = Field(arch, grid, (Vertex(), Vertex(), Center()), type; kwargs...)
    vcv = Field(arch, grid, (Vertex(), Center(), Vertex()), type; kwargs...)
    cvv = Field(arch, grid, (Center(), Vertex(), Vertex()), type; kwargs...)
    return FieldMask2D{type}(ccc, vvv, vcc, cvc, ccv, vvc, vcv, cvv)
end

@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Center,Center,Center}, ix, iy, iz) = ω.ccc[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Vertex,Vertex,Vertex}, ix, iy, iz) = ω.vvv[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Vertex,Center,Center}, ix, iy, iz) = ω.vcc[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Center,Vertex,Center}, ix, iy, iz) = ω.cvc[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Center,Center,Vertex}, ix, iy, iz) = ω.ccv[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Vertex,Vertex,Center}, ix, iy, iz) = ω.vvc[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Vertex,Center,Vertex}, ix, iy, iz) = ω.vcv[ix, iy, iz]
@propagate_inbounds at(ω::FieldMask3D, ::Tuple{Center,Vertex,Vertex}, ix, iy, iz) = ω.cvv[ix, iy, iz]

FieldMask(arch::Architecture, grid::StructuredGrid{1}, args...; kwargs...) = FieldMask1D(arch, grid, args...; kwargs...)
FieldMask(arch::Architecture, grid::StructuredGrid{2}, args...; kwargs...) = FieldMask2D(arch, grid, args...; kwargs...)
FieldMask(arch::Architecture, grid::StructuredGrid{3}, args...; kwargs...) = FieldMask3D(arch, grid, args...; kwargs...)
