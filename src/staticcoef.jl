struct StaticCoef{Value} end

const StaticCoeff = StaticCoef

StaticCoef(value) = StaticCoef{value}()

function StaticCoef(value::Real)
    isbits(value) || error("value must be isbits")
    if isinteger(value)
        value = Int(value)
    end
    return StaticCoef{value}()
end

value(::StaticCoef{Value}) where {Value} = Value

isnegative(c::StaticCoef) = value(c) < zero(value(c))

Base.iszero(c::StaticCoef) = iszero(value(c))
Base.isone(c::StaticCoef) = isone(value(c))
Base.isinteger(c::StaticCoef) = isinteger(value(c))
Base.abs(c::StaticCoef) = StaticCoef(abs(value(c)))

Base.:+(c::StaticCoef) = c
Base.:-(c::StaticCoef) = StaticCoef(-value(c))

Base.:+(a::StaticCoef, b::StaticCoef) = StaticCoef(value(a) + value(b))
Base.:-(a::StaticCoef, b::StaticCoef) = StaticCoef(value(a) - value(b))
Base.:*(a::StaticCoef, b::StaticCoef) = StaticCoef(value(a) * value(b))

function Base.:/(a::StaticCoef, b::StaticCoef)
    va = value(a)
    vb = value(b)
    # Keep integer division exact
    if va isa Integer && vb isa Integer
        return StaticCoef(va // vb)
    else
        return StaticCoef(va / vb)
    end
end
