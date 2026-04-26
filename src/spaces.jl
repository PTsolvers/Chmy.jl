abstract type Space <: STerm end

isuniform(::Space) = true

struct Segment <: Space end
struct Point <: Space end

offset(::Segment) = SLiteral(1 // 2)
offset(::Point)   = SLiteral(0)
