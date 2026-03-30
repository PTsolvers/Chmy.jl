abstract type Space <: STerm end

isuniform(::Space) = true

struct Segment <: Space end
struct Point <: Space end

scale(::Segment)  = 1
offset(::Segment) = -1

scale(::Point)  = 1
offset(::Point) = 0
