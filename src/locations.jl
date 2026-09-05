"""
Point location on a staggered grid.
"""
struct Point end

"""
Segment location on a staggered grid.
"""
struct Segment end

"""
Location on a staggered grid.
"""
const Location = Union{Point,Segment}

const 𝓅 = Point()
const 𝓈 = Segment()
