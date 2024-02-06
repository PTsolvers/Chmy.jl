module DoubleBuffers

export DoubleBuffer, swap!, front, back

mutable struct DoubleBuffer{T}
    front::T
    back::T
end

function swap!(db::DoubleBuffer)
    db.front, db.back = db.back, db.front
end

front(db::DoubleBuffer) = db.front
back(db::DoubleBuffer) = db.back

end
