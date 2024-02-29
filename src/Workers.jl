module Workers

export Worker

"""
    Worker

A worker that performs tasks asynchronously.

# Constructor
    Worker{T}(; [setup], [teardown]) where {T}

Constructs a new `Worker` object.

## Arguments
- `setup`: A function to be executed before the worker starts processing tasks. (optional)
- `teardown`: A function to be executed after the worker finishes processing tasks. (optional)
"""
mutable struct Worker{T}
    src::Channel{T}
    out::Base.Event
    task::Task

    function Worker{T}(; setup=nothing, teardown=nothing) where {T}
        src = Channel{T}()
        out = Base.Event(true)
        task = Threads.@spawn begin
            isnothing(setup) || invokelatest(setup)
            try
                for work in src
                    invokelatest(work)
                    notify(out)
                end
            finally
                isnothing(teardown) || invokelatest(teardown)
            end
        end
        errormonitor(task)
        return new{T}(src, out, task)
    end
end

Worker(; kwargs...) = Worker{Any}(; kwargs...)

function Base.close(p::Worker)
    close(p.src)
    wait(p.task)
    return
end

Base.isopen(p::Worker) = isopen(p.src)

function Base.put!(work::T, p::Worker{T}) where {T}
    put!(p.src, work)
    return
end

function Base.wait(p::Worker)
    if isopen(p.src)
        wait(p.out)
    else
        error("Worker is not running")
    end
    return
end

end
