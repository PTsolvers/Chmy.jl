module Workers

export Worker

mutable struct Worker{T}
    src::Channel{T}
    out::Base.Event
    task::Task

    function Worker{T}(; setup=nothing, teardown=nothing) where {T}
        src = Channel{T}()
        out = Base.Event(true)
        task = Threads.@spawn begin
            isnothing(setup) || Base.invokelatest(setup)
            try
                for work in src
                    Base.invokelatest(work)
                    notify(out)
                end
            finally
                isnothing(teardown) || Base.invokelatest(teardown)
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
