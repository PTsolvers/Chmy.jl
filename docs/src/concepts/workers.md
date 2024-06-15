# Task-based Parallelism

[Chmy.jl](https://github.com/PTsolvers/Chmy.jl) features task-based parallelism. The design of the task-based parallelism API in our package aims to interoperate seamlessly with the native Julia parallel computing support, which was detailed in the section _"Parallel Computing"_ in the [Julia official manual](https://docs.julialang.org/en/v1/manual/parallel-computing/#Parallel-Computing).

Our understanding of task-based parallelism builds upon the thread programming model, where each thread represents sequences of instructions to be executed. If tasks are performed on a single thread in a sequential manner, then the task execution is **blocking**.

On the other hand, **asynchronous** tasks comprise a split-up of workload such that the core can switch between the tasks to execute, resulting in the **non-blocking** execution of tasks.

## Single-threaded Concurrency

Before we delve into more details in [Chmy.jl](https://github.com/PTsolvers/Chmy.jl), let us review some parallel programming concepts in Julia. In the following example, we consider a `proc_sleep` function that causes the execution of the current task to halt for 3 seconds.

```julia
function proc_sleep()
    sleep(3)
    return 37
end
```

Having a single thread executing the function sequentially, the total time of execution will be around 9 seconds.

```julia
# takes roughly 9 seconds to execute
@elapsed begin
    p1 = proc_sleep()
    p2 = proc_sleep()
    p3 = proc_sleep()
    (p1, p2, p3)
end
```

Using the **asynchronous** approach, we can start each `proc_sleep` as a separate unit of task, allowing CPU to switch between tasks during execution.

```julia
# takes only about 3 seconds to execute
@elapsed begin
    t1 = @task proc_sleep(); schedule(t1)
    t2 = @task proc_sleep(); schedule(t2)
    t3 = @task proc_sleep(); schedule(t3)
    (fetch(t1), fetch(t2), fetch(t3))
end
```


## Multi-threaded Parallelism

In the previous section, we have been executing tasks concurrently on a single thread. To exploit the true power of multicore architectures, one can use `Threads.@spawn` macro to create a `Task` and schedule it to run on any available thread.

```julia
# takes about 3 seconds to execute
@elapsed begin
    t1 = Threads.@spawn proc_sleep()
    t2 = Threads.@spawn proc_sleep()
    t3 = Threads.@spawn proc_sleep()
    (fetch(t1), fetch(t2), fetch(t3))
end
```



## Channels

A `Channel` in Julia is a data buffer which can have multiple `Task`s reading from and writing to it using `fetch`/`take!` and `put!` functions respectively.

```julia
# Creates a channel with 2 slots of objects of any data type
ch = Channel(2)

# Use put! to place items
put!(ch, "foo")
put!(ch, 1_000)

# Use fetch to inspect the first item
fetch(ch)
```

We could also consume the item by using `take!` instead. This takes away the first item from the channel's data buffer, which is a first-in, first-out (FIFO) queue.

```julia
# Use take! to fetch items away from the buffer, this returns "foo"
take!(ch)

# This returns 1000
take!(ch)
```

## Workers

Task-based parallelism provides a highly abstract view for the program execution scheduling, although it may come with a performance overhead related to task creation and destruction. The overhead is currently significant when tasks are used to perform asynchronous operations on GPUs, where TLS context creation and destruction may be in the order of kernel execution time. Therefore, it may be desirable to have long-running tasks that do not get terminated immediately, but only when all queued subtasks (i.e. work units) are executed.

In [Chmy.jl](https://github.com/PTsolvers/Chmy.jl), we introduced the concept `Worker` for this purpose. A `Worker` is a special construct to extend the lifespan of a task created by `Threads.@spawn`. It possesses a `Channel` of subtasks to be executed on the current thread, where subtasks are submitted at construction time to the worker using `put!`.

With the help of the `Worker`, we can specify subtasks that need to be sequentially executed by enqueuing them to one `Worker`. Any work units that shall run in parallel should be put into separate workers instead.