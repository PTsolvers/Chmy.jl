# Workers

Task-based parallelism provides a highly abstract view for the program execution scheduling, although it may come with a performance overhead related to task creation and destruction. The overhead is currently significant when tasks are used to perform asynchronous operations on GPUs, where TLS context creation and destruction may be in the order of kernel execution time. Therefore, it may be desirable to have long-running tasks that do not get terminated immediately, but only when all queued subtasks (i.e. work units) are executed.

In [Chmy.jl](https://github.com/PTsolvers/Chmy.jl), we introduced the concept `Worker` for this purpose. A `Worker` is a special construct to extend the lifespan of a task created by `Threads.@spawn`. It possesses a `Channel` of subtasks to be executed on the current thread, where subtasks are submitted at construction time to the worker using `put!`.

With the help of the `Worker`, we can specify subtasks that need to be sequentially executed by enqueuing them to one `Worker`. Any work units that shall run in parallel should be put into separate workers instead.

Currently, we use Workers under the hood to hide the communications between computations. We split the computational domain into inner part, containing the bulk of the grid points, and thin outer part. We launch the same kernels processing the inner and outer parts in different Julia tasks. When the outer part completes, we launch the non-blocking MPI communication. Workers are a stateful representation of the long-running computation, needed to avoid significant overhead of creating a new task-local state each time a communication is performed.
