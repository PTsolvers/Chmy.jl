# Distributed

**Task-based parallelism** in [Chmy.jl](https://github.com/PTsolvers/Chmy.jl) is featured by the usage of [`Threads.@spawn`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@spawn), with an additional layer of a [Worker](../developer_documentation/workers.md) construct for efficiently managing the lifespan of tasks. Note that the task-based parallelism provides a high-level abstraction of program execution **not only** for **shared-memory architecture** on a single machine, but it can be also extended to **hybrid parallelism**, consisting of both shared and distributed-memory parallelism. The `Distributed` module in Chmy.jl allows users to leverage the hybrid parallelism through the power of abstraction.

We will start with some basic background knowledge for understanding the architecture of modern HPC clusters, the underlying memory model and the programming paradgm complied with it. We then introduce how Chmy.jl provides a high-level API for users to abstract the low-level details away and then a simple example showing how the `Distributed` module should be used.

## HPC Cluster & Distributed Memory

An **high-performance computing (HPC)** cluster consists of a **network** of independent computers combined into a system through specialized hardware. We call each computer a *node*, and each node manages its own private memory. Such system with interconnected nodes, without having access to memory of any other node, features the **distributed memory model**. The underlying fast interconnect architecture (*InfiniBand*) that physically connects the nodes in the **network** can transfer the data from one node to another in an extremely efficient manner through a communication protocol called remote direct memory access (RDMA).

```@raw html
<img src="../assets/compute_cluster.jpg" width="50%"/>
```

By using the *InfiniBand*, processes across different nodes can communicate with each other through the sending of messages in a high-throughput, low-latency fashion. The syntax and semantics of how **message passing** should proceed through such network is defined by a standard called the **Message-Passing Interface (MPI)**, and there are different libraries that implement the standard, resulting in a wide range of choice (MPICH, Open MPI, MVAPICH etc.) for users. 


!!! info "Message-Passing Interface (MPI) is a General Specification"
    In general, implementations based on **MPI standard** can be used for a great variety of computers, not just on HPC clusters, as long as these computers are connected by a communication network.


## Hybrid Parallelism

