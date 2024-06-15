# Boundary Conditions



## Boundary Functions

```bash
BoundaryFunction           
CBF                         
CDBF
ContinuousBoundaryFunction  
DBF                     
DiscreteBoundaryFunction

```


## Dirichlet Boundary Conditions


```bash

Dirichlet                   
DirichletKind       
```

```bash
julia> Chmy.BoundaryConditions.
AbstractBatch               
BCOrTuple                   
BatchSet
    
DimSide
        

EmptyBatch                  
ExchangeBatch               
FBC
FBCOrNothing                
FieldAndBC                  
FieldBatch
FieldBoundaryCondition      
FirstOrderBC                
FullDimensions
        
PerFieldBC
ReducedDimensions           
SDA                         
SG
SidesBCs                    
TupleBC                     
_params
_reduce                     
batch                       
batch_impl
batch_set                   
bc!                         
bc_kernel!
cpu_bc_kernel!              
default_bcs                 
default_exchange
delta_index                 
eval                        
expand
flux_sign                   
gpu_bc_kernel!              
halo_index
itp_halo_index              
neighbor_index
prune                       
regularise                  
regularise_exchange
regularise_impl             
reorder                     
value
```

## Neumann Boundary Conditions

```bash
Neumann                     
NeumannKind         
```


## Boundary Conditions on Distributed Fields

On a distributed architecture, we could offload the workload to be performed on a grid into distributed workload on subgrids.

TODO: 
TODO: add `exchange_halo!`