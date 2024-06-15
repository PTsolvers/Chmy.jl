# Grid Operators



## Difference Quotients

The gist of the finite difference relies on replacing derivatives by difference quotients anchored on structured grids.

```bash
    # Used by ∂x/y/z
δ                  
δx                 
δy                 
δz

∂   # not actually used?
∂x                 
∂y                 
∂z

    # compute field divergence
divg
```


## Masking

Masking is particularly important when performing finite differences on GPUs, as it allows for efficient and accurate computations by selectively applying operations only where needed, allowing more flexible control over the grid operators and improving performance.

Thus, by providing masked grid operators, we enable more flexible control over the domain on which the grid operators should be applied for advanced users. When computing masked derivatives, a mask being the subtype of `AbstractMask` is premultiplied at the corresponding grid location for each operand.

```bash
AbstractMask       
FieldMask          
FieldMask1D        
FieldMask2D        
FieldMask3D
at                 
```



## Interpolation

```bash
InterpolationRule  

HarmonicLinear     
hlerp              

Linear             
lerp

itp                
itp_dims           
itp_impl           
itp_knots
itp_knots_impl     
itp_rule           
itp_weight         
```


```julia
@kernel inbounds = true function update_q!(q, D, S, g)
    I = @index(Global, Cartesian)
    q.x[I] = -lerp(D, location(q.x), g, I) * ∂x(S, g, I)
    q.y[I] = -lerp(D, location(q.y), g, I) * ∂y(S, g, I)
end
```


## TODO


misc

```bash
# for both interpolation and mask
il                 
ir                 

               
m                  
p

left               
leftx
lefty              
leftz              

right              
rightx             
righty             
rightz             
vmag
```