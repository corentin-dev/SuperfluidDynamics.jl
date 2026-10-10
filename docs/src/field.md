# Fields

[`Field`](@ref) structures are used to contain the data used by the simulation. It depends on a [Grid](@ref Grids).

!!! info
    While [`Grid`](@ref Grids) is not distributed, [`Field`](@ref) is distributed.

## Type definition

A [`Field`](@ref) contains all the local information of the data. By local, we mean in term of process.

```@docs
Field
```

## Constructors

To construct a [`Field2D`](@ref), a more friendly constructor has been defined as follow:

```@docs
Field(g::AbstractGrid2D{FT,A}, t::FieldType;
               ndims::Integer=1,
               mpi_topo::AbstractDomainDecomposition) where {FT<:Real,A}
```

To construct a [`Field3D`](@ref), a more friendly constructor has been defined as follow:

```@docs
Field(g::AbstractGrid3D{FT,A}, t::FieldType;
               ndims::Integer=1,
               mpi_topo::AbstractDomainDecomposition) where {FT<:Real,A}
```

## Aliases

To have a more friendly type, some aliases has been declared. For the moment, there exists two kinds of fields: [`Field2D`](@ref) and [`Field3D`](@ref).

```@docs
Field2D
```

```@docs
Field3D
```
