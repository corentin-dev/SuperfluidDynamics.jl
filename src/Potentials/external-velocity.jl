"""
    PotentialExternalVelocity

Potential of an imposed external advection velocity ``u_\\text{adv}``, for
[`GrossPitaevskiiParameters`](@ref). The potential stored on the field is

```math
V = |u_\\text{adv}|^2 / α
```

with `α` a constructor argument (default `1`): the normalisation is the caller's
choice.

The potential has the following informations:

- `f`: the field on which the potential acts,
- `V`: a potential field (real),
- `uadvx`, `uadvy`: vector fields,
- `uadvz`: vector field (3D only; empty in 2D).

# Example

```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> β = 40.; coeffΔ = -0.05;
julia> PotentialExternalVelocity(field, α=-4*coeffΔ*β)
TaylorGreen Potential
```
"""
struct PotentialExternalVelocity{F} <: AbstractPotential{F}
    f::F
    V::AbstractArray
    uadvx::AbstractArray
    uadvy::AbstractArray
    uadvz::AbstractArray

    function PotentialExternalVelocity(f::F;
                                       uadv_function=uadv_function_taylorgreen2D,
                                       α::Real=1) where {F<:AbstractField2D}
        V = real.(similar(f.ϕ))
        uadvx = similar(V)
        uadvy = similar(V)
        uadvz = []
        # velocity field
        @. uadvx = uadv_function[1].(f.x, f.y)
        @. uadvy = uadv_function[2].(f.x, f.y)
        # potential
        @. V = (uadvx^2 + uadvy^2) / α
        return new{F}(f, V, uadvx, uadvy, uadvz)
    end

    function PotentialExternalVelocity(f::F;
                                       uadv_function=uadv_function_taylorgreen3D,
                                       α::Real=1) where {F<:AbstractField3D}
        V = real.(similar(f.ϕ))
        uadvx = similar(V)
        uadvy = similar(V)
        uadvz = similar(V)
        # velocity field
        @. uadvx = uadv_function[1].(f.x, f.y, f.z)
        @. uadvy = uadv_function[2].(f.x, f.y, f.z)
        @. uadvz = uadv_function[3].(f.x, f.y, f.z)
        # potential
        @. V = (uadvx^2 + uadvy^2 + uadvz^2) / α
        return new{F}(f, V, uadvx, uadvy, uadvz)
    end
end

uadv_function_taylorgreen2D = ((x, y) -> sin(x) * cos(y), (x, y) -> -cos(x) * sin(y))
uadv_function_taylorgreen3D = ((x, y, z) -> sin(x) * cos(y) * cos(z),
                               (x, y, z) -> -cos(x) * sin(y) * cos(z), (x, y, z) -> 0.0)

Base.show(io::IO, p::PotentialExternalVelocity) = print(io, "TaylorGreen Potential")
