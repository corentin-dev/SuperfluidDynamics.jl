# Numerical schemes

## Navier-Stokes

Incompressible Navier-Stokes equations in the vorticity-advection form,
`∂t u = νΔu − P(u × ∇×u)`, solved by a semi-implicit Runge-Kutta (RK4)
scheme following the GPS approach [1]. The viscous Laplacian is handled by
the exact implicit multiplier `exp(−ν Δt |k|²)`, with the spectral Helmholtz
projector `P` and 2/3-rule dealiasing. Works in 2D and 3D.

```@docs
NavierStokesParameters
NumModelRK4Imp
taylor_green!
```

## Two-fluid models

Two models for quantum two-fluid flows are provided. They share the incompressible
spectral Navier-Stokes solver for the normal fluid.

### NSGP (coupled Gross-Pitaevskii / Navier-Stokes)

The coupled model of Brachet et al. [2]: a non-stationary
Gross-Pitaevskii equation for the superfluid wavefunction ψ coupled, through the
Coste coupling (regularised superfluid velocity, counterflow, mutual friction,
GP coupling potential V_xm), to a forced incompressible Navier-Stokes equation
for the normal fluid. One-way or two-way coupling; RK1/RK2/RK4; the normal-fluid
Laplacian is implicit (`exp(−ν Δt |k|²)`), the GP one explicit, with a
mass-correction chemical potential and (optional) η_D dissipation.

```@docs
NSGPParameters
NumModelNSGP
```

### HVBK (linear Hall-Vinen-Bekarevich-Khalatnikov)

The linear HVBK two-fluid model [3]: two incompressible velocity fields (normal and
superfluid), each advecting its own vorticity, coupled by the **linear mutual
friction** `F = −½ rb |∇×u_s| (u_n − u_s)` with density weighting `ρs/ρt` and
`−ρn/ρt` (the total momentum is conserved). RK1/RK2, exact implicit viscous
multiplier, Helmholtz projection and 2/3 dealiasing. The implementation covers
the model above only: the nonlinear Iksada friction and the transverse
Anisov–Peiris–Pitaevskii term of [3] are not implemented.

```@docs
HVBKParameters
NumModelHVBK
```

## References

- [1] P. Parnaudeau, A. Suzuki, J.-M. Sac-Epée, "GPS: an efficient and
      spectrally accurate code for computing the Gross-Pitaevskii equation",
      ISC-2015, Research Posters Session (2015).
- [2] M. Brachet, G. Sadaka, Z. Zhang, V. Kalt, I. Danaila, "Coupling
      Navier-Stokes and Gross-Pitaevskii equations for the numerical
      simulation of two-fluid quantum flows", *J. Comput. Phys.* **488**,
      112193 (2023).
- [3] Z. Zhang, I. Danaila, E. Lévêque, L. Danaila, "Higher-order statistics
      and intermittency of a two-fluid Hall--Vinen--Bekarevich--Khalatnikov
      quantum turbulent flow", *J. Fluid Mech.* **962**, A22 (2023).
