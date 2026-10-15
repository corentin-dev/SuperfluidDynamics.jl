# Numerical schemes

## Gross-Pitaevskii

### Imaginary time

```@docs
NumModelBackwardEuler
NumModelBackwardEulerNoPrecond
NumModelCrankNicolson
NumModelCrankNicolsonQuasiNewton
```

### Real time

```@docs
NumModelSplit1
NumModelSplit2
NumModelCrankNicolsonT
NumModelCrankNicolsonQuasiNewtonT
NumModelGPRK
```

`NumModelGPRK` is the **explicit** Runge-Kutta integrator (RK1/RK2/RK4, port of
the reference `GP_RK4`), as opposed to the implicit/splitting schemes above.
It is unconditionally *accurate* but subject to the dispersive stability limit
`|coeffΔ| k_max² Δt ≲ 2` (RK1) / `≲ 2.8` (RK4), so it is intended for
benchmarks, short runs and as an accuracy reference; the implicit schemes are
preferred for long, high-resolution evolutions.

## Dealiasing

Only `NumModelGPRK` filters, and it is opt-in per position:
`filter = :product` (the default) filters the nonlinear **term** inside each stage,
`:solution` filters the **state** after each step, `:none` filters nothing.

`NumModelSplit1` and `NumModelSplit2` do **not** dealias, and that is intentional: the
time-splitting schemes of the reference GPS implementation are called without its
`flag_cutoff`, so the port does not filter either. The same holds for
`NumModelCrankNicolson`, `NumModelCrankNicolsonQuasiNewton`,
`NumModelCrankNicolsonT`, `NumModelBackwardEuler` and `NumModelExternalVelocity`,
which filter nothing — here the reference *does* apply its cutoff to the state, so
those five differ from GPS on that point. Measured consequence: small. The paired
fundamental-state comparisons agree with GPS to 1.7 ppm on the energy (see the runs
repository, `results/gps_paired_cpu.md`); that divergence has not been isolated from
the other differences between the two codes.

What not filtering costs, measured on the dipole family (`β = 1`, box 80):

* At the resolution-stable setting (`|coeffΔ| k_max² Δt = 0.126`, 128²) energy is
  conserved to `1e-06` over 2 000 steps by `Split2` and by `NumModelGPRK` RK4 with
  `filter = :none`. Filtering is what *breaks* conservation there: `:product` drifts
  by +0.11 %, `:solution` by −0.16 %.
* At `|coeffΔ| k_max² Δt = 2.021` (512², `Δt = 0.01`) `Split2` does not conserve
  energy at all: `E/E₀ = 338` at `t = 100`, with 93 % of the kinetic energy sitting
  above `2/3·kmax`. Removing that band divides `E_kin` by 37 and stops the growth.
* Filtering the state is destructive, not conservative: projecting onto the 2/3 band
  after every step removes 33 % of the energy over 2 000 steps (128²) and 30 % at
  512²/`Δt = 0.01`.

So `Δt` carries a `k_max²` constraint in the unfiltered splitting schemes; the
observed stable/exploding split falls in `|coeffΔ| k_max² Δt` and not in `Δt` or `dx`
alone. Filtering the state would trade that failure for a 30 % energy loss, which is
why it is not done by default anywhere.

### External velocity

```@docs
NumModelExternalVelocity
```
