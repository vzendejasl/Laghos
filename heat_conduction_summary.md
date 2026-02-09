# Heat Conduction Implementation in Laghos (DG-based)

This document summarizes the linear heat conduction implementation added to the Laghos Lagrangian hydrodynamics miniapp.

## 1. Mathematical Model

### Lagrangian Energy Equation
In the Lagrangian frame, the governing equation for specific internal energy ($e$) includes the work done by pressure, viscous dissipation, and heat conduction:

$$\rho \frac{de}{dt} = -p (\nabla \cdot \mathbf{v}) + \boldsymbol{\tau} : \nabla \mathbf{v} + \nabla \cdot (\kappa \nabla T)$$

where:
*   $\rho$ is the density.
*   $\mathbf{v}$ is the velocity.
*   $p$ is the pressure.
*   $\boldsymbol{\tau}$ is the viscous stress tensor.
*   $\kappa$ is the thermal conductivity.
*   $T$ is the temperature.

### Thermodynamic Relations (Ideal Gas)
Assuming an ideal gas law with gas constant $R=1$:
*   State Equation: $p = \rho (\gamma - 1) e$
*   Temperature: $T = (\gamma - 1) e$
*   Specific Heat at Constant Pressure: $c_p = \frac{\gamma}{\gamma - 1}$

### Conductivity ($\kappa$)
The conductivity is related to the dynamic viscosity ($\mu$) via the Prandtl number ($Pr$):
*   $\kappa = \frac{\mu c_p}{Pr} = \frac{\mu \gamma}{Pr (\gamma - 1)}$

### Effective Energy Diffusion
Substituting $T = (\gamma - 1) e$ into the conduction term:
$$\nabla \cdot (\kappa \nabla T) = \nabla \cdot \left( \kappa (\gamma - 1) \nabla e \right) = \nabla \cdot (\kappa_{e} \nabla e)$$
where the effective energy conductivity is:
$$\kappa_{e} = \kappa (\gamma - 1) = \frac{\mu \gamma}{Pr}$$

## 2. Weak Form Derivation

To solve $\rho \frac{de}{dt} = \dots + \nabla \cdot (\kappa_e \nabla e)$ using Finite Elements, we multiply by a test function $\phi \in L_2$ and integrate over the domain $\Omega$:

$$\int_\Omega \rho \frac{de}{dt} \phi \, d\Omega = \dots + \int_\Omega \nabla \cdot (\kappa_e \nabla e) \phi \, d\Omega$$

Using integration by parts (Green's First Identity) on the conduction term:
$$\int_\Omega \nabla \cdot (\kappa_e \nabla e) \phi \, d\Omega = \int_{\partial \Omega} (\kappa_e \nabla e \cdot \mathbf{n}) \phi \, d\Gamma - \int_\Omega \kappa_e \nabla e \cdot \nabla \phi \, d\Omega$$

### Discontinuous Galerkin (DG) Treatment
Since $e \in L_2$ is discontinuous across element boundaries, we must use the Interior Penalty (IP) method. The bilinear form for the diffusion operator $K(e, \phi)$ is:

$$a(e, \phi) = \sum_K \int_K \kappa_e \nabla e \cdot \nabla \phi \, d\Omega - \sum_f \int_f \{ \kappa_e \nabla e \cdot \mathbf{n} \} [\phi] \, d\Gamma - \sum_f \int_f \{ \kappa_e \nabla \phi \cdot \mathbf{n} \} [e] \, d\Gamma + \sum_f \int_f \sigma [e][\phi] \, d\Gamma$$

where:
*   $\{ \cdot \}$ denotes the average across a face.
*   $[\cdot]$ denotes the jump across a face.
*   $\sigma$ is the penalty parameter ensuring stability.

In Laghos, this is implemented using `DiffusionIntegrator` (domain) and `DGDiffusionIntegrator` (faces).

## 3. Step-by-Step Implementation Approach

1.  **Operator Extension:** Added `use_conduction` and `prandtl_number` to `LagrangianHydroOperator`.
2.  **Conductivity Calculation:** In `UpdateConductionOperator`, calculate $\kappa_e = \frac{\mu \gamma}{Pr}$ based on the current viscosity.
3.  **Operator Assembly:**
    *   Re-assemble the `K_cond` bilinear form every time step to account for the moving mesh (Jacobians change).
    *   Use `DiffusionIntegrator` for domain diffusion.
    *   Use `DGDiffusionIntegrator` for face jumps to maintain stability in $L_2$.
4.  **RHS Integration:** In `SolveEnergy`, compute the conduction contribution $R_{cond} = K_{cond} \cdot \mathbf{e}$ and subtract it from the energy time derivative.
5.  **Time Step Stability:** Added a parabolic CFL constraint in `GetTimeStepEstimate`:
    $$\Delta t_{cond} \le 0.5 \frac{h^2}{\kappa_e (p+1)^2}$$

## 4. Key Code Additions

### UpdateConductionOperator
```cpp
void LagrangianHydroOperator::UpdateConductionOperator(const Vector &S) const {
   const double Pr = prandtl_number;
   const double mu = (viscosity_const >= 0.0) ? viscosity_const : 0.0;
   const double kappa_e = mu * gamma / Pr;

   *u_cond_gf = kappa_e;
   u_cond_gf->ExchangeFaceNbrData();

   K_cond_bf = new ParBilinearForm(&L2);
   K_cond_bf->AddDomainIntegrator(new DiffusionIntegrator(*cond_coeff));
   K_cond_bf->AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(*cond_coeff, sigma_cond, kappa_dg_cond));
   K_cond_bf->Assemble();
   K_cond.Reset(K_cond_bf->ParallelAssemble(), true);
}
```

### Energy RHS Update
```cpp
if (use_conduction) {
   Vector cond_rhs(L2Vsize);
   K_cond->Mult(e_vec, cond_rhs);
   e_rhs.Add(-1.0, cond_rhs); // Subtract conduction from Energy RHS
}
```

## 5. Verification: Heat Conduction Test Problem

To verify the implementation, a test problem involving the decay of a sinusoidal temperature field was added.

### Problem Definition
*   **Domain:** $[0, 1]^3$ periodic.
*   **Initial Conditions:**
    *   $\rho(t=0) = 1.0$
    *   $\mathbf{v}(t=0) = \mathbf{u}_0$ (Uniform translation)
    *   $e(t=0) = e_{bg} + \hat{e} \sin(2\pi x) \sin(2\pi y) \sin(2\pi z)$
*   **Analytical Solution:** The energy perturbation decays exponentially:
    $$e(t) = e_{bg} + \hat{e} \exp(-k^2 \frac{\kappa_e}{\rho} t) \sin(\mathbf{k} \cdot \mathbf{x})$$
    where $k^2 = (2\pi)^2 \times \text{dim}$.

### Convergence Study
The `convergence_study.py` script automates this test. It calculates the RMS of the heat conduction power and compares it against the analytical value. The implementation shows $O(h^p)$ or $O(h^{p+1})$ convergence depending on the element orders $ok$ (kinematic) and $ot$ (thermodynamic).
