# Coupled PDE systems

Models that stack several scalar fields into one ODE state `(C, D, H, W)` (for example
[tumor; lymphocytes] in `ImmuneResponse3D`) use the same `TorchDiffEqSolver` as
`ReactionDiffusion3D`. Postprocessing and calibration should usually target a **single
component** (often component `0` for tumor density).

## State layout and extraction

::: tumortwin.models.pde_system

## Immune–tumor spatial model

::: tumortwin.models.immune_3d

## Workflow helpers (tutorial-aligned)

::: tumortwin.pde_workflow
