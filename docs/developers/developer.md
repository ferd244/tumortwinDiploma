# Guidance for developers

`TumorTwin` is designed to be a foundation for you to build on top of and extend!

The core workflow in `TumorTwin` is as follows:
# ![TumorTwin Workflow](../assets/workflow.png)

Extending the package to your own use-case or research task could involve modifying one or more of these core features - For each we provide a brief discussion about how you might think about and approach development:
1. [Simulating a new Patients](#simulating a-new-patient)
2. [Modifying the `PatientData` object](#modifying-the-patientdata-object)
3. [Modifying or creating a new `TumorGrowthModel`](#modifying-or-creating-a-new-tumorgrowthmodel)
4. [Introducing new treatment models](#introducing-new-treatment-models)
5. [Experimenting with different `Solver`s](#experimenting-with-different-solvers)
6. [Experimenting with different `Optimizer`s](#experimenting-with-different-optimizers)


## Simulating a new Patient
We store information about each patient in a `.json` configuration file ([HGG Example](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/input_files/HGG_demo_001/HGG_demo_001.json), [TNBC Example](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/input_files/TNBC_demo_001/TNBC_demo_001.json))

These files contain all the treatment and imaging data (via pointers to image files) for a specific patient, and are designed to be loaded into their respective `PatientData` objects, via their `.from_file()` method. Under the hood, `PatientData` objects are [`pydantic`](https://docs.pydantic.dev/latest/) data models, and we use their [json parsing](https://docs.pydantic.dev/latest/concepts/json/#json-parsing) to perform data validation. Thus, you should make sure that your `json` file is compatible with the `PatientData` object you intend to use.

Filepaths in a patient configuration can be written out in full. As a convenience, we also support the use of an environment variable called `image_dir` to specify a parent directory for your medical imaging data. Any instance of `{$image_dir}` in a configuration file will be replaced by either the corresponding `image_url` variable in a `.env` file, or by the contents of the `image_dir` keyword argument specified when calling the `.from_file()` method. This convenience can allow different users to use the *same* configuration files by specifying an `image_dir` that is specific to their file system.

## Modifying the `PatientData` object
If you want to modify or extend the data sources that feed in to a patient-specific digital twin (e.g., additional treatment parameters, additional imaging types, additional non-imaging data about the patient, etc.), then you will likely need to create a new `PatientData` object for your use-case.

We provide a [`BasePatientData`](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/tumortwin/types/base.py#L19) class which provides a basic structure and some convenient helper functions. We also provide two examples that inherit from this base class: [`HGGPatientData`](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/tumortwin/types/hgg_data.py#L12) and [`TNBCPatientData`](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/tumortwin/types/tnbc_data.py#L12). We suggest studying these examples in order to understand how you might extend them to your needs. 

## Modifying or creating a new `TumorGrowthModel`
We provide a basic [3D reaction-diffusion model of tumor growth](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/tumortwin/models/reaction_diffusion_3d.py) (for details see the [Theory page](../theory/theory.md)).

This model inherits from a base `TumorGrowthModel3D` class. At its core, that interface defines an explicit ordinary differential equation (ODE) model. The `forward(t, u)` method evaluates the right-hand-side of the equation, i.e., the time derivative `du/dt`. In our reaction-diffusion example, the `forward()` method performs spatial discretization of the PDE model, and evaluates the time-derivative `du/dt` for the vector of voxel-wise cellularities, `u`.

**Coupled PDE systems:** Several scalar fields can be stacked as one state `(C, D, H, W)` (see `PDESystemModel3D` and `ImmuneResponse3D`). The same `TorchDiffEqSolver` applies. For plotting, total cell count, and least-squares calibration on **one** component (usually tumor, index `0`), use `tumortwin.pde_workflow` and the tutorial `tutorials/PDE_System_Demo.ipynb`.

To implement your own model, subclass `TumorGrowthModel3D` (or `PDESystemModel3D` for multi-field systems). Implement `forward(t, u)` returning the same tensor shape as `u`, using the `PatientData` and parameters you need.

Note that this model will be used in the `torchdiffeq` solver, and as such it supports discrete events via the `callback_step` and `callback_step_adjoint` methods which allow you to modify the ODE solution at each timestep of the forward and backward passes, respectively (see [`torchdiffeq` documentation](https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md#callbacks) for more details).

**Important:** if you wish to leverage `pytorch`-based automatic differentiation, you will need to ensure that your `forward()` method only leverages differentiable operations 

## Introducing new treatment models
Treatment effects are handled entirely within the `TumorGrowthModel`. Treatments effects that appear in the right-hand side of the model (i.e., they affect the `du/dt` term) can be implemented directly into the model's `forward()` method. Treatments that involve a discrete/instantaneous effect on the solution value can be integrated via the model's `callback_step()` method. Data specifying the treatment (e.g. dosages, times) would typically be provided in the patient configuration file, and passed into the model via the `PatientData` object, while parameters in the treatment model (e.g., treatment sensitivities) would appear directly in the model.

## Experimenting with different `Solver` algorithms
We provide an interface to the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library via the [`TorchDiffEqSolver`](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/tumortwin/solvers/torch_solver.py#L32) object. This allows you to use any of the solvers and solver options supported by `torchdiffeq` (see documentation [here](https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md#further-documentation))).

**Adaptive methods (e.g. `dopri5`):** `torchdiffeq` does not use `grid_constructor` for these solvers; our `TorchDiffEqSolver` instead builds the same time grid as for `rk4` (based on `step_size` and treatment schedules) and passes it as `step_t`, so `callback_step` still runs on radiotherapy/chemotherapy days. Tune accuracy with `TorchDiffEqSolverOptions.rtol` and `.atol`; optional integrator knobs (e.g. `max_num_steps`) go in `ode_options`.

If you wish to implement your own solver, you may wish to create a subclass of the `ForwardSolver` class and implement the `solve()` method.

## Experimenting with different `Optimizer` strategies
A `TumorGrowthModel3D` model should be directly compatible with `pytorch.optim` [`Optimizer`](https://pytorch.org/docs/stable/optim.html) objects. This provides access to a variety of different optimization algorithms. Alternatively, you might wish to refer to our custom [Levenberg-Marquardt optimizer](https://github.com/OncologyModelingGroup/TumorTwin/blob/36c21b45b526cd506d3421b509af813e1357b473/tumortwin/optimizers/lm_optimizer.py) to see how you might implement a custom optimizer.

## Need further guidance?
If you are still unsure about how you might extend `TumorTwin` to suit your needs, feel free to reach out to one of the authors. The best way to do this is by [opening a GitHub Issue](https://github.com/OncologyModelingGroup/TumorTwin/issues/new/choose).