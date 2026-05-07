# HemoInvasion3D Parameter Calibration Analysis

## Problem Statement

The current HemoInvasion3D model parameters for patient `HGG_demo_001` give SSE ≈ 2783, which is unacceptably high. This document analyzes the root causes and proposes solutions.

## 1. Root Cause Analysis

### 1.1 The Fundamental Issue: No Treatment Terms

**The HemoInvasion3D model does NOT include radiotherapy (RT) or chemotherapy (CT) terms.**

From `hemo_invasion_3d.py`:
```python
self.radiotherapy_specification = None
self.chemotherapy_specifications = None
```

The model only describes natural tumor growth dynamics:
- `dn/dt = B*n*(1-n) - P(s)*n + div(Dn*(1-n)*grad(n)) - <grad(n), grad(phi)>`
- `dm/dt = P(s)*n - m*B*n - div(Dn*m*grad(n)) - <grad(m), grad(phi)>`
- `ds/dt = Ds*Δs - k_s*n*s/(s+s_star)`
- `Δphi = B*n`

But the patient `HGG_demo_001` undergoes **intensive chemoradiation** starting at day ~26:
- **29 fractions of RT** (2 Gy each, days 26-75)
- **Concurrent temozolomide** (105 mg daily during RT, then cycles of 215 mg)

This treatment causes **dramatic TTC reduction**:

| Visit | Day | TTC | Interpretation |
|-------|-----|------|---------------|
| 0 | 0 | 5917 | Pre-treatment baseline |
| 1 | 30 | 6305 | Early during treatment (slight growth) |
| 2 | 60 | 1060 | **Post-treatment nadir (−83% from peak)** |
| 3 | 90 | 1863 | Early regrowth |
| 4-9 | 120-270 | 6410→54451 | Aggressive regrowth |

**A pure growth model simply cannot fit the treatment-induced 83% TTC drop at day 60.** This is the primary source of high SSE when calibrating on visits that span the treatment period.

### 1.2 The time_scale_days Scaling Issue

The `forward()` output is divided by `time_scale_days` (default=120 in Hemo_Demo). This means the **effective** per-day rates are:

| Parameter | Model value | Effective rate (per day) |
|-----------|-------------|------------------------|
| B=0.010 | Growth rate | 0.010/120 = 8.3×10⁻⁵ /day |
| Dn=0.001 | Diffusion | 0.001/120 = 8.3×10⁻⁶ mm²/day |
| Ds=0.015 | Substrate diff | 0.015/120 = 1.25×10⁻⁴ mm²/day |

Compare to **literature values** for HGG:
- **Proliferation rate**: 2.73×10⁻⁴ to 2.73×10⁻² /day ([Nature 2024](https://www.nature.com/articles/s41540-024-00478-7/tables/1))
- **Diffusion coefficient**: 2.73×10⁻³ to 2.73×10⁻¹ mm²/day
- **Hormuth et al. (2021)**: D ≈ 10.65-19.38 × 10⁻³ mm²/day (murine glioma)

The current effective rates are **1-2 orders of magnitude too slow**. With B_eff = 8.3×10⁻⁵/day, the logistic doubling time would be ~8300 days (23 years), whereas typical HGG doubling times are 20-100 days.

## 2. Recommended Solutions

### Strategy A: Calibrate Only on Growth Phase (Post-Treatment, Visits 2→9)

Use visit 2 (day 60, TTC=1060) as the initial condition and calibrate on visits 3-9. This avoids the treatment period entirely and lets the model focus on what it can actually represent: **tumor regrowth**.

```python
# Re-initialize from visit 2 (post-treatment nadir)
visit2 = patient_data.visits[2]
cellularity2 = ADC_to_cellularity(visit2.adc_image, visit2.roi_enhance_image, visit2.roi_nonenhance_image)
initial_n = torch.from_numpy(cellularity2.array).float().to(device)

# Use visits 2-9 for calibration
target_timepoints = [v.time for v in patient_data.visits[2:]]
y_target = torch.stack(measured_maps[2:], dim=0)
```

**Recommended parameters for post-treatment regrowth** (based on literature + model scaling):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `B` | 0.04–0.08 | Higher growth rate to match aggressive post-treatment regrowth |
| `Dn` | 0.005–0.015 | Eff. D ≈ 4–12 ×10⁻⁵ mm²/day; moderate invasion |
| `Ds` | 0.02–0.05 | Nutrient diffusion, order of magnitude above Dn |
| `k_s` | 0.1–0.3 | Nutrient consumption, accelerated near large tumor masses |
| `s_star` | 0.1–0.3 | Half-saturation for nutrient uptake |
| `s_crit` | 0.2–0.5 | Quiescence transition threshold |
| `time_scale_days` | 30.0 | **Reduce from 120 to 30** to bring effective rates into biological range |

### Strategy B: Reduce time_scale_days Dramatically

The `time_scale_days=120` was likely set for numerical stability but results in biologically unrealistic effective rates. Reducing to 30 (the code default) or even lower amplifies the rates by 4×:

| time_scale_days | B=0.04 effective rate | Doubling time |
|-----------------|----------------------|---------------|
| 120 | 3.3×10⁻⁴ /day | ~2100 days |
| 30 | 1.3×10⁻³ /day | ~530 days |
| 10 | 4.0×10⁻³ /day | ~173 days |
| 1 | 0.04 /day | ~17 days |

For **time_scale_days=1** (no rescaling), B=0.04 gives a doubling time of 17 days which is within the biological range for HGG.

**However**, reducing time_scale_days may require reducing the solver `step_size` to maintain numerical stability.

### Strategy C: Calibrate on Pre-Treatment Visits Only (0→1)

If you need SSE to be low and want to keep things simple, calibrate only on visits 0 and 1 (days 0-30):

```python
target_timepoints = [v.time for v in patient_data.visits[:2]]
y_target = torch.stack(measured_maps[:2], dim=0)
```

With the default parameters (B=0.010, Dn=0.001, etc.), the **2-visit SSE ≈ 289** (measured with euler/1-day step). This is already much lower because the model doesn't need to explain the treatment dip.

### Strategy D: Wider Parameter Bounds for Grid Search + LM

The notebook calibration bounds are too narrow. Recommended wider bounds:

```python
# 3-parameter calibration [B, Dn, k_s]
bounds_3p = torch.tensor([
    [0.003, 0.200],   # B (was [0.005, 0.080])
    [0.0002, 0.050],  # Dn (was [0.0005, 0.020])
    [0.005, 0.500],   # k_s (was [0.020, 0.300])
], dtype=torch.float64)

# 5-parameter calibration [B, Dn, Ds, k_s, s_crit]
bounds_5p = torch.tensor([
    [0.003, 0.200],   # B
    [0.0002, 0.050],  # Dn
    [0.001, 0.100],   # Ds (was [0.002, 0.060])
    [0.005, 0.500],   # k_s
    [0.05, 0.80],     # s_crit (was [0.150, 0.650])
], dtype=torch.float64)
```

### Strategy E: Add Treatment Terms to HemoInvasion3D

The most correct approach is to extend `HemoInvasion3D` to include RT/CT terms, similar to `ReactionDiffusion3D`:

```python
# In forward(), add after growth terms:
# RT: instantaneous cell kill (handled by solver grid constructor events)
# CT: exponential decay term
#   -alpha_ct * C * exp(-beta_ct * (t - t_admin)) * n
```

This requires modifying `hemo_invasion_3d.py` to inherit the treatment mechanism from the base `ReactionDiffusion3D` model. This is the only approach that can properly fit the full 10-visit trajectory.

## 3. Literature Parameter Reference

### From Nature (2024) - Glioma Reaction-Diffusion Model
| Parameter | Value Range | Units |
|-----------|-------------|-------|
| Cell diffusivity (D) | 2.73×10⁻³ – 2.73×10⁻¹ | mm²/day |
| Proliferation rate (b) | 2.73×10⁻⁴ – 2.73×10⁻² | day⁻¹ |
| Carrying capacity (K) | 10² | cells/mm |
| Oxygen diffusion | 1.51×10² | mm²/day |

### From Hormuth et al. (IEEE TBME 2021) - Murine Glioma
| Parameter | RDAM | RD | RDM | Units |
|-----------|------|-----|-----|-------|
| D (median) | 10.65×10⁻³ | 17.46×10⁻³ | 19.38×10⁻³ | mm²/day |
| k (initial guess) | 2.5 | 2.5 | 2.5 | day⁻¹ |

### From Hormuth et al. (Sci Reports 2021) - Clinical HGG
| Metric | Value |
|--------|-------|
| Enhancing volume error | median −2.5% |
| TTC correlation | Kendall τ = 0.79 |
| CCC (continuous calibration) | 0.91 |

### From Swanson et al. - Classical HGG Parameter Estimates
| Parameter | Low-grade | High-grade | Units |
|-----------|-----------|------------|-------|
| D | 1–5 | 5–50 | mm²/year |
| D | 0.003–0.014 | 0.014–0.137 | mm²/day |
| ρ (proliferation) | 0.001–0.01 | 0.01–0.1 | day⁻¹ |

## 4. Recommended Parameter Sets to Try

### Set 1: Conservative growth (pre-treatment focus, visits 0-1)
```python
B=0.010, Dn=0.001, Ds=0.015, k_s=0.040, s_star=0.250
time_scale_days=120.0, step_size=0.02 days, method="rk4"
# Expected SSE (2 visits): ~200-300
```

### Set 2: Moderate growth (post-treatment regrowth, visits 2-9)
```python
B=0.060, Dn=0.008, Ds=0.030, k_s=0.150, s_star=0.200
time_scale_days=30.0, step_size=0.02 days, method="rk4"
s_crit=0.25, s_smooth=0.05
# Use visit 2 as initial condition
```

### Set 3: Aggressive growth (capturing rapid regrowth)
```python
B=0.080, Dn=0.015, Ds=0.050, k_s=0.250, s_star=0.150
time_scale_days=30.0, step_size=0.01 days, method="rk4"
s_crit=0.20, s_smooth=0.05
# Use visit 2 as initial condition
```

### Set 4: Biologically-motivated (no time rescaling)
```python
B=0.020, Dn=0.010, Ds=0.050, k_s=0.100, s_star=0.200
time_scale_days=1.0, step_size=0.001 days, method="rk4"
s_crit=0.30, s_smooth=0.05
# Warning: may need very small step_size for stability
```

## 5. Summary of Recommendations

1. **Primary recommendation**: Re-initialize from visit 2 (post-treatment) and calibrate on visits 2-9. This sidesteps the treatment modeling gap entirely.

2. **Reduce `time_scale_days`** from 120 to 30 (or lower) to bring effective rates into the biological range.

3. **Widen calibration bounds** substantially, especially for B (up to 0.2) and k_s (up to 0.5).

4. **Increase LM iterations** from 6-8 to 20-30 for better convergence.

5. **Long-term**: Extend `HemoInvasion3D` to include RT/CT treatment terms. This is the only way to properly fit the full treatment trajectory.
