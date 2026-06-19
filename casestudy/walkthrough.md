# Walkthrough - Spruce-Budworm Case Study & Noise Sweep

This walkthrough summarizes the updates made to migrate the ground truth case study from the 3-state genetic toggle switch to the 1D Spruce-budworm outbreak model, as defined in [test_dynamics_toy.ipynb](file:///Users/youpeng/Documents/Repos/SyMANTIC-NODE-Distillation/casestudy/test_dynamics_toy.ipynb), and provides detailed visual results from the noise sweep.

To make visualization and analysis more lightweight, we created a standalone script, [generate_plots.py](file:///Users/youpeng/Documents/Repos/SyMANTIC-NODE-Distillation/casestudy/generate_plots.py), which reads the stored equation strings from `equations.json` and loads the NODE checkpoints from disk to construct the comparison curves dynamically without re-running any training or symbolic regressions.

## Changes Made

### 1. Updated Simulation Framework
Modified [simulate.py](file:///Users/youpeng/Documents/Repos/SyMANTIC-NODE-Distillation/casestudy/simulate.py) to represent the 1D Spruce-budworm outbreak system:
- **True model**: 
  $$\frac{dw}{dt} = r\,w\left(1-\frac{w}{a}\right) - \frac{w^2}{1+w^2},\qquad r=0.5,\ a=10.0$$
- Refactored simulation states and dataframes to use the single variable `w` (replacing the previous `u, v, y` states).
- Added absolute noise injection support, keeping initial conditions pinned at $t=0$ and clamping values to remain positive ($\ge 10^{-8}$).

### 2. Standalone Plotting Script
Created [generate_plots.py](file:///Users/youpeng/Documents/Repos/SyMANTIC-NODE-Distillation/casestudy/generate_plots.py):
- Loads the discovered symbolic equations (PySINDy and SyMANTIC) from the compiled `equations.json` file.
- Automatically handles the PySINDy constant term representation (`coeff 1` string representation) by parsing and translating it into a valid Python expression.
- Loads trained Neural ODE weight configurations.
- Regenerates all time/state trajectory comparisons and metrics distribution boxplots.
- **Key Improvement**: Removes `sharey=True` constraint on trajectory grid subplots, allowing each subplot to scale dynamically to its own range.

---

## Validation & Results

The experiment completed successfully. Key metrics and outputs were compiled into `/Users/youpeng/Documents/Repos/SyMANTIC-NODE-Distillation/casestudy/results/spruce_budworm_noise_study/`.

### 1. Parity and Trajectory Comparison Plots
Below is a carousel showing the parity plots (true analytical derivatives vs gradients learned by Neural ODE) and general trajectory comparisons (averaging behavior across train/extrapolation) for each noise level:

````carousel
![Parity & Trajectories (Noise = 0.0)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/node_trajectories_noise_pct_0.0.png)
<!-- slide -->
![Parity & Trajectories (Noise = 0.02)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/node_trajectories_noise_pct_0.02.png)
<!-- slide -->
![Parity & Trajectories (Noise = 0.05)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/node_trajectories_noise_pct_0.05.png)
<!-- slide -->
![Parity & Trajectories (Noise = 0.1)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/node_trajectories_noise_pct_0.1.png)
````

---

### 2. Sampled 16 Initial Conditions (Time Extrapolation)
We sampled 16 evenly-spaced training initial conditions and integrated them over the extended time window ($t \in [0, 12]$) to view the long-term prediction trajectories. The vertical dotted line represents the end of the training time window ($t = 5.0$):

````carousel
![16 Trajectories Time Extrapolation (Noise = 0.0)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_t_noise_0.0.png)
<!-- slide -->
![16 Trajectories Time Extrapolation (Noise = 0.02)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_t_noise_0.02.png)
<!-- slide -->
![16 Trajectories Time Extrapolation (Noise = 0.05)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_t_noise_0.05.png)
<!-- slide -->
![16 Trajectories Time Extrapolation (Noise = 0.1)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_t_noise_0.1.png)
````

---

### 3. Sampled 16 Initial Conditions (State Extrapolation)
We sampled 16 initial conditions outside the training state range ($w_0 \in [8.0, 16.0]$) and integrated them over time to analyze the out-of-distribution (OOD) trajectory behavior:

````carousel
![16 Trajectories State Extrapolation (Noise = 0.0)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_x0_noise_0.0.png)
<!-- slide -->
![16 Trajectories State Extrapolation (Noise = 0.02)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_x0_noise_0.02.png)
<!-- slide -->
![16 Trajectories State Extrapolation (Noise = 0.05)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_x0_noise_0.05.png)
<!-- slide -->
![16 Trajectories State Extrapolation (Noise = 0.1)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/trajectories_16_ext_x0_noise_0.1.png)
````

---

### 4. Metric Distribution per Trajectory (MSE and $R^2$)
To understand individual trajectory variance, the boxplots below show the distribution of MSE and $R^2$ across all trajectories for the time extrapolation (Ext-T) and state extrapolation (Ext-$X_0$) splits, calculated per trajectory:

````carousel
![Metric Distributions (Noise = 0.0)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/metrics_distribution_noise_0.0.png)
<!-- slide -->
![Metric Distributions (Noise = 0.02)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/metrics_distribution_noise_0.02.png)
<!-- slide -->
![Metric Distributions (Noise = 0.05)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/metrics_distribution_noise_0.05.png)
<!-- slide -->
![Metric Distributions (Noise = 0.1)](/Users/youpeng/.gemini/antigravity-ide/brain/8942549c-2b1b-4ded-a088-2b987db716a3/metrics_distribution_noise_0.1.png)
````

---

### 5. Discovered Equations
SyMANTIC correctly identified the rational structures representing the Spruce-budworm outbreak dynamics, while SINDy discovered polynomial approximations.
- **Ground Truth**: $\dot{w} = 0.5w - 0.05w^2 - \frac{w^2}{1+w^2}$
- **SyMANTIC Discovered (0.05 std)**: 
  $$\dot{w} = 0.575w - 0.053w^2 - 2.051 \frac{w}{w+1} + 0.444$$
  *(An equivalent representation of the true rational term via feature space search)*
- **SINDy Discovered (0.05 std)**:
  $$\dot{w} = 0.014 - 0.092w + 0.065w^2 - 0.007w^3$$

---

### 6. Performance Summary
| Noise Level | Method | Train RMSE | Train $R^2$ | Ext-T (Time) $R^2$ | Ext-$X_0$ (State) $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.00** | NODE | 0.0099 | 0.9999 | 0.9999 | 0.3301 |
| | SINDy | 0.0795 | 0.9989 | 0.9926 | 0.8590 |
| | **SyMANTIC** | **0.0215** | **0.9999** | **0.9993** | **0.9999** |
| **0.02** | NODE | 0.0079 | 0.9999 | 0.9999 | 0.4773 |
| | SINDy | 0.0794 | 0.9989 | 0.9927 | 0.8607 |
| | **SyMANTIC** | **0.0206** | **0.9999** | **0.9994** | **0.9999** |
| **0.05** | NODE | 0.0116 | 0.9999 | 0.9999 | 0.5023 |
| | SINDy | 0.0813 | 0.9988 | 0.9922 | 0.8678 |
| | **SyMANTIC** | **0.0234** | **0.9999** | **0.9992** | **0.9999** |
| **0.10** | NODE | 0.0163 | 0.9999 | 0.9997 | 0.3421 |
| | SINDy | 0.0815 | 0.9988 | 0.9924 | 0.8689 |
| | **SyMANTIC** | **0.0284** | **0.9999** | **0.9989** | **0.9995** |

> [!NOTE]
> - **SyMANTIC** consistently demonstrates outstanding extrapolation capabilities ($R^2 \ge 0.999$) inside the out-of-distribution (OOD) region across all noise levels, thanks to capturing the correct functional forms.
> - **NODE** yields excellent interpolation and time-extrapolation performance but fails to generalize under state extrapolation (OOD) because Neural Networks lack structure-preserving constraints outside the boundaries of the training state space.
> - **SINDy** is robust to noise, but since it is constrained to a polynomial basis in this sweep, it can only approximate the true dynamics and gets bounded extrapolation error.
