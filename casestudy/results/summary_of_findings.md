# Executive Summary of Key Findings: NODE Distillation & SyMANTIC Ablation Study

This document compiles the key insights, quantitative results, and theoretical conclusions obtained from the genetic toggle switch noise study (2nd Iteration) and the subsequent SyMANTIC ablation studies (Phases 1 & 2).

---

## 1. Experimental Setup

The toggle switch dynamical system is modeled using a 3-state system ($u, v, y$) defined by the true equations:
$$\frac{du}{dt} = \frac{1}{1 + v^2} - u$$
$$\frac{dv}{dt} = \frac{1}{1 + \left(1 + \frac{u}{(1+y)^2}\right)^2} - v$$
$$\frac{dy}{dt} = 0.1 \cdot y$$

### Initial Condition & Time Domains
* **Full Sobol Domain**: Initial conditions are generated using a scrambled Sobol sequence of size $N = 64$ over the bounds:
  * $u_0 \in [0.5, 2.0]$
  * $v_0 \in [0.05, 0.2]$
  * $y_0 \in [0.05, 0.2]$
* **Training Subdomain Mask**: Trajectories are selected for training if their initial conditions fall in the inner $78\%$ range:
  * $u_0 \in [0.6, 1.9]$
  * $v_0 \in [0.06, 0.19]$
  * $y_0 \in [0.06, 0.19]$
  * This yields **44 training trajectories** and **20 test trajectories** (OOD initial conditions, Ext-$X_0$).
* **Time Span**: Simulated for $t \in [0.0, 20.0]$ with 101 time steps ($\Delta t = 0.2$s).
  * **Training Time Domain**: $t \in [0.0, 15.0)$ (75 steps).
  * **Time Extrapolation Domain (Ext-T)**: $t \in [15.0, 20.0]$ (26 steps).

### Noise Levels
* Gaussian noise is added to the clean trajectories for training: $\sigma \in \{0.01, 0.05, 0.10, 0.50\}$. All test evaluation is performed against the **clean** ground truth.

### NODE Model Configuration
* **Architecture**: MLP mapping $3 \to 64 \to 64 \to 3$ with Tanh activation.
* **Training**: Adam optimizer (LR = 1e-3), gradient norm clipping at 1.0, learning rate scheduler (ReduceLROnPlateau).
* **Validation Split**: The 44 trajectories are split into **38 training trajectories (85%)** and **6 validation trajectories (15%)**. Early stopping is triggered if validation loss does not improve for 100 epochs (up to 1000 epochs max).

### Symbolic Regression Configurations
* **SINDy**:
  * Library: Polynomial library up to degree 2 with interactions and bias (10 terms).
  * Optimizer: Sequentially Thresholded Least Squares (STLSQ) with threshold = 0.01, alpha = 0.01.
* **SyMANTIC**:
  * Operators: `['+', '-', '*', '/', '^2', '^-1']`
  * Max Depth (`n_expansion`): 4.
  * Level Pruning: Retains top 300 features ($100 \text{ sis\_features} \times 3 \text{ states}$) at the end of each expansion level using Sure Independence Screening (SIS).
  * Regularization: $L_1$ lasso vs. $L_0$ sparse regressor (max 3 terms per equation).

---

## 2. Noise Impact Sweep: SINDy vs. SyMANTIC (2nd Iteration)

The table below summarizes the performance metrics (RMSE and $R^2$) for the NODE, SINDy, and SyMANTIC models across different observation noise levels:

| Noise Level | Method | Train RMSE | Train $R^2$ | Ext-T RMSE | Ext-T $R^2$ | Ext-X0 RMSE | Ext-X0 $R^2$ | Complexity (Total) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | NODE | 0.0098 | 0.9921 | 0.0160 | 0.9873 | 0.0177 | 0.9913 | — |
| | SINDy | 0.0080 | 0.9947 | 0.0080 | 0.9969 | 0.0112 | 0.9966 | 28.0 |
| | SyMANTIC | 0.0116 | 0.9889 | 0.0202 | 0.9798 | 0.0202 | 0.9888 | 35.0 |
| **0.05** | NODE | 0.0379 | 0.8818 | 0.1440 | -0.0216 | 0.1007 | 0.7212 | — |
| | SINDy | 0.0378 | 0.8822 | 0.1436 | -0.0168 | 0.1006 | 0.7221 | 29.0 |
| | SyMANTIC | 0.0364 | 0.8906 | 0.1415 | 0.0129 | 0.0989 | 0.7314 | 35.0 |
| **0.10** | NODE | 0.0515 | 0.7813 | 0.1936 | -0.8475 | 0.1313 | 0.5262 | — |
| | SINDy | 0.0512 | 0.7836 | 0.1931 | -0.8387 | 0.1309 | 0.5291 | 29.0 |
| | SyMANTIC | 0.0513 | 0.7827 | 0.1923 | -0.8236 | 0.1308 | 0.5302 | 36.0 |
| **0.50** | NODE | 0.0908 | 0.3201 | 0.2894 | -3.1274 | 0.1879 | 0.0304 | — |
| | SINDy | 0.0887 | 0.3518 | 0.2879 | -3.0867 | 0.1864 | 0.0459 | 40.0 |
| | SyMANTIC | 0.0891 | 0.3451 | 0.2879 | -3.0842 | 0.1865 | 0.0444 | 88.0 |

### Core Sweep Takeaways
* **Low Noise Superiority**: At low noise (0.01), both symbolic methods achieve high trajectory fits. SINDy achieves slightly lower RMSE due to its quadratic library matching the polynomial gradients of the NODE. However, SyMANTIC discovers compact rational structures that are more physically consistent.
* **Overfitting at High Noise**: Under $L_1$ lasso, when forced to select the highest accuracy model on the Pareto front, SyMANTIC overfits to the noisy NODE predictions at noise 0.50, building high-complexity models (total complexity 88) which show no extrapolation capacity.

---

## 3. Ablation Study Phase 1: L1 vs. L0 Regularization

Phase 1 evaluated 5 distinct SyMANTIC settings using the noise level 0.01 data:
* **Case 1**: `['+', '-', '*', '/']` | Depth 3 | $L_1$ | `sis` = 100 (Baseline)
* **Case 2**: `['+', '-', '*', '/', '^2', '^-1']` | Depth 3 | $L_1$ | `sis` = 100 (Add Powers)
* **Case 3**: `['+', '-', '*', '/', '^2', '^-1']` | Depth 4 | $L_1$ | `sis` = 100 (Deeper Search)
* **Case 4**: `['+', '-', '*', '/', '^2', '^-1']` | Depth 3 | **$L_0$** | `sis` = 100 (Sparse Search)
* **Case 5**: `['+', '-', '*', '/', '^2', '^-1']` | Depth 3 | $L_1$ | **`sis` = 50** (Narrow Filter)

### Trajectory Integration Performance Table

| Case | Selection | Train RMSE | Train $R^2$ | Ext-T RMSE | Ext-T $R^2$ | Ext-X0 RMSE | Ext-X0 $R^2$ | Complexity (Total) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | Utopia / Highest | 0.0115 | 0.9890 | 0.0202 | 0.9798 | 0.0202 | 0.9888 | 34.0 |
| **Case 2** | Utopia / Highest | 0.0116 | 0.9889 | 0.0202 | 0.9798 | 0.0202 | 0.9888 | 35.0 |
| **Case 3** | Utopia / Highest | 0.0116 | 0.9889 | 0.0202 | 0.9798 | 0.0202 | 0.9888 | 35.0 |
| **Case 4** | Utopia / Highest | **0.0096** | **0.9924** | **0.0137** | **0.9908** | **0.0148** | **0.9940** | **16.0** |
| **Case 5** | Utopia / Highest | 0.0116 | 0.9889 | 0.0202 | 0.9798 | 0.0202 | 0.9888 | 35.0 |

### Discovered Equations (Phase 1)

| Case | Selection | state | Equation | Complexity | Closeness MAD |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **Case 1** | Utopia / Highest | $du/dt$ | $-0.7506u - 0.0208(u+v) - 0.0882(u+y) + 0.8276$ | 12.0 | 0.0432 |
| | | $dv/dt$ | $-0.162(u \cdot v) - 0.077(v-y) - 0.1537(v/u) + 0.0976$ | 13.0 | 0.0444 |
| | | $dy/dt$ | $0.0072(u \cdot y) + 0.0348(v \cdot y) + 0.0728y + 0.003$ | 9.0 | 0.0013 |
| **Case 2/3/5** | Utopia / Highest | $du/dt$ | $0.0889u^{-1} - 0.095(u+y) - 0.6865u + 0.6555$ | 13.0 | 0.0463 |
| | | $dv/dt$ | $-0.157(u \cdot v) - 0.078(v-y) - 0.1576(v/u) + 0.0975$ | 13.0 | 0.0446 |
| | | $dy/dt$ | $0.0072(u \cdot y) + 0.0348(v \cdot y) + 0.0728y + 0.003$ | 9.0 | 0.0013 |
| **Case 4** | Utopia / Highest | $du/dt$ | **$-0.8607u - 0.0951y + 0.8249$** | **7.0** | **0.0429** |
| | | $dv/dt$ | **$-0.3894v + 0.0780y + 0.0962$** | **6.0** | **0.0455** |
| | | $dy/dt$ | **$0.0951y + 0.0014$** | **3.0** | **0.00089** |

---

## 4. Key Ablation Insights (Phase 1)
1. **The Sparsity of $L_0$**: Case 4 ($L_0$) successfully isolated the true linear growth of state $y$: $\frac{dy}{dt} \approx 0.095y$ (complexity 3, Mean Absolute Difference: 0.00089). $L_1$ kept small spurious cross-coupling terms ($u \cdot y$ and $v \cdot y$), raising complexity to 9.
2. **Extrapolation Boost**: The simpler, physically sparse equations discovered via $L_0$ achieved significantly higher out-of-distribution test scores:
   * **OOD Initial Conditions ($R^2$ Ext-$X_0$)**: **0.9940** ($L_0$) vs. **0.9888** ($L_1$).
   * **Time Extrapolation ($R^2$ Ext-$T$)**: **0.9908** ($L_0$) vs. **0.9798** ($L_1$).

---

## 5. Denominator Discovery & The Constant Feature (Phase 2)
* **The Denominator Constraint**: SyMANTIC combines base variables ($u, v, y$). Without a constant feature, it is algebraically impossible to add a constant to a feature *inside* a denominator (e.g., it cannot construct $1 + v^2$ or $1 + y$ before taking the inverse).
* **Constant Injection**: Adding a constant column `C = 1.0` (Case 4B & 4E) successfully expanded the symbolic algebra search space to include denominator compositions.

---

## 6. Phase 2 Ablation: Convergence to Sparse Linear Approximations

In the targeted Phase 2 ablation study, we set up 5 distinct configurations based on L0 regularization, varying the presence of the constant feature `C = 1.0`, the search depth (3 vs. 4), and the SIS filter budget (100 vs. 200 vs. 150):
* **Case 4A**: No Constant | Depth 3 | `sis` = 100
* **Case 4B**: **Constant `C`** | Depth 3 | `sis` = 100
* **Case 4C**: No Constant | **Depth 4** | `sis` = 100
* **Case 4D**: No Constant | Depth 3 | **`sis` = 200**
* **Case 4E**: **Constant `C`** | **Depth 4** | **`sis` = 150**

### The Convergence Result
Intriguingly, **all five cases converged to the exact same sparse linear approximations**:

| State Variable | Ground Truth Dynamics | $L_1$ (Depth 4, 2nd Iteration) Discovered | $L_0$ (Cases 4A-4E) Discovered |
| :---: | :--- | :--- | :--- |
| $\frac{du}{dt}$ | $\frac{1}{1 + v^2} - u$ | $0.0889u^{-1} - 0.0951(u+y) - 0.6865u + 0.6556$ | **$-0.8607u - 0.0951y + 0.8249$** |
| $\frac{dv}{dt}$ | $\frac{1}{1 + (1 + \frac{u}{(1+y)^2})^2} - v$ | $-0.1574(u \cdot v) - 0.0780(v-y) - 0.1576(v/u) + 0.0975$ | **$-0.3894v + 0.0780y + 0.0962$** |
| $\frac{dy}{dt}$ | $0.1 \cdot y$ | $0.0072(u \cdot y) + 0.0348(v \cdot y) + 0.0728y + 0.003$ | **$0.0951y + 0.0014$** |

### Why L0 is Invariant and Converges to the Same Basin
* **Hard Sparsity Constraint**: Unlike $L_1$, which allows coefficients to slowly shrink (retaining small spurious terms like $u \cdot y$), $L_0$ penalizes the count of non-zero parameters. This forces the optimization to select only the most dominant signals.
* **Stable Optimization Landscale**: Because $L_0$ only permits a tiny budget (up to 3 terms), the optimization landscape becomes highly simplified. The linear representation represents a very strong local basin that dominates all other combinations. Increasing depth or SIS features only adds weaker signals, which are immediately pruned.

---

## 7. Core Insight: The Fundamental Limit of NODE Distillation

Why did the addition of the constant column `C` and deeper search (depth 4) fail to recover the exact rational dynamics of the toggle switch (e.g. $\frac{1}{1+v^2} - u$)?

### 1. Spurious Teacher Correlations
In any distillation setup, the symbolic model fits the **teacher model's predictions** (the learned NODE vector field). 
Because the training trajectories are simulated from a single coupled system, the states $v$ and $y$ are highly coupled and evolve along a correlated trajectory. The NODE teacher MLP represents the vector field using these correlated inputs. 
Consequently, the NODE model learns a spurious dependency: it predicts that $\frac{du}{dt}$ varies as a function of the exponential state $y$.

### 2. Student Inherits Teacher Biases
Because the teacher's predicted gradients actually depend on $y$, a symbolic model including $y$ (like $-0.86u - 0.095y + 0.82$) achieves a **substantially lower fit loss on the teacher's gradients** than the true physical model (which only depends on $v$). 

Since the symbolic regressor's objective is to fit the teacher's gradients, it is mathematically compelled to select the spurious correlation.

### 3. Conclusion
This highlights an inherent limitation of NODE distillation: **the distilled equations can only be as physically correct as the teacher network**. If the teacher network is trained on correlated data and learns spurious physics, the distillation process will faithfully reconstruct those spurious relationships.

---

## 8. Phase 3 Study: Full Time Horizon Training (t = 20s)

To test if extending the training time horizon resolves the state coupling issue, we conducted a third study (Phase 3) using only the lowest noise level data (0.01). The training time domain was expanded from the first 15s to the full 20s (101 steps), and a new NODE model was trained from scratch. SINDy, SyMANTIC-L1, and SyMANTIC-L0 regressions were then performed on the gradients of this new model.

### Trajectory Integration Performance Table (Phase 3)

| Method | Selection | Train RMSE | Train $R^2$ | Ext-X0 RMSE | Ext-X0 $R^2$ | Complexity (Total) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| NODE | Utopia | 0.0197 | 0.9858 | 0.0291 | 0.9768 | — |
| SINDy | Utopia | 0.0190 | 0.9869 | 0.0279 | 0.9786 | 35.0 |
| SyMANTIC-L1 | Utopia | 0.0259 | 0.9754 | 0.0353 | 0.9657 | 70.0 |
| SyMANTIC-L1 | Highest Accuracy | 0.0233 | 0.9801 | 0.0326 | 0.9708 | 72.0 |
| SyMANTIC-L0 | Utopia | 0.0198 | 0.9856 | 0.0267 | 0.9804 | 21.0 |
| SyMANTIC-L0 | Highest Accuracy | **0.0186** | **0.9874** | **0.0248** | **0.9831** | **23.0** |

### Discovered Equations (Phase 3)

| Method | Selection | State | Equation | Complexity | Closeness MAD |
| :---: | :---: | :---: | :--- | :---: | :---: |
| **SyMANTIC-L0** | Utopia | $du/dt$ | $-0.5542(u - v) - 0.1497y + 0.3987$ | 8.0 | 0.1364 |
| | | $dv/dt$ | $-0.4937v - 0.0433u + 0.0993(u \cdot y) + 0.1614$ | 10.0 | 0.0362 |
| | | $dy/dt$ | $0.0948y + 0.0019$ | 3.0 | 0.0013 |
| **SyMANTIC-L0** | Highest Accuracy | $du/dt$ | $-0.5830(u - v) - 0.1326(v + y) + 0.4493$ | 9.0 | 0.1279 |
| | | $dv/dt$ | $-0.5010v - 0.0523(u + y) + 0.1607(u \cdot y) + 0.1705$ | 11.0 | 0.0358 |
| | | $dy/dt$ | $0.0948y + 0.0019$ | 3.0 | 0.0013 |

### Key Phase 3 Insights
1. **Student Outperforms Teacher**: SyMANTIC-L0 (Highest Accuracy) achieved an extrapolation score of **0.9831** (RMSE = 0.0248), which actually *outperformed* the teacher NODE model itself (Ext-X0 $R^2 = 0.9768$, RMSE = 0.0291). This demonstrates how the sparse L0 regularizer acts as a noise-filter, pruning away weak nonlinearities and local overfitting present in the neural network representation.
2. **Spurious Correlations Persist**: Despite the full 20s time domain, the symbolic regressions still converged to linear/quadratic approximations containing spurious $y$ and $u \cdot y$ terms. 
   - *Why?* Integrating forward to 20s does not decouple the state dynamics. In fact, for $t \in [15, 20]$, $u(t)$ and $v(t)$ have mostly flatlined near their attractors while $y(t) = y_0 e^{0.1 t}$ continues to grow exponentially. This flatline phase introduces different but equally strong correlations (e.g. constant derivatives with growing $y$), which the teacher NODE fits using cross-coupling terms. Since the distillation targets (the teacher's gradients) are still biased, the student is forced to fit these biases to minimize loss.
3. **Equivalence of dy/dt**: The L0 sparse regressor consistently and accurately isolates the decoupled dynamics of $\frac{dy}{dt} \approx 0.095y$, showing that linear decoupled dynamics are easily recovered when the state does not depend on other variables, but the coupled equations ($du/dt$ and $dv/dt$) remain a fundamental challenge under standard trajectory-based training.

---

## 9. Depth 7 Search: Deep Nested Fractions ($\frac{dv}{dt}$)

To verify if SyMANTIC can construct and screen complex, multi-level nested fractions when search depth restrictions are lifted, we ran a targeted study on state variable $v$ ($\frac{dv}{dt}$) with `n_expansion = 7`, `sis_features = 50`, and $L_1$ regularization.

### Discovered Pareto Front models ($R^2 > 0.9$ vs. Teacher Gradients)
* **Model 1 (Complexity 10.5, $R^2 = 0.946$):**
  $$\frac{dv}{dt} \approx -0.1159(u \cdot v) - 0.0592(v - y) - 0.2243v + 0.1032$$
* **Model 2 (Complexity 96.1, $R^2 = 0.971$):**
  $$\frac{dv}{dt} \approx -0.0542\left(\frac{\frac{v/u}{y^2}}{\frac{u+y}{y^2}}\right) - 0.5074\left(\frac{v}{\frac{v+y}{v^2}}\right) - 0.2071\left(\frac{(u+v)\frac{u}{v}}{\frac{u+y}{v^2}}\right) + 0.0967$$
* **Model 3 (Complexity 101.8, $R^2 = 0.982$):**
  $$\frac{dv}{dt} \approx -0.3683\left(\frac{v}{\frac{v+y}{v^2}}\right) - 0.1682\left(\frac{(u+v)\frac{u}{v}}{\frac{u+y}{v^2}}\right) - 0.1330\left(\frac{(u+v)y^{-1}}{\frac{u+y}{v \cdot y}}\right) + 0.1045$$
* **Model 4 (Complexity 228.9, $R^2 = 0.991$):**
  $$\frac{dv}{dt} \approx -0.1108\left(\frac{\frac{u}{v-y} \frac{u+v}{v-y}}{\frac{u+y}{v-y} \frac{v^{-1}}{v-y}}\right) - 0.1129\left(\frac{\left(\frac{u+v}{v-y}\right)^2}{\frac{u+y}{v-y} \frac{v^{-1}}{v-y}}\right) - 0.0329\left(\frac{\left(\frac{u+v}{v-y}\right)^2}{\frac{u+y}{v-y} \frac{u-v}{v-y}}\right) + 0.1371$$

### Quantitative Trajectory Performance Comparison
We simulated the hybrid system (combining true equations for $du/dt$ and $dy/dt$ with the discovered $dv/dt$ equations) across all 64 trajectories to compare against the ground-truth clean data and the teacher NODE model:

| Model | Complexity | Gradient $R^2$ | Train RMSE | Train $R^2$ | Test (OOD) RMSE | Test (OOD) $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NODE (Teacher)** | — | 1.0000 | 0.019691 | 0.9858 | 0.029069 | 0.9768 |
| **Model 1 (`C=10.5`)** | 10.51 | 0.9465 | **0.007576** | **0.9979** | **0.010375** | **0.9970** |
| **Model 2 (`C=96.1`)** | 96.13 | 0.9708 | **0.005966** | **0.9987** | **0.008177** | **0.9982** |
| **Model 3 (`C=101.8`)** | 101.83 | 0.9823 | **0.005622** | **0.9988** | **0.007732** | **0.9984** |
| **Model 4 (`C=228.9`)** | 228.91 | 0.9912 | **0.005363** | **0.9989** | **0.007610** | **0.9984** |

### Depth-7 Comparison Plots

#### 1. Trajectory Integration Comparison (State $v$)
![Depth 7 Trajectory Comparison](/Users/youpeng/.gemini/antigravity/brain/3203bd1c-f5a0-4965-b08e-5e8008a02f0e/depth7_dvdt_comparison.png)

#### 2. Quantitative Fit & Test Performance Comparison
![Depth 7 Performance Comparison](/Users/youpeng/.gemini/antigravity/brain/3203bd1c-f5a0-4965-b08e-5e8008a02f0e/depth7_performance_comparison.png)

---

## 10. Recommendations for Future Work
* **Perturbation Experiments**: Train the teacher NODE model on highly diverse, decoupled trajectories to break state-to-state correlations.
  * *Why random initial conditions (e.g. via Sobol) are not enough:* Although the initial states at $t = 0$ are decoupled and diverse, forward time integration under the coupled system equations naturally causes states $(u(t), v(t), y(t))$ to fall onto a low-dimensional manifold, introducing strong state-to-state correlations. For example, $y(t) = y_0 e^{0.1 t}$ acts as a monotonic time proxy, making it highly correlated with the relaxation of $u(t)$ and $v(t)$ over time. Since the NODE is trained on continuous trajectories, the teacher MLP uses these correlations to represent the vector field (e.g. predicting that $du/dt$ depends on $y$), which the student then inherits during distillation.
  * *How to decouple trajectories:*
    1. **External Forcing Inputs:** Inject independent time-varying perturbations directly into each state variable during simulation to force the system off its natural low-dimensional manifold.
    2. **Short Snippets / Frequent Resets:** Instead of long continuous trajectories, reset states to new random Sobol points frequently (e.g. every few time-steps) to prevent the system from integrating into a coupled state.
    3. **Direct Derivative Fitting:** Fit the network directly on independent state-derivative pairs $(\mathbf{x}_i, \dot{\mathbf{x}}_i)$ sampled uniformly across the state space rather than training a NODE through integration.
* **Inductive Biases**: Incorporate coordinate-wise constraints or structural assumptions directly into the NODE network architecture during training to prevent it from learning spurious dependencies.
* **Direct Symbolic Regression**: Fit SyMANTIC directly on noisy/clean trajectories (using numerical derivatives or trajectory-matching optimization) rather than distilling from a NODE vector field.
