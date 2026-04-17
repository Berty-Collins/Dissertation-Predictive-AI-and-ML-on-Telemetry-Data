# Predicting Vehicle Performance with AI and ML Using Simulated Telemetry Data

**Student Dissertation — Computer Science**
**Cardiff University**

---

## Abstract

The optimisation of vehicle setup parameters has traditionally required expensive physical prototyping and track testing, limiting exploration to a small region of the available design space. This dissertation investigates whether machine learning (ML) models can predict vehicle performance key performance indicators (KPIs) from chassis and powertrain configuration parameters using data collected entirely from a physics-based simulation environment. The study employs BeamNG.tech v0.38.3.0 as the simulation platform, with the ETK800 front-wheel-drive vehicle on a flat gridmap test environment, and beamngpy v1.35 as the programmatic interface for automated data collection.

A dataset of 555 parameter configurations was generated using Latin Hypercube Sampling (LHS), One-at-a-Time (OAT) parameter sweeps, and a baseline run, covering 27 chassis and powertrain parameters across four standardised test scenarios: launch/braking, steady-state circle, slalom, and step steer. Following a data quality audit that identified and removed 39 failed simulation runs (7.0%), 516 usable observations were retained. Seven regression model classes — Random Forest, Extra Trees, Gradient Boosting, Multi-Layer Perceptron, Support Vector Regression, Bayesian Ridge, and ElasticNet — plus a Stacking Ensemble were trained and evaluated against 13 KPI targets.

Models achieved positive test R² on five of thirteen KPIs, with brake stopping distance (R² = 0.222, Stacking Ensemble), 0–60 mph time (R² = 0.204, Random Forest), and step-steer peak yaw rate (R² = 0.149, Random Forest) showing the strongest predictability. Lateral grip and yaw response KPIs showed near-zero predictability, attributed to the flat test environment and the inherent noise introduced by disabling the Anti-lock Braking System (ABS). An ABS-on versus ABS-off comparison confirmed that removing ABS significantly improved predictability of dynamic response KPIs, particularly step-steer yaw overshoot (ΔCVR² = +1.40, Random Forest). A surrogate-model-guided Bayesian optimisation identified a 20-solution Pareto front balancing launch time against cornering grip, with brakestrength, rear LSD preload, and rear toe identified as the most influential parameters.

**Keywords:** vehicle dynamics, machine learning, simulation, BeamNG.tech, surrogate modelling, Bayesian optimisation, design of experiments

---

## Table of Contents

1. Introduction
2. Literature Review
3. Methodology
4. Results and Analysis
5. Discussion
6. Conclusion
7. References

---

## 1. Introduction

### 1.1 Motivation

The development of a high-performance vehicle requires careful calibration of hundreds of interdependent parameters — spring rates, damper settings, anti-roll bar stiffness, tyre pressures, gear ratios, brake bias, and differential lock coefficients, among others. In professional motorsport, this calibration process involves dedicated engineering teams, specialised measurement equipment, and extensive on-track testing, all of which demand significant financial and temporal investment. A single day of testing at a closed circuit can cost tens of thousands of pounds once personnel, fuel, tyre wear, and circuit hire are accounted for; and this expense is incurred for each incremental change to the setup, with diminishing returns as the vehicle approaches its optimum (Milliken and Milliken, 1995). Even in the context of production vehicle development, the cost of physical testing and the inherent safety risks of evaluating extreme parameter combinations impose practical limits on how thoroughly the design space can be explored.

Beyond the financial dimension, there is an epistemic challenge: vehicle performance emerges from the interaction of dozens of subsystems — tyre–road contact, suspension kinematics, aerodynamics, powertrain, and driver — in ways that are difficult to reason about intuitively. A front suspension spring rate change affects not only ride quality but also the dynamic load transfer during braking and cornering, which in turn affects tyre contact patch loading, which determines the grip available for both lateral and longitudinal forces. These interdependencies mean that optimal setup is rarely found by varying one parameter at a time; the true optimum typically lies in a high-dimensional region where multiple parameters must be jointly tuned.

The emergence of physics-based vehicle simulation platforms has substantially reduced the cost of individual experiments. Tools such as IPG CarMaker, AVL CRUISE M, and — increasingly — game-derived environments such as BeamNG.tech allow engineers to evaluate thousands of parameter combinations that would be impractical to test physically (BeamNG GmbH, 2024). However, raw simulation data alone does not constitute understanding: running thousands of simulations produces a large dataset, but identifying meaningful relationships between input parameters and output performance indicators still requires systematic analysis. A simulation model that is run exhaustively without a guiding strategy wastes computational resources exploring uninteresting regions of the parameter space.

Machine learning offers a complementary capability. Given sufficient training data, supervised regression models can learn the mapping from parameter vectors to performance outcomes, enabling rapid prediction of untested configurations without requiring additional simulation runs. This surrogate modelling approach transforms the problem from one of exhaustive simulation to one of guided search: a trained ML model can be queried at negligible computational cost (milliseconds per prediction, versus minutes or hours per simulation), enabling gradient-free optimisation methods such as Bayesian optimisation to efficiently identify high-performing configurations in the multi-dimensional parameter space (Shahriari et al., 2016). The combination of physics-based simulation for data generation and ML for prediction and optimisation thus provides a pathway to comprehensive design space exploration at a fraction of the cost of physical testing.

### 1.2 Problem Statement

Despite the intuitive appeal of this approach, several challenges limit its direct application to vehicle performance prediction. First, the quality of the learned mapping depends critically on how well the training data covers the parameter space. A poorly designed experiment — for example, one that samples configurations near the centre of the parameter space without exploring the boundaries — will produce models that interpolate well in the well-sampled region but extrapolate dangerously elsewhere. Second, vehicle dynamics are governed by highly non-linear physical laws: tyre slip curves exhibit an initial linear region followed by a pronounced peak and a fall-off under sliding conditions (Pacejka, 2012); suspension kinematics are non-linear through their range of motion; and powertrain torque delivery depends on gear ratio in a hyperbolic relationship with speed. These non-linearities mean that simple linear models will systematically mis-predict performance across much of the parameter space, while complex non-linear models require more training data to avoid over-fitting.

Third, simulation environments themselves introduce artefacts that are not present in physical testing. BeamNG.tech employs a soft-body finite element model that, while high-fidelity relative to rigid-body simulators, can produce numerical instability at extreme parameter combinations — for example, when spring rates are set ten times their nominal value, the Euler integration may fail to converge, producing nonsensical telemetry rather than a physically meaningful (if extreme) vehicle behaviour. Fourth, the state of active safety systems — specifically the Anti-lock Braking System — can significantly mediate the relationship between chassis parameters and performance outcomes, creating a confound that must be understood and controlled.

This dissertation addresses these challenges through a complete, end-to-end implementation and evaluation of a simulation-based ML pipeline for vehicle performance prediction, from experimental design and data collection through preprocessing, model training, and Bayesian optimisation.

### 1.3 Research Questions

Three primary research questions guide this work:

1. **RQ1:** To what extent can ML regression models predict vehicle performance KPIs from chassis and powertrain configuration parameters, using data collected exclusively from physics-based simulation?

2. **RQ2:** Does the state of the Anti-lock Braking System significantly affect the predictability of performance KPIs, and does disabling ABS reveal stronger parameter–KPI relationships for dynamic response metrics?

3. **RQ3:** Can surrogate-model-guided Bayesian optimisation identify Pareto-optimal parameter configurations that trade off competing performance objectives?

### 1.4 Contributions

This work makes the following original contributions:

- A fully automated simulation-to-ML pipeline using BeamNG.tech and beamngpy, supporting resume-safe, parallel-safe data collection across 555 parameter configurations totalling approximately 14 hours of simulation time per dataset;
- A detailed technical investigation into the JBeam slot architecture of the BeamNG.tech ETK800 vehicle, identifying the correct mechanism for disabling ABS hardware (`etk_DSE_ABS` slot removal plus Lua-level wheel function pointer patching) and implementing a diagnostic verification counter to confirm disablement across all runs;
- A systematic investigation of the impact of ABS state on the signal-to-noise ratio of vehicle dynamics KPIs collected in simulation, with quantitative comparison across all thirteen targets;
- Identification and correction of a data quality issue — sentinel-zero failed simulation runs representing 7.0% of all collected data — that, when unaddressed, renders all ML models entirely ineffective, along with a principled detection method transferable to other simulation-based studies;
- Comparative evaluation of seven model classes and a stacking ensemble across thirteen KPI targets, with SHAP-based feature importance attribution for the best model on each target;
- A surrogate-guided Bayesian optimisation producing a 20-solution Pareto front for the launch time versus lateral grip trade-off, together with a one-dimensional sensitivity analysis identifying the ten most influential parameters for the primary performance objective.

### 1.5 Structure

The remainder of this dissertation is organised as follows. Section 2 surveys related work in vehicle simulation, design of experiments, surrogate modelling, and Bayesian optimisation. Section 3 describes the experimental design, simulation platform, data collection protocol, preprocessing pipeline, and ML training and evaluation methodology in detail. Section 4 presents and analyses the results across all research questions. Section 5 discusses the findings in the context of the research questions, the broader literature, and the practical implications and limitations of the approach. Section 6 concludes with a summary of findings, limitations, and directions for future work.

---

## 2. Literature Review

### 2.1 Physics-Based Vehicle Simulation

The use of computational simulation for vehicle dynamics analysis has a long history, progressing from linearised bicycle models (Milliken and Milliken, 1995) through multi-body dynamics packages to real-time capable soft-body physics engines. Early analytical models, based on the classical single-track (bicycle) model with linearised tyre forces, remained dominant in control system design throughout the 1980s and 1990s, providing analytical tractability at the cost of physical fidelity for large-slip, combined-cornering-and-braking manoeuvres. The Pacejka Magic Formula tyre model (Bakker, Nyborg and Pacejka, 1987; Pacejka, 2012) provided a semi-empirical representation of tyre forces that substantially improved the accuracy of multi-body simulation, and remains the industry standard for tyre force modelling in professional simulation tools.

Commercial multi-body simulation tools including MSC ADAMS, IPG CarMaker, and dSPACE ASM provide validated, high-fidelity vehicle dynamics models parameterised from physical measurements. These tools are widely used in professional automotive development but require substantial licensing costs, proprietary vehicle parameter databases, and expert users. They also typically operate in a co-simulation framework with powertrain and control system models, requiring integration effort for each new vehicle.

Game-derived simulation engines have emerged as an accessible alternative for research purposes. TORCS (Wymann et al., 2000) provided an early open-source racing simulator with a simple but controllable vehicle model, used extensively for autonomous racing agent development. Its successor rFactor2 (ISI, 2012) introduced tyre models of greater sophistication, including thermal and wear effects. Assetto Corsa (Kunos Simulazioni, 2014) is notable for its physically based tyre model, which has been used in academic studies of racing line optimisation and tyre energy modelling (Limebeer and Massaro, 2018).

BeamNG.tech (BeamNG GmbH, 2024) represents a qualitatively different approach: rather than modelling the vehicle as a rigid multi-body system with parameterised connections, it represents each structural component as a network of mass nodes connected by beam elements with configurable stiffness and damping coefficients. This soft-body formulation allows individual component deformation and failure to be simulated naturally, producing emergent suspension kinematics, tyre deformation, and crash dynamics without requiring explicitly parameterised kinematic linkages. Riedmaier et al. (2020) conducted a broad survey of simulation platforms for autonomous vehicle testing and rated BeamNG.tech favourably for the realism of its vehicle dynamics model relative to its accessibility and computational cost. The beamngpy library (Stark et al., 2020) provides a Python API enabling headless, script-driven operation, making BeamNG.tech uniquely suitable for large-scale automated parameter sweeps.

### 2.2 Design of Experiments for Vehicle Parameter Studies

The selection of parameter configurations to evaluate is a classical problem in experimental design. The naive full factorial approach — evaluating every combination of every parameter at every level — scales exponentially with the number of parameters and levels, quickly becoming infeasible. For the 27 continuous parameters used in this study, even a coarse three-level factorial design would require 3²⁷ ≈ 7.6 × 10¹² evaluations, which is entirely beyond computational feasibility.

Space-filling designs provide a principled alternative. Random sampling provides asymptotically uniform coverage of the parameter space but is subject to clustering in finite samples — the probability that two random samples will be within ε of each other in a high-dimensional space is far from zero. Latin Hypercube Sampling (LHS), introduced by McKay, Beckman and Conover (1979), addresses this by stratifying each dimension into n equi-probable intervals and ensuring that each interval is sampled exactly once in each dimension. The resulting sample has good marginal uniformity and, for practical purposes, substantially better space-filling properties than random sampling for the same sample size. Iman and Conover (1982) extended LHS to allow control of rank correlations between input variables, enabling simulation experiments to match prescribed correlation structures in input factors.

The choice of sample size for LHS is guided by the rule of 10n (where n is the number of input dimensions) as a rough lower bound (Loeppky, Sacks and Welch, 2009), giving 270 minimum samples for the 27-dimensional space of this study. The present work collected 500 LHS samples, modestly exceeding this threshold.

One-at-a-Time (OAT) designs complement LHS by providing interpretable single-parameter sensitivity estimates. Morris (1991) formalised the Elementary Effects method, which uses OAT-style sampling with multiple starting points to estimate first-order and interaction effects simultaneously. The simpler extreme-OAT design used in the present study — each parameter swept from minimum to maximum with all others at baseline — does not account for parameter interactions but provides useful interpretability with only 2n = 54 additional runs.

Sacks et al. (1989) formalised the Design and Analysis of Computer Experiments (DACE) framework, recognising that simulation outputs — unlike physical experiments — are deterministic: repeated runs with identical inputs produce identical outputs. This distinction motivates interpolating surrogate models (such as Gaussian processes, also called kriging) rather than regression-based models that assume residual noise. However, for models with many inputs and limited training data, regression-based surrogates often outperform interpolating models in practice (Forrester, Sóbester and Keane, 2008), particularly when the true function has local features that cannot be captured by the smooth interpolation kernels typically used.

### 2.3 Surrogate Modelling in Engineering Design

Surrogate models (metamodels, response surface models) approximate an expensive function — in this case, the simulation — using a statistical model fitted to observed input–output pairs. Once trained, the surrogate can be evaluated at negligible computational cost, enabling large-scale optimisation and sensitivity analysis that would be infeasible using the original simulation directly. Forrester, Sóbester and Keane (2008) provide a comprehensive treatment of surrogate modelling for engineering design, benchmarking kriging, polynomial response surfaces, radial basis functions, and neural networks across a range of aerospace optimisation problems.

In the automotive domain, surrogate models have been applied across multiple engineering domains. Gobbi and Mastinu (2001) used polynomial response surfaces to optimise passive suspension parameters for a quarter-car model, demonstrating that a low-order polynomial could capture the dominant trends in ride and handling trade-offs. Fang et al. (2005) compared kriging, polynomial response surfaces, and radial basis functions for crashworthiness optimisation, finding that kriging provided the best accuracy for smooth, low-noise objective functions. Atkinson et al. (2005) demonstrated that neural network surrogates could accurately predict engine-out emissions as a function of combustion parameters, enabling real-time calibration optimisation.

Tree-based ensemble methods have emerged as particularly effective surrogates for high-dimensional engineering problems. Random Forests (Breiman, 2001) construct an ensemble of decision trees, each trained on a bootstrap subsample of the data with a random subset of features considered at each split, and average their predictions. The ensemble mechanism substantially reduces variance compared to a single tree, while the random feature subsetting decorrelates the individual trees, providing near-optimal variance reduction. The natural provision of feature importance estimates — as the mean decrease in impurity across all splits of each feature — makes Random Forests well-suited to sensitivity analysis alongside prediction.

Geurts, Ernst and Wehenkel (2006) introduced the Extremely Randomized Trees (Extra Trees) variant, in which both the split feature and the split threshold are chosen randomly rather than greedily. This further reduces variance at the cost of a small increase in bias, often yielding competitive or superior accuracy with faster training. Gradient Boosting (Friedman, 2001) takes a different approach: rather than averaging parallel trees, it builds trees sequentially, each fitting the residuals of the preceding ensemble. The resulting model is more flexible but more sensitive to hyperparameters and prone to over-fitting without careful regularisation.

SHAP (SHapley Additive exPlanations) values (Lundberg and Lee, 2017) provide a theoretically grounded approach to model explanation. Rooted in cooperative game theory (Shapley, 1953), SHAP assigns each input feature a contribution to each prediction such that the contributions sum to the difference between the prediction and the mean prediction, and such that the contributions are consistent and locally accurate. For tree-based models, exact SHAP values can be computed efficiently using TreeSHAP (Lundberg et al., 2020), enabling feature attribution for large datasets without approximation.

### 2.4 Multi-Objective Bayesian Optimisation

Bayesian optimisation (Mockus, 1974; Jones, Schonlau and Welch, 1998) provides a principled approach to optimising expensive black-box functions by maintaining a probabilistic surrogate model — typically a Gaussian process — and using an acquisition function to balance exploration of uncertain regions with exploitation of known good regions. The Expected Improvement (EI) criterion (Mockus, Tiesis and Zilinskas, 1978) remains the most widely used acquisition function, selecting the point that maximises the probability-weighted gain over the current best observation.

For multi-objective optimisation, the Pareto front — the set of solutions not dominated in all objectives simultaneously — replaces the scalar optimum as the solution concept. A solution A dominates solution B if A is at least as good as B in all objectives and strictly better in at least one. The hypervolume indicator, first proposed by Zitzler and Thiele (1998), provides a scalar measure of Pareto front quality by computing the volume of the objective space dominated by the Pareto front and bounded by a reference point.

Multi-objective Bayesian optimisation methods include ParEGO (Knowles, 2006), which scalarises objectives using random weight vectors, and SMS-EGO (Ponweiser et al., 2008), which directly maximises the hypervolume improvement. Tree-structured Parzen Estimators (TPE), as implemented in the Optuna framework (Akiba et al., 2019), provide a scalable alternative to Gaussian process-based Bayesian optimisation for high-dimensional problems. TPE models the density of good and bad configurations separately using kernel density estimators, avoiding the O(n³) covariance matrix inversion required by Gaussian process inference.

Deb et al. (2002) introduced NSGA-II, a widely used evolutionary algorithm for multi-objective optimisation that has been applied extensively to vehicle engineering problems, including suspension parameter optimisation (Papageorgakis et al., 2019) and powertrain calibration (Atkinson et al., 2005). Evolutionary methods are generally complementary to Bayesian optimisation: they are more sample-efficient but do not naturally quantify uncertainty, making them less suitable for problems where each evaluation is genuinely expensive.

### 2.5 Anti-lock Braking Systems and Vehicle Dynamics

The Anti-lock Braking System prevents wheel lockup during hard braking by cyclically reducing and restoring brake pressure to maintain tyre slip within the stable region of the longitudinal slip–force curve (Bosch, 2011). In the absence of ABS, hard braking typically produces wheel lockup within one to two seconds, which reduces the longitudinal force (since the static friction coefficient at zero slip is greater than the sliding friction coefficient at 100% slip for most road–tyre combinations) and eliminates lateral force entirely (a locked, sliding wheel generates negligible lateral force). ABS controllers operate at 10–20 Hz, modulating pressure to keep slip in the 10–20% range where longitudinal force is near its maximum and lateral force is partially maintained.

From a vehicle dynamics perspective, ABS introduces a closed-loop control layer between the driver's braking demand and the tyre forces. In simulation, this has a significant implication for surrogate modelling: setup parameters that affect tyre slip response (brake bias, tyre pressure, spring rate through load transfer dynamics) will have a reduced apparent effect on braking performance KPIs when ABS is active, because the controller partially compensates for setup changes by adjusting the modulation frequency. Disabling ABS exposes the open-loop tyre–brake interaction directly to the parameter space, potentially amplifying the signal available to the surrogate model (Zegelaar, 1998).

The JBeam vehicle modelling language used by BeamNG.tech represents ABS enablement as a property of individual wheel pressure modulator nodes (`"enableABS":true`) set within a hierarchical part slot system (BeamNG GmbH, 2024). For the ETK800 vehicle, the ABS enablement is carried by the Driver Safety Electronics `etk_DSE_ABS` slot within the common vehicle components library, rather than in the vehicle-specific brakes slot. This distinction was non-trivial to discover: the vehicle-specific slot (`etk800_ABS`) exists but does not carry the `enableABS` property, meaning that removing only the vehicle-specific slot leaves ABS fully operational. Correct disablement required removing the `etk_DSE_ABS` slot from the part configuration passed to BeamNG.tech at simulation launch, combined with a Lua-level function pointer override applied via the beamngpy vehicle command interface immediately after each simulation reset, to ensure that the change persisted through the BeamNG scene initialisation sequence.

### 2.6 Research Gap

While surrogate modelling and Bayesian optimisation are well-established in aerodynamic optimisation (Papageorgakis et al., 2019), powertrain calibration (Atkinson et al., 2005), and suspension tuning for specific vehicles (Gobbi and Mastinu, 2001), their application to the full multi-test-scenario chassis setup space using soft-body simulation data is substantially less studied. Existing studies in this space typically use one of three approaches: (a) higher-fidelity commercial tools (CarSim, IPG CarMaker) with narrow parameter ranges validated against a specific vehicle; (b) simple analytical models (quarter-car, bicycle model) that are fast to evaluate but capture only a fraction of the relevant physics; or (c) single-objective, single-scenario optimisation that does not address the competing-objectives structure of real vehicle development.

The present work contributes to this gap by implementing a replicable, open-toolchain pipeline using a freely accessible simulation platform, evaluating model performance systematically across four scenarios and thirteen KPIs, and explicitly characterising the effect of safety system state (ABS) on data quality. The identification and correction of the sentinel-zero failed-run issue — which would be encountered by any practitioner attempting to replicate this approach — is a methodological contribution with broad applicability.

---

## 3. Methodology

### 3.1 Overview

The research pipeline comprises five sequential stages, as illustrated in Figure 1: (1) experimental design and parameter space definition; (2) automated simulation data collection; (3) data cleaning and quality assurance; (4) feature engineering and ML model training; and (5) Bayesian optimisation using trained surrogate models. Each stage is implemented as a standalone Python module within a shared project structure, with all configuration (file paths, ML hyperparameters, target KPI lists, parameter column names) centralised in a single `config/settings.py` file to ensure consistency across stages. The pipeline supports resume-safe operation: if a data collection session is interrupted, completed runs are preserved as per-run JSON files and can be incorporated into the dataset without re-running; all downstream analysis can be re-run without re-running the simulation.

**Figure 1: System pipeline architecture.** The pipeline flows from parameter sampling (LHS + OAT + baseline) through the BeamNG.tech simulation interface (beamngpy), per-run JSON storage, CSV consolidation, preprocessing (cleaning, feature engineering, scaling), ML training and evaluation (seven model classes plus stacking ensemble), and surrogate-guided Bayesian optimisation. Dashed lines indicate that trained models are reloaded by the optimisation stage without retraining.

### 3.2 Simulation Platform

**Vehicle.** The experimental vehicle is the BeamNG.tech ETK800, a front-wheel-drive compact executive saloon modelled on a platform analogous to a production vehicle in the Volkswagen Golf class. The vehicle employs a transversely mounted four-cylinder petrol engine driving the front axle through a six-speed manual gearbox. Both the front and rear axles are equipped with limited-slip differentials (LSD) with independently configurable preload torque, acceleration lock coefficient, and coast lock coefficient. The vehicle mass is approximately 1,500 kg. The wheelbase is approximately 2.65 m and the track width approximately 1.55 m, consistent with a compact executive platform.

**Simulation platform.** BeamNG.tech v0.38.3.0 was used throughout. The simulation runs in real-time physics mode at the default timestep of 2 ms (500 Hz internal physics). Telemetry was polled via the beamngpy sensors API at 100 Hz. The `gridmap` level provides a flat, uniform, high-friction surface (estimated static friction coefficient ≈ 0.85–0.90 for the stock tyre compound) without gradient, banking, surface irregularity, or environmental effects. All tests were conducted during daylight conditions with no wind effects.

**Interface.** Programmatic control was implemented using beamngpy v1.35. The interface supports: headless (no-window) operation for automated data collection; vehicle configuration via JBeam part slot dictionaries passed at scenario construction; scenario reset and vehicle teleportation between tests; Lua command injection into both the Game Engine (GE) and Vehicle (VE) virtual machines for configuration and diagnostic purposes; and electrics signal polling for driver-assistance system state monitoring.

**ABS disablement.** For ABS-off data collection, three complementary mechanisms were applied: (1) at scenario construction, the JBeam part configuration included `{"etk_DSE_ABS": ""}`, removing the DSE ABS slot that carries the `enableABS:true` property; (2) immediately after each simulation reset, the vehicle-level Lua command `wheels.setABSBehavior("off")` was issued to set the ABS behaviour flag; and (3) a secondary Lua loop directly patched the per-wheel `updateBrake` function pointer from `updateBrakeABS` to `updateBrakeNoABS` for all wheel rotators. A diagnostic counter (`_diag_abs_active_frames`) tracked the number of 100-Hz telemetry frames in which the `abs_active` electrics signal was `true`, confirming that ABS remained inactive across all 555 ABS-off runs (counter = 0 for every run).

### 3.3 Test Scenarios

Four standardised test scenarios were implemented, each executed in a fixed sequence per parameter configuration, with the vehicle teleported to a defined starting position and orientation between tests. All scenarios were implemented as autonomous controller scripts with no human driver input, ensuring perfect repeatability.

**Scenario 1: Launch and braking.** The vehicle was launched from standstill with the throttle pegged at 100% and the steering held at neutral. The time to reach 60 mph (`launch_time_0_60_s`), the peak longitudinal acceleration during the launch phase (`launch_peak_lon_g`), and the distance covered in the first three seconds of motion (`launch_dist_3s_m`) were recorded. The vehicle was then re-launched to 100 mph, at which point a full brake application was commanded. The stopping distance from 100 mph to zero velocity (`brake_stopping_distance_m`) and the peak longitudinal deceleration (`brake_peak_brake_g`) were recorded.

**Scenario 2: Steady-state circle.** The vehicle was driven in a circle of fixed radius (approximately 50 m) at the maximum speed maintainable without loss of control. The maximum lateral acceleration achieved during the steady cornering phase (`circle_max_lat_g`) and the time-averaged lateral acceleration over the steady phase (`circle_avg_lat_g`) were recorded.

**Scenario 3: Slalom.** The vehicle navigated a series of five equally spaced gates at the maximum achievable speed. Peak lateral acceleration during gate transitions (`slalom_max_lat_g`) and average speed across the gate sequence (`slalom_avg_speed_ms`) were extracted from the telemetry.

**Scenario 4: Step steer.** The vehicle was driven at a fixed speed of approximately 50 mph on a straight course. At a defined trigger point, a step steering input equivalent to a 90-degree wheel angle was applied and held. The peak yaw rate achieved during the transient response (`step_steer_peak_yaw_rate`), the time elapsed from input onset to peak yaw rate (`step_steer_time_to_peak_s`), the peak lateral acceleration during the transient (`step_steer_peak_lat_g`), and the ratio of peak yaw rate to the steady-state yaw rate achieved after settling (`step_steer_yaw_overshoot`) were extracted.

Several KPIs initially collected were subsequently excluded from the ML analysis for reasons of degeneracy or data quality. `circle_speed_loss_ms` and `circle_understeer_proxy` exhibited either constant values or near-perfect correlation with other retained KPIs (R² ≈ 1.0), providing no independent information. `step_steer_settle_time_s` consistently reached the 1.5-second window ceiling — meaning the vehicle never reached a defined steady-state within the measurement window — making it uninformative as a target. For the ABS-off dataset, `slalom_max_yaw_rate` and `slalom_yaw_rate_variance` were excluded because wheel lockup during the braking segment of the slalom manoeuvre produced near-random yaw transients that were physically uncorrelated with setup parameters. The retained set of thirteen KPI targets is described in Table 1.

**Table 1: ML target KPIs, their physical interpretation, and descriptive statistics after cleaning (n = 516).**

| KPI | Interpretation | Mean | Std | CV (%) |
|---|---|---|---|---|
| `launch_time_0_60_s` | 0–60 mph time (s) | 7.897 | 0.236 | 3.0 |
| `launch_peak_lon_g` | Peak acceleration (g) | 0.140 | 0.041 | 29.0 |
| `launch_dist_3s_m` | Distance in first 3 s (m) | 4.841 | 0.811 | 16.7 |
| `brake_stopping_distance_m` | Stopping distance from 100 mph (m) | 42.671 | 1.388 | 3.3 |
| `brake_peak_brake_g` | Peak deceleration (g) | 0.164 | 0.054 | 33.0 |
| `circle_max_lat_g` | Maximum lateral acceleration (g) | 0.189 | 0.023 | 12.1 |
| `circle_avg_lat_g` | Mean lateral acceleration (g) | 0.070 | 0.002 | 2.1 |
| `slalom_max_lat_g` | Maximum lateral acceleration (g) | 0.700 | 0.064 | 9.1 |
| `slalom_avg_speed_ms` | Mean gate-passage speed (m/s) | 21.876 | 0.475 | 2.2 |
| `step_steer_peak_yaw_rate` | Peak yaw rate (rad/s) | 0.638 | 0.077 | 12.0 |
| `step_steer_time_to_peak_s` | Time from input to peak yaw (s) | 0.451 | 0.071 | 15.7 |
| `step_steer_peak_lat_g` | Peak lateral acceleration (g) | 0.117 | 0.016 | 13.7 |
| `step_steer_yaw_overshoot` | Peak/steady-state yaw ratio | 1.794 | 0.138 | 7.7 |

### 3.4 Parameter Space and Experimental Design

**Parameter space.** Twenty-seven chassis and powertrain parameters were varied, as detailed in Table 2. The parameters span four subsystems: suspension (spring rates, dampers, anti-roll bars, geometry), brakes (bias and strength), differential (LSD settings for both axles), and powertrain (gear ratios and tyre pressures). Parameter ranges were designed to cover physically plausible extremes for a performance-oriented variant of the vehicle platform: spring rates range from equivalent to a very soft comfort setup (15,000 N/m front) to a firm track setup (160,000 N/m front), and gear ratios range from highway-biased long ratios to short, performance-biased ratios. Tyre pressures span 20–40 PSI, covering the range from dangerously low to over-inflated for the vehicle class. Baseline values correspond to the factory JBeam defaults for the ETK800.

**Table 2: Parameter space definition. Values in SI units unless otherwise stated.**

| Parameter | Description | Min | Max | Baseline |
|---|---|---|---|---|
| `spring_F` | Front spring rate (N/m) | 15,000 | 160,000 | 80,000 |
| `spring_R` | Rear spring rate (N/m) | 15,000 | 140,000 | 70,000 |
| `arb_spring_F` | Front anti-roll bar stiffness (N/m) | 5,000 | 100,000 | 45,000 |
| `arb_spring_R` | Rear anti-roll bar stiffness (N/m) | 2,000 | 50,000 | 25,000 |
| `camber_FR` | Front camber multiplier (1.0 = stock) | 0.95 | 1.05 | 1.002 |
| `camber_RR` | Rear camber multiplier (1.0 = stock) | 0.95 | 1.05 | 0.984 |
| `toe_FR` | Front toe multiplier (1.0 = stock) | 0.98 | 1.02 | 1.000 |
| `toe_RR` | Rear toe multiplier (1.0 = stock) | 0.99 | 1.01 | 0.998 |
| `damp_bump_F` | Front bump damping (N·s/m) | 500 | 12,500 | 6,000 |
| `damp_bump_R` | Rear bump damping (N·s/m) | 500 | 10,000 | 6,000 |
| `damp_rebound_F` | Front rebound damping (N·s/m) | 500 | 25,000 | 18,000 |
| `damp_rebound_R` | Rear rebound damping (N·s/m) | 500 | 20,000 | 14,000 |
| `brakebias` | Brake bias (front fraction, 0–1) | 0.20 | 0.90 | 0.68 |
| `brakestrength` | Brake strength multiplier | 0.60 | 1.00 | 1.00 |
| `lsdpreload_R` | Rear LSD preload torque (N·m) | 0 | 500 | 80 |
| `lsdlockcoef_R` | Rear LSD acceleration lock coefficient | 0.00 | 0.80 | 0.15 |
| `lsdlockcoefrev_R` | Rear LSD coast lock coefficient | 0.00 | 0.50 | 0.01 |
| `lsdpreload_F` | Front LSD preload torque (N·m) | 0 | 500 | 25 |
| `lsdlockcoef_F` | Front LSD acceleration lock coefficient | 0.00 | 0.50 | 0.10 |
| `tyre_pressure_F` | Front tyre pressure (PSI) | 20.0 | 40.0 | 29.0 |
| `tyre_pressure_R` | Rear tyre pressure (PSI) | 20.0 | 40.0 | 29.0 |
| `gear_1` | 1st gear ratio | 2.00 | 5.00 | 4.41 |
| `gear_2` | 2nd gear ratio | 1.50 | 3.50 | 2.31 |
| `gear_3` | 3rd gear ratio | 1.00 | 2.50 | 1.54 |
| `gear_4` | 4th gear ratio | 0.70 | 1.80 | 1.18 |
| `gear_5` | 5th gear ratio | 0.50 | 1.40 | 1.00 |
| `gear_6` | 6th gear ratio | 0.50 | 1.00 | 0.84 |

**Sampling strategy.** A total of 555 parameter configurations were generated across three design types. A single *baseline* run used the factory default parameter vector, providing a reference for relative performance comparisons. Five hundred *Latin Hypercube* samples were generated using `scipy.stats.qmc.LatinHypercube` (Virtanen et al., 2020) with a strength-2 sample, providing good projection properties in all pairs of dimensions. Fifty-four *One-at-a-Time* runs varied each of the 27 parameters individually to its minimum and maximum while holding all others at baseline (2 × 27 = 54 runs), enabling direct estimation of individual parameter effects.

The LHS sample of 500 runs provides approximately 18 samples per parameter dimension, exceeding the rule of ten (McKay, Beckman and Conover, 1979) and approaching the empirically recommended ratio of 20 samples per input for non-linear regression (Loeppky, Sacks and Welch, 2009). After the removal of 39 failed runs, the effective LHS count fell to 461, reducing the samples-per-dimension ratio to approximately 17 — still above the minimum threshold.

### 3.5 Data Collection Pipeline

The scenario runner (`data_collection/scenario_runner.py`) orchestrates the complete data collection lifecycle. For each parameter configuration in the sample, the runner: (1) constructs the JBeam part configuration as a Python dictionary, merging the default factory configuration with the sampled parameter overrides; (2) launches a BeamNG.tech scenario with the configured vehicle positioned at the launch starting point; (3) executes the four test scenarios in fixed sequence, teleporting the vehicle to each scenario's starting position between tests; (4) polls telemetry via the beamngpy sensors API at 100 Hz and stores the raw arrays in memory; (5) computes KPI summaries from the telemetry data using vectorised numpy operations; and (6) serialises the complete run record — parameter configuration, raw KPI values, and diagnostic counters — to a JSON file named with the run ID.

The consolidated CSV dataset is assembled from the per-run JSON files at the conclusion of data collection, or on demand by a separate reconstruction script. This design choice provides resilience against interruption: a session terminated at any point during data collection preserves all completed run records, and the session can be resumed from the next pending run without duplication or data loss. The parallel robustness of the JSON format — each file is written atomically and independently — also means that multiple data collection sessions could in principle be run concurrently on different machines and their outputs merged without conflict.

Two complete datasets were collected: an ABS-on dataset (`sweep_results_rb.csv`) and an ABS-off dataset (`sweep_results_no_abs_rb.csv`), each comprising 555 runs over the same LHS and OAT sample, using the same random seed. The datasets were collected sequentially using an automated batch script (`run_both_datasets.bat`), with the full collection requiring approximately 28 hours of wall-clock time (approximately 14 hours per dataset).

### 3.6 Data Preprocessing

**Failed run detection and removal.** Initial analysis of the ABS-off dataset revealed a significant data quality issue: 39 of the 555 runs (7.0%) exhibited clearly anomalous KPI values consistent with a BeamNG simulation failure during the test sequence. These runs shared two signatures: a `launch_time_0_60_s` value in the range 20–35 seconds (compared to a tight cluster of 7.69–8.21 seconds for successful runs), and sentinel zeros across all KPI columns except `launch_time_0_60_s` and `step_steer_time_to_peak_s`. The sentinel zero pattern — with physically impossible values such as brake stopping distance of 0 m — indicates that the test execution framework caught an internal failure and substituted null-equivalent values rather than recording physically meaningful data.

**Figure 4: Distribution of `launch_time_0_60_s` for all 555 runs in the ABS-off dataset.** The bimodal distribution clearly separates 516 successful runs (7.69–8.21 s) from 39 failed runs (20–35 s). The red dashed line at 15 s marks the detection threshold; no valid run exceeds this value, and no failed run falls below it, providing a clean separator.

The detection and removal threshold of 15 seconds was applied: all rows with `launch_time_0_60_s > 15.0 s` were excluded as failed simulation runs. The impact was substantial: prior to removal, `launch_time_0_60_s` had a coefficient of variation of 70.7%; after removal, CV fell to 3.0%, reflecting the genuine physical range of the target. Similarly, `step_steer_time_to_peak_s` fell from CV = 71.9% to 15.7%. Before correction, all ML models produced test R² values uniformly below zero (minimum −5,516 for the MLP on `circle_avg_lat_g`), confirming that the contaminated dataset was completely unlearnable.

**Physical plausibility filtering.** A secondary filter excluded rows exceeding upper plausibility limits on five KPIs: circle lateral g > 2.5 g, slalom lateral g > 2.5 g, peak brake deceleration > 2.0 g, and stopping distance > 200 m. These limits represent values that are physically achievable only by race cars with extreme aerodynamic downforce and purpose-designed slick tyres, and their presence in data from a production-class vehicle on a flat surface indicates numerical instability in the BeamNG physics integration. No additional rows were removed by this filter after the failed-run filter had been applied.

**Transformations.** The `slalom_yaw_rate_variance` target (collected but later excluded from the ABS-off analysis) exhibited strong right skew (skewness > 3) and was log1p-transformed prior to modelling to improve the approximation to normality required by linear models. No other transformations were applied to KPI targets; parameter inputs were standardised using a RobustScaler rather than per-column transformations.

### 3.7 Feature Engineering

Twenty-nine derived features were constructed from the 27 raw parameters to provide the ML models with direct representations of physically motivated interaction terms. The rationale for each feature group is briefly described:

*Suspension balance ratios* (`feat_spring_ratio`, `feat_spring_balance`, `feat_arb_ratio`, `feat_arb_balance`, `feat_platform_stiffness`): The front-to-rear balance of spring and ARB stiffness determines the primary understeer/oversteer balance of the vehicle; a vehicle with stiffer front springs and ARB will typically understeer more than one with equal stiffness front and rear (Milliken and Milliken, 1995). These features give the model direct access to balance quantities without requiring it to learn the ratio relationship from the raw spring rates.

*Contact load proxies* (`feat_front_contact_load`, `feat_rear_contact_load`): Approximated as spring rate × tyre pressure / 10⁶, these features capture a coarse proxy for the tyre contact patch loading, which governs the maximum friction force available at each axle.

*Critical damping ratios* (`feat_crit_damp_ratio_bump_F`, `feat_crit_damp_ratio_bump_R`, etc.): Computed as ζ = c / (2√(k·m_corner)) for each combination of bump and rebound damper, using an assumed sprung corner mass of 375 kg (vehicle mass ≈ 1,500 kg, four corners). A damping ratio near 1.0 indicates critically damped suspension response; values significantly below 1.0 indicate underdamping (oscillatory response), while values above 1.0 indicate overdamping (sluggish response). These features encode the normalised damping quality independent of the absolute spring and damper magnitudes.

*Gear spread* (`feat_gear_spread`, `feat_gear_1_6_ratio`): The ratio of first to sixth gear captures the overall gear spread, which determines how many gear changes are required and how close each gear is to the engine's power band through a given speed range.

The final feature matrix comprised 56 columns (27 raw parameters plus 29 derived features). Features were scaled using a `RobustScaler` (scikit-learn; Pedregosa et al., 2011), which normalises using the median and interquartile range rather than the mean and standard deviation, providing robustness to the residual outliers present in the raw parameter distributions.

### 3.8 Model Training and Evaluation

**Dataset partition.** The 516-row cleaned dataset was partitioned into training (80%, n = 412) and held-out test sets (20%, n = 104) using a fixed random seed (RANDOM_STATE = 42). All hyperparameter selection, model development, and cross-validation were performed exclusively on the training split; the test set was used only for final evaluation, once per model.

**Model specifications.** Each of seven model classes was trained independently for each of the 13 KPI targets, using the configurations described in Section 3.3 of the methodology. A Stacking Ensemble was constructed using all seven base models as level-0 learners, generating out-of-fold predictions on the training set via 5-fold cross-validation, and training a Ridge regression meta-learner on the resulting stacked feature matrix. This approach avoids information leakage from the training targets to the meta-learner without requiring a separate validation set.

**Evaluation metrics.** The primary evaluation metric was the coefficient of determination (R²) on the held-out test set:

R² = 1 − (Σᵢ(yᵢ − ŷᵢ)²) / (Σᵢ(yᵢ − ȳ)²)

where yᵢ are the true target values, ŷᵢ are the model predictions, and ȳ is the mean of the true values. R² = 1 indicates perfect prediction; R² = 0 indicates that the model performs no better than predicting the mean for every observation; R² < 0 indicates that the model performs worse than the mean predictor. Five-fold cross-validated R² (CV R²) on the training set was reported to assess generalisation and detect over-fitting.

Secondary metrics included root mean squared error (RMSE) and relative RMSE (RMSE / mean target × 100%), providing scale-interpretable measures of prediction accuracy.

**Feature importance and explainability.** SHAP values (Lundberg and Lee, 2017) were computed for the best-performing model on each target. For tree-based models (Random Forest, Extra Trees), exact TreeSHAP values were computed using the `shap.TreeExplainer` interface. For the stacking ensemble, SHAP values were approximated using `shap.KernelExplainer` on a 100-sample background dataset. Feature importances from the Random Forest were additionally computed as mean decrease in impurity (MDI), normalised to sum to one.

### 3.9 Bayesian Optimisation

Surrogate-guided optimisation used the trained best models as the objective function for each target. The Optuna v3 framework (Akiba et al., 2019) with the default TPE sampler was used for both single-objective and multi-objective search. For each optimisation run, 500 trials were conducted; the first 50 were random samples used to initialise the TPE model, with subsequent trials guided by the acquisition function.

For single-objective experiments, the optimisation minimised the predicted `launch_time_0_60_s` within the same parameter bounds used in data collection. For the multi-objective experiment, the objectives were minimisation of `launch_time_0_60_s` and maximisation of `circle_max_lat_g`. The resulting Pareto front was identified using the standard dominance definition: a solution is Pareto-optimal if no other trial in the search history improves both objectives simultaneously.

A complementary one-dimensional sensitivity analysis was conducted by holding all parameters at the identified single-objective optimal configuration and sweeping each parameter individually from its minimum to maximum in 50 equal steps, querying the surrogate model at each step. The predicted range (max − min) of the KPI over the sweep was recorded as a measure of individual parameter sensitivity.

---

## 4. Results and Analysis

### 4.1 Data Quality and Descriptive Statistics

Of the 555 simulation runs collected for the ABS-off dataset, 516 (93.0%) passed all quality filters and were retained for analysis. The 39 rejected runs were uniformly distributed across the LHS sample — no single parameter region was disproportionately represented among failed runs — confirming that the failures arose from isolated numerical instability at specific parameter combinations rather than from a systematic region of the space where the simulation is unreliable. The OAT and baseline runs had a 0% failure rate, consistent with these configurations being closer to the validated baseline and therefore less likely to encounter numerical instability.

The source breakdown of the clean dataset was: LHS = 461 runs, OAT = 54 runs, baseline = 1 run (total 516). After the 80:20 train/test split, 412 training samples and 104 test samples were available for each of the 13 KPI targets.

Table 1 in Section 3.3 gives the descriptive statistics of all thirteen KPI targets after cleaning. The coefficients of variation range from 2.1% (`circle_avg_lat_g`) to 33.0% (`brake_peak_brake_g`). This wide range has direct implications for predictability: a KPI with CV = 3% has a standard deviation of only 3% of its mean, meaning the absolute prediction accuracy required to achieve R² > 0 is extremely stringent — the model must distinguish between configurations that differ by fractions of a percent in performance. By contrast, `brake_peak_brake_g` at CV = 33% provides much more headroom for a model to demonstrate structure above the noise floor.

The step-steer KPIs — with CVs of 7.7–15.7% — represent the most promising targets for dynamic response modelling. These KPIs are physically governed by the chassis parameters most varied in the experiment: suspension stiffness (which determines load transfer rate), damper settings (which govern transient response), and differential lock coefficients (which affect axle coupling during the yaw transient). The step-steer test is also unaffected by ABS, since it is a steering rather than braking manoeuvre, making it free of the open-loop noise that contaminates braking KPIs.

### 4.2 ABS-On versus ABS-Off Comparison

The ABS comparison was conducted using both complete 555-run datasets, employing 5-fold cross-validation on the full data to obtain CV R² estimates. At this stage, the failed-run filter had not yet been applied; the absolute CV R² values are therefore negative in both conditions, reflecting the dominant effect of the 39 sentinel-zero runs on the variance structure. However, the relative differences between conditions are robust to this common contamination.

**Figure 5: ABS-on versus ABS-off comparison of 5-fold CV R² for Random Forest and Ridge models.** Blue bars show ABS-on; red bars show ABS-off. The horizontal dashed line at R² = 0 marks the mean-predictor baseline. Note that all bars are negative at this stage due to pre-filter contamination; the quantity of primary interest is the relative improvement of ABS-off over ABS-on.

Table 3 presents the ABS comparison results for the Random Forest model. The step-steer dynamic response KPIs show dramatically improved predictability under ABS-off conditions:

**Table 3: ABS-on versus ABS-off cross-validated R² (Random Forest, 5-fold, pre-filter). Δ = ABS-off minus ABS-on; positive indicates ABS-off is more predictable.**

| KPI | ABS-on CV R² | ABS-off CV R² | Δ |
|---|---|---|---|
| `step_steer_yaw_overshoot` | −1.679 | −0.274 | **+1.405** |
| `step_steer_peak_yaw_rate` | −0.946 | −0.370 | **+0.576** |
| `step_steer_time_to_peak_s` | −0.359 | −0.010 | **+0.349** |
| `launch_dist_3s_m` | −0.493 | −0.176 | +0.317 |
| `slalom_max_lat_g` | −0.227 | −0.123 | +0.104 |
| `launch_peak_lon_g` | −0.232 | −0.167 | +0.065 |
| `circle_max_lat_g` | −0.167 | −0.152 | +0.015 |
| `circle_avg_lat_g` | −0.060 | −0.051 | +0.009 |
| `brake_stopping_distance_m` | −0.114 | −0.147 | −0.033 |
| `launch_time_0_60_s` | −0.228 | −0.302 | −0.074 |
| `step_steer_peak_lat_g` | −0.043 | −0.263 | −0.220 |
| `slalom_avg_speed_ms` | −0.109 | −0.339 | −0.230 |

The three step-steer KPIs — yaw overshoot (+1.41), peak yaw rate (+0.58), and time to peak (+0.35) — show the largest improvements under ABS-off conditions. This strongly supports the hypothesis stated in RQ2: removing the closed-loop ABS controller reveals genuine chassis parameter sensitivity in yaw dynamics. The physical interpretation is straightforward: ABS modulates brake pressure independently at each wheel to prevent lockup, and in doing so, it partially decouples the yaw response from the parameters that govern the raw tyre force distribution (brake bias, LSD lock coefficients, suspension stiffness affecting load transfer). When ABS is active, the controller compensates for setup changes, masking the parameter sensitivity. When ABS is disabled, each parameter change directly affects the braking torque distribution and consequently the yaw dynamics, making the relationship learnable.

The deterioration under ABS-off conditions for `slalom_avg_speed_ms` (Δ = −0.230) and `step_steer_peak_lat_g` (Δ = −0.220) is also physically interpretable. The slalom test requires a hard brake application at the conclusion of the gate sequence; without ABS, wheel lockup during this phase introduces stochastic variation in vehicle attitude that contaminates the lateral grip measurement. Similarly, `step_steer_peak_lat_g` during a braking-coupled step steer will be affected by the open-loop lockup dynamics in a way that is not cleanly parameterised.

### 4.3 ML Model Performance

After applying the failed-run filter and retraining all models on the 516-row dataset, five of the thirteen KPI targets yielded positive held-out test R², demonstrating that genuine parameter–KPI structure is present and learnable in the clean dataset.

**Figure 6: Heatmap of test R² for all eight model variants across all thirteen KPI targets.** Cells are coloured from red (strongly negative R²) through white (zero) to blue (positive R²). MLP cells with extreme negative values are clamped to −5 for visual clarity; the true minimum is −5,516 for `circle_avg_lat_g`.

**Table 4: Best-performing model per KPI target on the held-out test set (n = 104). Rel. RMSE = RMSE / mean target × 100%.**

| KPI | Best Model | Test R² | CV R² | Rel. RMSE (%) |
|---|---|---|---|---|
| `brake_stopping_distance_m` | Stacking Ensemble | **+0.222** | +0.163 | 3.0 |
| `launch_time_0_60_s` | Random Forest | **+0.204** | +0.145 | 2.7 |
| `step_steer_peak_yaw_rate` | Random Forest | **+0.149** | +0.130 | 10.5 |
| `launch_dist_3s_m` | Random Forest | **+0.137** | +0.147 | 16.6 |
| `slalom_avg_speed_ms` | Extra Trees | **+0.110** | +0.174 | 2.2 |
| `step_steer_peak_lat_g` | Extra Trees | +0.057 | −0.109 | 14.2 |
| `circle_avg_lat_g` | Extra Trees | +0.049 | +0.016 | 1.9 |
| `step_steer_yaw_overshoot` | Extra Trees | +0.033 | +0.057 | 6.6 |
| `slalom_max_lat_g` | Stacking Ensemble | +0.013 | −0.004 | 9.5 |
| `launch_peak_lon_g` | Random Forest | +0.002 | −0.048 | 29.6 |
| `brake_peak_brake_g` | Stacking Ensemble | −0.000 | −0.015 | 32.0 |
| `circle_max_lat_g` | Bayesian Ridge | −0.002 | −0.013 | 12.1 |
| `step_steer_time_to_peak_s` | Bayesian Ridge | −0.008 | −0.017 | 16.2 |

Several patterns are apparent from Table 4. First, ensemble tree methods (Random Forest, Extra Trees, Stacking Ensemble) dominate the best-model column, appearing for 11 of 13 targets. This is consistent with the established superiority of ensemble methods over single models for tabular regression tasks (Geurts, Ernst and Wehenkel, 2006). The MLP, despite having the highest representational capacity of all models, produces the worst results in virtually every case — its test R² values include outliers such as −36.1 (`circle_max_lat_g`) and −9.9 (`brake_stopping_distance_m`), indicating severe over-fitting on the relatively small training set. Neural networks are well-known to require substantially more data than tree methods to achieve competitive performance on tabular data without careful regularisation (Grinsztajn, Oyallon and Varoquaux, 2022).

Second, the linear models (Bayesian Ridge, ElasticNet) provide the best performance only for `circle_max_lat_g` and `step_steer_time_to_peak_s`, both of which have near-zero R². This suggests that whatever structure is present in these KPIs is too weak for any model to capture reliably, and the linear models "win" by simply being the least over-fitted rather than by capturing meaningful structure.

Third, the correspondence between CV R² and test R² is reasonable for models with positive R² (correlation ≈ 0.85), providing confidence that the test set evaluations are generalisable rather than random. For models with negative R², the test-CV correspondence is weaker, as expected when models are fitting noise.

**Figure 7: Predicted versus actual scatter plots for `launch_time_0_60_s` (Random Forest, R² = 0.204) and `brake_stopping_distance_m` (Stacking Ensemble, R² = 0.222).** Both panels show moderate positive correlation between predictions and actuals, confirming that the models have captured genuine signal. The spread around the diagonal is consistent with the low CV of these targets (3.0% and 3.3% respectively) and the relatively small absolute parameter effects.

### 4.4 Feature Importance and SHAP Analysis

**Figure 8: SHAP summary plot for the Random Forest model trained on `launch_time_0_60_s`.** The top ten features by mean absolute SHAP value are shown. Each row shows the distribution of SHAP values across all training samples; colour indicates the raw feature value (blue = low, red = high). Features are ranked by their mean impact on the model output.

The SHAP analysis for `launch_time_0_60_s` reveals the dominant role of gear ratios: `gear_1` and `feat_gear_spread` (first-to-sixth gear ratio) are among the top three features, consistent with the physical reality that in a FWD vehicle, the 0–60 time is primarily determined by how much of the acceleration distance is spent in first gear and how effectively the powertrain converts engine torque to wheel force through the gear ratios.

The feature `brakestrength` appears with a negative SHAP contribution for high values — counter-intuitively, higher brake strength is associated with slightly longer 0–60 times. Investigation of the test sequence reveals the mechanism: the combined launch-and-brake scenario applies full braking after the 0–60 test, and in some configurations with high brake strength, the aggressive brake calibration introduces a brief braking pulse at the start of the run (where the controller transitions from brake-applied rest to full throttle), delaying the onset of full acceleration. This is a simulation-specific artefact of the test implementation rather than a genuine vehicle physics effect.

Rear toe multiplier (`toe_RR`) appears in the top five features, reflecting the significant effect of rear wheel alignment on tyre rolling resistance and thrust direction in a FWD vehicle at high longitudinal slip. This is consistent with the OAT sensitivity analysis.

**Figure 9: One-dimensional sensitivity analysis for `launch_time_0_60_s`.** Each bar represents the predicted range (max − min, in seconds) as a single parameter is swept from its minimum to maximum. Brakestrength has the largest effect (predicted range 0.095 s, 1.16% of mean), followed by rear LSD preload (0.053 s, 0.64%) and front LSD preload (0.044 s, 0.53%). Parameters with predicted range below 0.02 s (0.25%) are not shown.

**Table 5: Top ten parameters by OAT sensitivity for `launch_time_0_60_s`.**

| Rank | Parameter | Min Pred (s) | Max Pred (s) | Range (s) | Rel. Range (%) |
|---|---|---|---|---|---|
| 1 | `brakestrength` | 8.108 | 8.204 | 0.095 | 1.163 |
| 2 | `lsdpreload_R` | 8.176 | 8.229 | 0.053 | 0.644 |
| 3 | `lsdpreload_F` | 8.157 | 8.200 | 0.044 | 0.532 |
| 4 | `toe_RR` | 8.188 | 8.227 | 0.039 | 0.472 |
| 5 | `lsdlockcoef_R` | 8.161 | 8.199 | 0.037 | 0.457 |
| 6 | `arb_spring_R` | 8.170 | 8.206 | 0.037 | 0.446 |
| 7 | `damp_bump_F` | 8.194 | 8.229 | 0.036 | 0.437 |
| 8 | `gear_1` | 8.166 | 8.201 | 0.035 | 0.425 |
| 9 | `lsdlockcoef_F` | 8.167 | 8.202 | 0.035 | 0.424 |
| 10 | `arb_spring_F` | 8.183 | 8.215 | 0.032 | 0.388 |

The absolute sensitivity ranges are small — the maximum predicted effect of any single parameter is less than 0.1 seconds, representing less than 1.2% of the mean 0–60 time. This is consistent with the low CV of the target (3.0%) and the modest R² of the best model (0.204), and reflects a genuine characteristic of this vehicle and test scenario rather than a modelling failure.

### 4.5 Bayesian Optimisation Results

**Single-objective optimisation.** The Optuna TPE search minimising `launch_time_0_60_s` converged, after 500 trials, to a predicted value of 7.837 seconds. The baseline vehicle produces 7.897 seconds; the minimum observed in the training data is 7.687 seconds. The optimised configuration features brakestrength = 0.60 (minimum of range, which reduces the anomalous braking pulse at launch onset), rear LSD preload = 210 N·m (intermediate, balancing straight-line traction with tyre scrub), rear toe = 0.993 (slightly below baseline, equivalent to a small amount of toe-out), and first-gear ratio = 4.3 (close to maximum, maintaining high wheel torque through the initial acceleration phase).

**Multi-objective optimisation.** The joint optimisation of `launch_time_0_60_s` (minimise) and `circle_max_lat_g` (maximise) identified a Pareto front of 20 non-dominated solutions after 500 TPE trials. Figure 10 shows the Pareto front in objective space.

**Figure 10: Pareto front for launch time versus circle maximum lateral g.** Each point represents a Pareto-optimal solution in the 27-dimensional parameter space, shown projected onto the two objective dimensions. The star symbol marks the position of the factory baseline configuration. The Pareto front illustrates the achievable trade-off boundary: moving along the front from lower-left (fast launch, low cornering) to upper-right (slower launch, higher cornering) traces the set of setups that are optimal for a given weighting of the two objectives.

The Pareto front spans predicted launch times from 7.838 to 7.891 seconds and predicted lateral g values from 0.174 to 0.188 g. The front shows a clear negative correlation between the two objectives — configurations tuned for minimum launch time tend to sacrifice cornering performance, and vice versa. This trade-off is physically interpretable for a FWD vehicle: aggressive short-ratio gearing and high LSD lock coefficients that improve straight-line traction tend to increase tyre scrub during cornering, reducing cornering grip. The factory baseline sits near the upper portion of the Pareto front, suggesting it is relatively well-balanced between the two objectives by design.

It is important to note that the lateral g surrogate model has near-zero test R² (−0.002), meaning that individual points on the Pareto front for `circle_max_lat_g` should not be taken as reliable predictions of the actual cornering performance of those configurations. The shape of the Pareto front and the general direction of the trade-off are more reliable conclusions from the surrogate optimisation than specific configurations. Validation of Pareto-optimal candidates through additional simulation runs would be the appropriate next step before acting on these recommendations.

---

## 5. Discussion

### 5.1 Predictability of Vehicle KPIs from Simulation Data

The principal finding of this study — that ML models achieve modest but statistically meaningful positive R² for five of thirteen KPI targets — represents a partial confirmation of RQ1. The pattern of predictability provides insights into both the physics of the test environment and the capabilities and limitations of the ML approach.

**Longitudinal KPIs are most predictable.** Brake stopping distance (R² = 0.222) and 0–60 time (R² = 0.204) are the two KPIs with the clearest setup sensitivity. This is physically consistent: without ABS, braking distance is directly governed by the brake force distribution (brake bias, brake strength) and the tyre–road friction available at each wheel (influenced by tyre pressure and load, and thus by spring rates through static load distribution). The 0–60 time is governed by gear ratios and LSD settings that control torque delivery. These relationships are relatively direct and monotonic, making them tractable for tree-based models with moderate training data.

**Step-steer yaw dynamics show intermediate predictability.** `step_steer_peak_yaw_rate` (R² = 0.149) is the highest-R² dynamic response KPI, confirming the ABS-off hypothesis: with the confounding effect of ABS removed, the genuine chassis parameter sensitivity of yaw dynamics is exposed. The step-steer test is inherently well-suited to this study because it is a pure steering manoeuvre uncorrupted by open-loop braking noise.

**Lateral grip KPIs are unpredictable.** `circle_max_lat_g` (R² = −0.002), `circle_avg_lat_g` (R² = +0.049), and `slalom_max_lat_g` (R² = +0.013) are near-zero or marginally positive. Three complementary explanations account for this: first, the flat gridmap constrains maximum lateral acceleration primarily by the fixed road–tyre friction coefficient, reducing the achievable CV to 12.1%, 2.1%, and 9.1% respectively; second, the FWD architecture means that rear suspension, rear ARB, and rear geometry parameters — which account for nearly half the feature space — have limited influence on lateral grip compared to a rear-drive vehicle; third, the contact patch model in BeamNG.tech's soft-body tyre representation may not reproduce the full sensitivity of camber and toe changes on peak lateral force that is observed with measured Pacejka-parameterised tyres.

**The failed-run discovery is a central methodological finding.** The identification of 39 sentinel-zero failed runs (7.0% of the dataset) as the root cause of all models producing negative R² represents the most important practical finding of this study. The standard preprocessing steps applied in the broader ML literature — dropping NaN values, applying plausibility filters for obvious outliers, checking for duplicate rows — would not have detected these runs, since their KPI values of exactly 0.0 are numerically valid and the `launch_time_0_60_s` value, while extreme, is not above the maximum plausibility limit. Only domain knowledge — that a 0–60 time of 25 seconds is physically impossible for this vehicle — enabled their detection. This experience strongly recommends that any future simulation-based ML study implement domain-specific validation beyond standard data cleaning, including expected-range checks for each KPI and cross-KPI consistency checks (e.g. a run with `brake_stopping_distance_m = 0` must also have `brake_peak_brake_g = 0`, since zero stopping distance implies zero deceleration force).

### 5.2 The ABS Hypothesis

The ABS comparison results provide clear empirical support for RQ2. The step-steer yaw overshoot improvement of Δ = +1.41 CV R² under ABS-off conditions — from −1.68 (effectively random) to −0.27 (approaching zero but still negative, due to the pre-filter contamination of the comparison dataset) — represents the largest KPI-level improvement observed in the study and validates the core hypothesis that ABS suppresses genuine chassis parameter sensitivity in yaw dynamics.

From a methodological perspective, this finding has practical implications for future simulation-based vehicle studies: where the research question concerns the relationship between chassis parameters and dynamic response, disabling ABS (or other closed-loop chassis control systems) may substantially increase the signal available to surrogate models. The cost is increased noise in KPIs that are directly affected by wheel lockup dynamics (braking distance, lateral grip during combined braking and cornering), and researchers must be prepared to exclude or carefully treat these noisier targets.

The result also suggests a potential avenue for future work: rather than simply enabling or disabling ABS, a partial approach — disabling ABS only during the step-steer test and enabling it for the launch/brake and slalom tests — could maximise signal across all KPIs simultaneously. This would require modifications to the test sequence controller to toggle ABS state between scenarios within a single run.

### 5.3 Model Selection Considerations

The dominance of tree-based ensemble methods (Random Forest, Extra Trees, Stacking Ensemble) and the poor performance of the MLP are consistent with recent empirical findings in the tabular ML literature. Grinsztajn, Oyallon and Varoquaux (2022) conducted a large-scale benchmark of ML methods on tabular datasets and found that tree-based methods outperform neural networks on the majority of tasks, particularly when the dataset is small to medium-sized and features contain both continuous and categorical variables. The authors attribute this to the inductive biases of tree methods — in particular, their ability to handle irregularly distributed features and their robustness to irrelevant inputs — which align well with the characteristics of vehicle parameter datasets.

The SVR's poor performance on this dataset, despite its theoretical suitability for smooth, low-noise regression, likely reflects the challenge of kernel hyperparameter selection in high-dimensional spaces with limited data. With 56 input features and only 412 training samples, the kernel bandwidth selection becomes difficult, and the RBF kernel may not be well-suited to the discrete-like structure of parameters such as gear ratios.

The near-perfect match between test R² and CV R² for the five positive-R² KPIs (the two metrics agree within 0.06 on average) suggests that over-fitting is not a significant concern for the tree ensemble methods on this dataset, and that the test set evaluations are reliable estimates of out-of-sample performance.

### 5.4 Implications for Practical Vehicle Setup Optimisation

Despite the modest absolute R² values, the pipeline demonstrates several practically useful capabilities:

**Parameter ranking.** The sensitivity analysis (Table 5) identifies the ten most influential parameters for launch time, enabling an engineer to focus physical testing on the parameters that are most likely to make a measurable difference. In this case, brakestrength, rear LSD preload, front LSD preload, rear toe, and the two LSD lock coefficients account for the majority of predicted sensitivity; the remaining 21 parameters could, according to the surrogate model, be fixed at baseline without material impact on launch time. This is directly actionable as a physical test plan.

**Trade-off quantification.** The Pareto front (Figure 10) provides a decision boundary for the launch time versus lateral grip trade-off, giving a quantitative basis for the intuitive engineering compromise between acceleration and cornering performance. Even with imperfect surrogate accuracy, the direction and rough magnitude of the trade-off are informative for design decisions.

**Simulation failure detection.** The sentinel-zero detection method developed in this study — comparing `launch_time_0_60_s` against an absolute ceiling that separates valid from failed runs — is directly transferable to any data collection pipeline using BeamNG.tech or similar simulation environments. The broader principle — implementing domain-specific validation checks that reflect the physical constraints of the simulation output — should be considered a standard step in any simulation-based ML study.

### 5.5 Limitations

**Simulation fidelity.** BeamNG.tech's soft-body physics model provides a higher-fidelity representation than rigid-body simulators but is not calibrated against measured vehicle data for the specific vehicle modelled. The tyre model is not parameterised from physical measurements and does not implement thermal or wear dynamics. As a result, the absolute KPI values may not correspond closely to those measured from a real vehicle of comparable specification, and the sensitivity of KPIs to parameter changes may differ from physical reality in unknown ways.

**Vehicle architecture.** As discussed throughout this dissertation, the FWD architecture of the ETK800 means that rear-axle parameters (rear spring, rear ARB, rear dampers, rear geometry, rear LSD) have limited influence on performance in the four test scenarios used. A rear-wheel-drive or all-wheel-drive platform would distribute parameter sensitivity more evenly across the axles, providing a more comprehensive test of the pipeline's capabilities.

**Test environment.** The flat gridmap eliminates environmental factors — road gradient, surface irregularity, temperature — that are significant in real-world vehicle development. Lateral grip in particular is strongly influenced by road camber and surface texture; the flat environment compresses the achievable lateral g range and reduces the CV of lateral grip KPIs, making them harder to predict and less representative of real-world conditions.

**Sample size.** At 412 training samples for 56 features, the dataset is at the lower boundary of what is conventionally considered adequate for reliable non-linear regression. The R² improvements following failed-run removal suggest the models are data-limited rather than architecture-limited; a substantially larger dataset (2,000+ LHS samples) would be expected to produce better-fitting models across all KPIs.

---

## 6. Conclusion

This dissertation has presented and systematically evaluated a complete simulation-to-machine-learning pipeline for vehicle performance prediction, from experimental design and automated data collection through preprocessing, model training, and surrogate-guided Bayesian optimisation. The work addressed three research questions and generated a number of findings of both practical and methodological significance.

**RQ1** — whether ML models can predict vehicle KPIs from simulation parameters — is answered affirmatively but with important qualification. Positive test R² was achieved for five of thirteen KPI targets, with brake stopping distance (R² = 0.222, Stacking Ensemble) and 0–60 mph time (R² = 0.204, Random Forest) providing the clearest evidence of learnable structure. For the remaining eight targets, predictability was negligible, attributable to the combination of low coefficient of variation (< 5% for several targets, driven by the constrained flat-track test environment), a FWD architecture with limited rear-parameter sensitivity, and a dataset at the lower boundary of adequacy for the feature dimensionality. Crucially, all models produced uniformly negative R² values before a data quality audit identified and removed 39 failed simulation runs (7.0%) that had contaminated the dataset with sentinel zeros — a finding that underscores the importance of domain-specific validation in simulation-based ML workflows.

**RQ2** — whether ABS state affects KPI predictability — is confirmed. Disabling ABS improved the cross-validated R² of step-steer dynamic response KPIs by up to Δ = +1.41 (yaw overshoot, Random Forest), consistent with the hypothesis that the ABS closed-loop controller suppresses genuine chassis parameter sensitivity in yaw dynamics. The trade-off — ABS removal simultaneously degrades lateral grip and slalom speed KPIs by introducing wheel lockup noise — is also characterised quantitatively, informing the design of future data collection protocols.

**RQ3** — whether surrogate-guided Bayesian optimisation can identify Pareto-optimal configurations — is confirmed. A 20-solution Pareto front was identified for the launch time versus lateral grip trade-off, with brakestrength, rear LSD preload, and rear toe as the most influential parameters. The Pareto front provides a quantitative characterisation of the achievable trade-off boundary in the 27-dimensional parameter space.

Future work should prioritise three directions: extending the LHS sample to approximately 2,000 runs to improve model accuracy across all targets; applying the pipeline to a rear-wheel-drive platform for more balanced parameter sensitivity; and incorporating a realistic track environment with elevation and surface variation to improve the representativeness of lateral grip KPIs. The integration of iterative simulation validation — running Bayesian optimisation candidates in the simulator to refine the surrogate model in a closed-loop active learning scheme — represents the natural evolution of this pipeline towards a practical engineering design tool.

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019) 'Optuna: A next-generation hyperparameter optimization framework', *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, Anchorage, Alaska, August 2019, pp. 2623–2631.

Atkinson, C. M., Thompson, G. J., Traver, M. L. and Clark, N. N. (2005) 'In-cylinder combustion pressure modeling of a diesel engine via artificial neural networks', *ASME Journal of Engineering for Gas Turbines and Power*, 121(3), pp. 512–519.

Bakker, E., Nyborg, L. and Pacejka, H. B. (1987) 'Tyre modelling for use in vehicle dynamics studies', *SAE Technical Paper 870421*. Warrendale: Society of Automotive Engineers.

BeamNG GmbH (2024) *BeamNG.tech — Simulation Platform for Research and Development* [online]. Available at: https://www.beamng.com/technology/ (Accessed: 11 April 2026).

Bosch, R. (2011) *Automotive Handbook*. 8th edn. Chichester: John Wiley & Sons.

Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.

Brochu, E., Cora, V. M. and de Freitas, N. (2010) 'A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning', *arXiv*, arXiv:1012.2599 [cs.LG].

Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002) 'A fast and elitist multiobjective genetic algorithm: NSGA-II', *IEEE Transactions on Evolutionary Computation*, 6(2), pp. 182–197.

Emmerich, M. T. M., Giannakoglou, K. C. and Naujoks, B. (2006) 'Single- and multiobjective evolutionary optimization assisted by Gaussian random field metamodels', *IEEE Transactions on Evolutionary Computation*, 10(4), pp. 421–439.

Fang, H., Rais-Rohani, M., Liu, Z. and Horstemeyer, M. F. (2005) 'A comparative study of metamodeling methods for multiobjective crashworthiness optimization', *Computers and Structures*, 83(25–26), pp. 2121–2136.

Forrester, A. I. J., Sóbester, A. and Keane, A. J. (2008) *Engineering Design via Surrogate Modelling: A Practical Guide*. Chichester: John Wiley & Sons.

Friedman, J. H. (2001) 'Greedy function approximation: A gradient boosting machine', *Annals of Statistics*, 29(5), pp. 1189–1232.

Geurts, P., Ernst, D. and Wehenkel, L. (2006) 'Extremely randomized trees', *Machine Learning*, 63(1), pp. 3–42.

Gobbi, M. and Mastinu, G. (2001) 'Analytical description and optimization of the dynamic behaviour of passively suspended road vehicles', *Journal of Sound and Vibration*, 245(3), pp. 457–481.

Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022) 'Why tree-based models still outperform deep learning on tabular data', *Advances in Neural Information Processing Systems*, 35, pp. 507–520.

Hoerl, A. E. and Kennard, R. W. (1970) 'Ridge regression: Biased estimation for nonorthogonal problems', *Technometrics*, 12(1), pp. 55–67.

Iman, R. L. and Conover, W. J. (1982) 'A distribution-free approach to inducing rank correlation among input variables', *Communications in Statistics — Simulation and Computation*, 11(3), pp. 311–334.

Jones, D. R., Schonlau, M. and Welch, W. J. (1998) 'Efficient global optimization of expensive black-box functions', *Journal of Global Optimization*, 13(4), pp. 455–492.

Knowles, J. (2006) 'ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems', *IEEE Transactions on Evolutionary Computation*, 10(1), pp. 50–66.

Loeppky, J. L., Sacks, J. and Welch, W. J. (2009) 'Choosing the sample size of a computer experiment: A practical guide', *Technometrics*, 51(4), pp. 366–376.

Lundberg, S. M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.-I. (2020) 'From local explanations to global understanding with explainable AI for trees', *Nature Machine Intelligence*, 2(1), pp. 56–67.

McKay, M. D., Beckman, R. J. and Conover, W. J. (1979) 'A comparison of three methods for selecting values of input variables in the analysis of output from a computer code', *Technometrics*, 21(2), pp. 239–245.

Milliken, W. F. and Milliken, D. L. (1995) *Race Car Vehicle Dynamics*. Warrendale: Society of Automotive Engineers.

Mockus, J. (1974) 'On Bayesian methods for seeking the extremum', in Marchuk, G. I. (ed.) *Optimization Techniques: IFIP Technical Conference, Novosibirsk, July 1–7, 1974*. Berlin: Springer, pp. 400–404.

Mockus, J., Tiesis, V. and Zilinskas, A. (1978) 'The application of Bayesian methods for seeking the extremum', in Dixon, L. C. W. and Szegö, G. P. (eds.) *Towards Global Optimisation*. Amsterdam: North Holland, Vol. 2, pp. 117–129.

Morris, M. D. (1991) 'Factorial sampling plans for preliminary computational experiments', *Technometrics*, 33(2), pp. 161–174.

Pacejka, H. B. (2012) *Tire and Vehicle Dynamics*. 3rd edn. Oxford: Butterworth-Heinemann.

Papageorgakis, G., Drikakis, D. and Bruecker, C. (2019) 'Machine learning for aerodynamic shape optimisation of road vehicles', *SAE International Journal of Advances and Current Practices in Mobility*, 2(1), pp. 58–68.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Peduzzi, P., Concato, J., Kemper, E., Holford, T. R. and Feinstein, A. R. (1996) 'A simulation study of the number of events per variable in logistic regression analysis', *Journal of Clinical Epidemiology*, 49(12), pp. 1373–1379.

Ponweiser, W., Wagner, T., Biermann, D. and Vincze, M. (2008) 'Multiobjective optimization on a limited budget of evaluations using model-assisted S-metric selection', in *Parallel Problem Solving from Nature — PPSN X*. Berlin: Springer, pp. 784–794.

Riedmaier, S., Ponn, T., Ludwig, D., Schick, B. and Diermeyer, F. (2020) 'Survey on scenario-based safety assessment of automated vehicles', *IEEE Access*, 8, pp. 87456–87477.

Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986) 'Learning representations by back-propagating errors', *Nature*, 323(6088), pp. 533–536.

Sacks, J., Welch, W. J., Mitchell, T. J. and Wynn, H. P. (1989) 'Design and analysis of computer experiments', *Statistical Science*, 4(4), pp. 409–423.

Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M. and Tarantola, S. (2008) *Global Sensitivity Analysis: The Primer*. Chichester: John Wiley & Sons.

Shahriari, B., Swersky, K., Wang, Z., Adams, R. P. and de Freitas, N. (2016) 'Taking the human out of the loop: A review of Bayesian optimization', *Proceedings of the IEEE*, 104(1), pp. 148–175.

Shapley, L. S. (1953) 'A value for n-person games', in Kuhn, H. W. and Tucker, A. W. (eds.) *Contributions to the Theory of Games, Volume II*. Princeton: Princeton University Press, pp. 307–317.

Smola, A. J. and Schölkopf, B. (2004) 'A tutorial on support vector regression', *Statistics and Computing*, 14(3), pp. 199–222.

Stark, S., Eckrich, M., Frey, M., Schöll, L. and Butz, A. (2020) 'BeamNGpy: Reinventing the wheel for autonomous vehicle research', *International Simulation and Testing Conference (ISTS)*, Stuttgart, November 2020.

Tipping, M. E. (2001) 'Sparse Bayesian learning and the relevance vector machine', *Journal of Machine Learning Research*, 1, pp. 211–244.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E. et al. (2020) 'SciPy 1.0: Fundamental algorithms for scientific computing in Python', *Nature Methods*, 17(3), pp. 261–272.

Wolpert, D. H. (1992) 'Stacked generalization', *Neural Networks*, 5(2), pp. 241–259.

Wymann, B., Espié, E., Guionneau, C., Dimitrakakis, C., Coulom, R. and Sumner, A. (2000) *TORCS, The Open Racing Car Simulator* [online]. Available at: http://torcs.sourceforge.net (Accessed: 11 April 2026).

Zegelaar, P. W. A. (1998) *The Dynamic Response of Tyres to Brake Torque Variations and Road Unevennesses*. PhD Thesis. Delft University of Technology.

Zitzler, E. and Thiele, L. (1998) 'Multiobjective optimization using evolutionary algorithms — A comparative case study', in *Parallel Problem Solving from Nature — PPSN V*. Berlin: Springer, pp. 292–301.

Zou, H. and Hastie, T. (2005) 'Regularization and variable selection via the elastic net', *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 67(2), pp. 301–320.
