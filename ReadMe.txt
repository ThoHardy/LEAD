================================================================================
                    LEAD TOOLBOX - DOCUMENTATION
================================================================================

OVERVIEW
--------
The LEAD (Latent Evidence Accumulation Dynamics) toolbox provides a framework 
for modeling neural dynamics of evidence accumulation using latent variables 
extracted from EEG/MEG data. It combines multivariate pattern analysis (MVPA) 
with state-space modeling to study how the brain accumulates sensory evidence 
over time, particularly in the context of consciousness and perceptual 
decision-making.

LEAD addresses two key limitations of traditional evidence accumulation models:

1. LATENT ENCODING: Rather than assuming the "decision variable" is directly 
   observable (e.g., via raw gamma power), LEAD uses temporal generalization 
   of multivariate classifiers to extract 1D time-series representing the 
   latent activation of specific encoding patterns through time.

2. NONLINEAR DYNAMICS: Instead of fixed decision criteria, LEAD incorporates 
   nonlinear dynamics with saddle-node bifurcations, enabling the model to 
   capture spontaneous transitions and metastable states relevant for 
   consciousness research where no explicit task is required.


================================================================================
THEORETICAL BACKGROUND
================================================================================

MULTIVARIATE PATTERN ANALYSIS (MVPA)
-------------------------------------
The toolbox leverages MVPA techniques to decode latent neural representations
and extract two complementary pieces of information:

1. LATENT ACTIVATION (Decision Function):
   The decision function of a linear classifier provides a 1D time-series 
   representing the strength of evidence for a particular cognitive state.
   This serves as the "latent variable" that feeds into LEAD models.

2. ENCODING PATTERN (Forward Model):
   While classifier weights (backward model) are optimized for discrimination,
   they are not directly interpretable as the neural pattern encoding the 
   cognitive state. To obtain the true encoding pattern, classifier weights 
   must be transformed into a forward model using the procedure described by 
   Haufe et al. (2014).
   
   Forward model transformation:
   A = Σ_x * W * Σ_s^(-1)
   
   where:
   - W = classifier weights (backward model)
   - Σ_x = covariance of neural data
   - Σ_s = covariance of latent activations
   - A = activation pattern (forward model, neurophysiologically interpretable)
   
   The forward model reveals which brain regions/sensors actually generate 
   the discriminative signal, whereas backward model weights can be non-zero 
   even at channels statistically independent of the process of interest.

- Reference: Haufe et al. (2014) "On the interpretation of weight vectors of 
  linear models in multivariate neuroimaging" 
  https://pubmed.ncbi.nlm.nih.gov/24239590/

TEMPORAL GENERALIZATION
------------------------
Classifiers trained on one time window are tested across all time points to 
reveal the temporal dynamics of neural representations:

- Temporal generalization matrices reveal when encoding patterns emerge, 
  persist, and transform
- Reference: King & Dehaene (2014) "Temporal generalization" 
  https://pubmed.ncbi.nlm.nih.gov/24593982/

BIFURCATION-BASED DYNAMICS
---------------------------
Traditional evidence accumulation models use fixed decision thresholds, which 
are problematic for consciousness research where:

- No explicit decision is required from subjects
- Neural dynamics exhibit spontaneous transitions between states
- Metastable states emerge without external triggers

LEAD models incorporate saddle-node bifurcations through sigmoid nonlinearities,
allowing the system to exhibit:

- Spontaneous ignition when evidence crosses a critical threshold
- Bistability between low and high activity states
- Hysteresis effects consistent with conscious access

Reference: Sergent et al. (2021) "Bifurcation in brain dynamics reveals a 
signature of conscious processing" https://pubmed.ncbi.nlm.nih.gov/33608533/


================================================================================
PACKAGE STRUCTURE
================================================================================

LEAD/
├── model.py              - Core model implementations and UKF engine
├── fitting_tools.py      - Advanced fitting strategies for model optimization
├── dataprocess.py        - MVPA and temporal generalization utilities
├── visual.py             - Visualization functions for trajectories and distributions
└── __init__.py           - Package initialization


================================================================================
1. MODEL.PY - CORE MODELING FRAMEWORK
================================================================================

This module contains the UKF engine, abstract base class, and concrete model
implementations for latent evidence accumulation dynamics.

--------------------------------------------------------------------------------
1.1 UKF ENGINE
--------------------------------------------------------------------------------

compute_ukf_loglikelihood(state_series, input_series, dt, process_noise_scalar,
                          measure_noise_scalar, transition_function_factory,
                          n_jobs=8, batch_size=20)

    Computes log-likelihood using parallelized Unscented Kalman Filter.
    
    The UKF handles nonlinear state-space models by propagating uncertainty 
    through sigma points, enabling accurate likelihood estimation for models 
    with sigmoid nonlinearities and bifurcation dynamics.
    
    Parameters:
        state_series         - Dict mapping category → (n_trials, n_timesteps) arrays
                              These are the latent variables extracted via MVPA
        input_series         - Dict mapping category → (n_trials, n_timesteps) arrays
                              External inputs (e.g., stimulus presence indicators)
        dt                   - Time step duration (default: 0.01 = 10ms)
        process_noise_scalar - Standard deviation of process noise
        measure_noise_scalar - Standard deviation of measurement noise
        transition_function_factory - Function that creates model dynamics
        n_jobs               - Number of parallel jobs (default: 8)
        batch_size           - Trials per batch (default: 20)
    
    Returns:
        Total log-likelihood across all trials and categories

--------------------------------------------------------------------------------
1.2 LEAD_abstract - BASE CLASS
--------------------------------------------------------------------------------

Abstract base class that all LEAD models inherit from. Provides common 
functionality for parameter management, simulation, and fitting.

CORE PARAMETERS (all models):
    tau              - Time constant for leaky integration (ms)
    process_noise    - Process noise standard deviation (intrinsic variability)
    measure_noise    - Measurement noise standard deviation (observation noise)

ABSTRACT METHODS (must be implemented by subclasses):
    input_function(input_value, signal_category)
        Defines how external input affects the dynamics
        Typically: linear weighting of stimulus strength
    
    nonlinearity(state, input_value, signal_category)
        Defines nonlinear state-dependent dynamics
        Typically: sigmoid functions creating bifurcation behavior
    
    loglikelihood(state_series, input_series)
        Computes model log-likelihood using UKF

CORE DYNAMICS:
    core(state, input_value, signal_category)
        Implements the fundamental differential equation:
        
        dx/dt = -x/tau + f(input) + g(x, input)
        
        where:
        - First term: Leaky integration (exponential decay toward baseline)
        - Second term: Linear input drive (bottom-up sensory evidence)
        - Third term: Nonlinear feedback (recurrent amplification/bifurcation)
        
        Uses Euler-Maruyama approximation for discrete time steps

SIMULATION METHODS:
    euler_maruyama(input_series, n_trials=1)
        Generates simulated latent trajectories with process noise
        Useful for:
        - Model validation and sanity checks
        - Generating predictions for new stimuli
        - Understanding model behavior across parameter space
        
        Returns: Dict of simulated states for each category
    
    measure_simulations(simulations)
        Adds measurement noise to simulated states
        Simulates the observation process (e.g., classifier variability)
        
        Returns: Dict of noisy observations

PARAMETER MANAGEMENT:
    get_params() / get_parameters()
        Returns dict of all model parameters
    
    set_params(params_dict) / set_parameters(params_dict)
        Updates parameters from dictionary
    
    set_params_from_list(params_list)
        Updates parameters from list (order follows _param_names)
    
    save_params(save_path)
        Saves parameters to JSON file
    
    load_params(save_path)
        Loads parameters from JSON file

FITTING METHOD:
    fit(state_series, input_series, init_params, bounds, 
        fixed_params=[], feedback=False)
        
        Fits model parameters by maximizing likelihood via L-BFGS-B optimization
        
        Parameters:
            state_series  - Training data states (latent variables from MVPA)
            input_series  - Training data inputs
            init_params   - Initial parameter values (list)
            bounds        - Parameter bounds (list of tuples)
            fixed_params  - Parameters to keep fixed (list of names)
            feedback      - Print optimization progress (bool)

--------------------------------------------------------------------------------
1.3 CONCRETE MODEL IMPLEMENTATIONS
--------------------------------------------------------------------------------

All models follow the general form:
    dx/dt = -x/tau + f(input, category) + g(x, input, category)

Models are organized by complexity, from linear baselines to bifurcation models.


--- StratifiedLinear ---

    Baseline linear model with category-specific input weights.
    
    dx/dt = -x/tau + w_category * input
    
    Parameters: tau, process_noise, measure_noise, w0, w1, w2, w3, w4, w5, w6
    
    Use case: 
        - Baseline model for comparison
        - Tests whether simple linear accumulation is sufficient
        - Category-specific weights capture different sensitivities to 
          stimulus intensities
    
    Special methods:
        loglikelihood_kalman() - Exact analytical solution using Kalman Filter
                                 (for verification and debugging)


--- NonLinear1 ---

    Single-category model with sigmoid feedback (basic bifurcation model).
    
    dx/dt = -x/tau + w*input + gain/(1 + exp(sharpness*(threshold - x)))
    
    Parameters: tau, process_noise, measure_noise, input_weight, gain, 
                threshold, sharpness
    
    Bifurcation mechanism:
        - When x < threshold: minimal feedback (low activity state)
        - When x ≈ threshold: rapid transition (saddle-node bifurcation)
        - When x > threshold: strong feedback (high activity state)
    
    Use case: 
        - Models conscious access as ignition phenomenon
        - Captures all-or-none transitions
        - Suitable for single stimulus category experiments


--- StratifiedNonLinear1 ---

    Multi-category version of NonLinear1 with category-specific input weights.
    
    dx/dt = -x/tau + w_category*input + gain/(1 + exp(sharpness*(threshold - x)))
    
    Parameters: tau, process_noise, measure_noise, gain, threshold, sharpness,
                w0, w1, w2, w3, w4, w5, w6
    
    Use case: 
        - Different stimulus intensities with shared bifurcation threshold
        - Models graded bottom-up input with all-or-none ignition
        - Appropriate for experiments with multiple stimulus strengths


--- NonLinear2 ---

    State-dependent gain modulation (linear feedback amplification).
    
    dx/dt = -x/tau + w*input + (a*x + b)*sigmoid(x)
    
    Parameters: tau, process_noise, measure_noise, input_weight, a, b, 
                threshold, sharpness
    
    Bifurcation mechanism:
        - Feedback strength scales linearly with current state
        - Creates richer dynamics with potential for oscillations
        - Parameter 'a' controls self-amplification strength
    
    Use case: 
        - Models where feedback depends on accumulated evidence
        - Can exhibit damped oscillations or overshooting


--- StratifiedNonLinear2 ---

    Multi-category version of NonLinear2.
    
    Parameters: tau, process_noise, measure_noise, a, b, threshold, sharpness,
                w0, w1, w2, w3, w4, w5, w6
    
    Use case:
        - Combines category-specific inputs with state-dependent amplification


--- GainModulation ---

    Input-dependent gain modulation (multiplicative amplification).
    
    dx/dt = -x/tau + w*input + gain*input*sigmoid(x)
    
    Parameters: tau, process_noise, measure_noise, input_weight, gain, 
                threshold, sharpness
    
    Bifurcation mechanism:
        - Feedback is gated by both state (sigmoid) and input
        - Input effectiveness increases with accumulated evidence
        - Creates input-dependent bifurcation points
    
    Use case: 
        - Models where stimulus presence modulates recurrent amplification
        - Captures context-dependent ignition


--- StratifiedGainModulation ---

    Multi-category gain modulation with category-specific weights and gains.
    
    dx/dt = -x/tau + w_category*input + g_category*sigmoid(x)
    
    Parameters: tau, process_noise, measure_noise, threshold, sharpness,
                w0-w6 (input weights), g0-g6 (gain parameters)
    
    Use case: 
        - Most flexible model with category-specific linear and nonlinear terms
        - Different stimulus categories can have different bifurcation strengths
        - Appropriate when both bottom-up and recurrent processing vary 
          across conditions


================================================================================
2. FITTING_TOOLS.PY - ADVANCED FITTING STRATEGIES
================================================================================

Nonlinear models with bifurcations are challenging to fit due to:
- Multiple local minima in likelihood landscape
- Strong parameter interactions (especially threshold and gain)
- Sensitivity to initialization

These "clever" fitting strategies exploit model structure for robust estimation.

--------------------------------------------------------------------------------

clever_fit_linear(state_train, input_train, n_loops=2, 
                  input_start_index=75, input_stop_index=100)

    Optimized fitting strategy for StratifiedLinear models.
    
    Strategy:
        1. Fit noise parameters (process_noise, measure_noise) on category 0 
           (resting state where w0=0, so only noise contributes)
        2. Fix noise parameters and fit input weights w1-w6 using data from 
           active stimulus categories
        3. Iterate to refine all parameters jointly
    
    Rationale:
        - Separating noise estimation from weight estimation improves 
          identifiability
        - Resting state provides clean estimate of intrinsic noise
    
    Parameters:
        state_train        - Training state data (latent variables)
        input_train        - Training input data
        n_loops            - Number of refinement iterations (default: 2)
        input_start_index  - Start of input window for weight estimation
        input_stop_index   - End of input window for weight estimation
    
    Returns:
        Fitted StratifiedLinear model


clever_fit_gainmodul(linear_prefitted, state_train, input_train, n_loops=2,
                     input_start_index=75, input_stop_index=100, feedback=False)

    Optimized fitting for StratifiedGainModulation models.
    
    Strategy:
        1. Initialize from pre-fitted linear model (inherits tau, noise, weights)
        2. Grid search over threshold values (critical bifurcation parameter)
        3. For each threshold candidate:
           a. Fix threshold and sharpness
           b. Alternate between fitting:
              - Linear weights (w0-w6) with gains fixed
              - Gain parameters (g0-g6) with weights fixed
           c. Iterate n_loops times
        4. Select threshold with highest likelihood
        5. Final joint optimization of all parameters
    
    Rationale:
        - Threshold is the most critical parameter for bifurcation behavior
        - Grid search avoids local minima in threshold space
        - Alternating optimization exploits parameter structure
        - Warm start from linear model provides sensible initialization
    
    Parameters:
        linear_prefitted   - Pre-fitted StratifiedLinear model
        state_train        - Training state data
        input_train        - Training input data
        n_loops            - Iterations per threshold value (default: 2)
        input_start_index  - Start of input window
        input_stop_index   - End of input window
        feedback           - Print progress information (bool)
    
    Returns:
        Fitted StratifiedGainModulation model


clever_fit_nonlinear1(linear_prefitted, state_train, input_train, n_loops=2,
                      input_start_index=75, input_stop_index=100, feedback=False)

    Optimized fitting for StratifiedNonLinear1 models.
    
    Strategy:
        1. Initialize from pre-fitted linear model
        2. Grid search over threshold values
        3. For each threshold:
           a. Alternate between fitting:
              - Input weights (w0-w6) with nonlinear params fixed
              - Nonlinear parameters (gain, sharpness) with weights fixed
           b. Iterate n_loops times
        4. Select best threshold
        5. Final joint optimization
    
    Rationale:
        - Similar to clever_fit_gainmodul but for additive (not multiplicative) 
          nonlinearity
        - Separates bottom-up input estimation from recurrent amplification
    
    Parameters:
        Same as clever_fit_gainmodul
    
    Returns:
        Fitted StratifiedNonLinear1 model


================================================================================
3. DATAPROCESS.PY - MVPA AND TEMPORAL GENERALIZATION
================================================================================

This module implements the data preprocessing pipeline that extracts latent 
variables from EEG/MEG data using multivariate pattern analysis.

--------------------------------------------------------------------------------

STG(data_ref, tmin, tmax, substract_pattern=None)

    Segmented Temporal Generalization analysis.
    
    Extracts 1D latent time-series representing the temporal dynamics of 
    specific encoding patterns identified by multivariate classifiers.
    
    Method:
        1. Train L2-regularized logistic regression classifier on EEG/MEG data 
           in the [tmin, tmax] window to discriminate stimulus present vs absent
        2. Apply trained classifier to all time points to obtain decision 
           function values (signed distance from decision boundary)
        3. Use leave-one-block-out cross-validation (20 blocks) to avoid 
           overfitting
        4. Demean and normalize relative to resting state (category 0)
        5. Return time-series for each stimulus category
    
    The decision function values can be interpreted as the latent activation 
    of the encoding pattern learned in the [tmin, tmax] window, generalized 
    across time.
    
    Parameters:
        data_ref          - Path to MNE epochs file (.fif format)
        tmin              - Start time (ms) for training window
        tmax              - End time (ms) for training window
        substract_pattern - Optional (tmin2, tmax2) tuple to remove a specific
                           encoding pattern via orthogonal projection
                           (useful for isolating late vs early components)
    
    Returns:
        Dict mapping category → (n_trials, n_timesteps) decision values
        - Category 0: resting state (no stimulus)
        - Categories 1-6: increasing stimulus intensities
        - Values are z-scored relative to category 0
    
    Implementation details:
        - Decimates data by factor of 5 for computational efficiency
        - Uses StandardScaler for feature normalization
        - Classifier: LogisticRegression(solver='liblinear', penalty='l2')
        - Cross-validation: leave-one-block-out (20 blocks)
    
    Theoretical interpretation:
        The decision function at time t reflects how strongly the neural 
        activity pattern at t resembles the pattern that discriminates 
        stimulus presence in the [tmin, tmax] window. This provides a 
        continuous measure of evidence accumulation without assuming a 
        specific neural substrate (e.g., gamma power).


================================================================================
4. VISUAL.PY - VISUALIZATION TOOLS
================================================================================

Visualization functions for exploring latent dynamics and model predictions.

--------------------------------------------------------------------------------

colormap(cat)

    Returns standard RGB color for a given category.
    
    Parameters:
        cat - Category index (0-6)
    
    Returns:
        RGB tuple
    
    Color scheme:
        0: Black (resting state / no stimulus)
        1: Blue (lowest stimulus intensity)
        2: Cyan
        3: Yellow-green
        4: Orange
        5: Red
        6: Dark red (highest stimulus intensity)


trajectories(time_series, trials=None, avg=False, cursor=None, 
             save_path=None, title=None, figsize=(10,5))

    Plots latent state trajectories over time.
    
    Parameters:
        time_series - Dict mapping category → (n_trials, n_timesteps) arrays
                     Can be observed data or model simulations
        trials      - Dict specifying which trials to plot per category
                     Example: {0: [0,1,2], 5: [0,1,2,3,4]}
                     If None, plots all trials
        avg         - If True, overlay category averages as solid lines
        cursor      - List of time points to mark with vertical dashed lines
                     (e.g., stimulus onset, training window boundaries)
        save_path   - Optional path to save figure
        title       - Optional plot title
        figsize     - Figure size tuple (default: (10,5))
    
    Displays:
        - Individual trajectories (dotted lines if avg=True)
        - Category averages (solid lines if avg=True)
        - Time axis from -500 to 2000 ms
        - Color-coded by stimulus category
        - Legend showing category labels
    
    Use cases:
        - Visualize observed latent dynamics from MVPA
        - Compare model simulations to data
        - Identify bifurcation points and ignition times
        - Assess trial-to-trial variability


densities(time_series, xlims=None, tlims=None, save_path=None, 
          title=None, figsize=(10,3))

    Plots kernel density estimates of state distributions.
    
    Useful for visualizing bistability and bifurcation structure.
    
    Parameters:
        time_series - Dict mapping category → (n_trials, n_timesteps) arrays
        xlims       - Optional (xmin, xmax) for x-axis limits
        tlims       - Optional (tmin, tmax) time window for averaging states
                     If specified, averages states within this window before 
                     computing density
        save_path   - Optional path to save figure
        title       - Optional plot title
        figsize     - Figure size tuple (default: (10,3))
    
    Displays:
        - Kernel density estimate for each category
        - Color-coded by stimulus category
        - X-axis: latent state value
        - Y-axis: probability density
    
    Use cases:
        - Identify bimodal distributions (signature of bistability)
        - Compare state distributions across stimulus intensities
        - Visualize separation between low and high activity states
        - Assess whether bifurcation models are appropriate


================================================================================
TYPICAL WORKFLOW
================================================================================

1. DATA EXTRACTION (MVPA)
   ----------------------
   - Load EEG/MEG epochs using MNE
   - Apply STG() to extract latent time-series
   - Choose training window [tmin, tmax] based on:
     * Prior knowledge of encoding timing
     * Temporal generalization matrices
     * Peak decoding performance
   
   Example:
   >>> from LEAD import STG
   >>> latent_vars = STG('subject01_epochs.fif', tmin=75, tmax=100)

2. DATA PREPARATION
   ----------------
   - Organize latent variables into state_series dictionary
   - Create input_series (typically binary: stimulus on/off)
   - Split into training and test sets if desired
   
   Example:
   >>> state_train = latent_vars
   >>> input_train = {cat: np.ones_like(states) for cat, states in state_train.items()}

3. LINEAR BASELINE
   ---------------
   - Always start with StratifiedLinear model
   - Provides baseline likelihood for model comparison
   - Parameters serve as initialization for nonlinear models
   
   Example:
   >>> from LEAD.fitting_tools import clever_fit_linear
   >>> linear_model = clever_fit_linear(state_train, input_train, n_loops=3)
   >>> baseline_ll = linear_model.loglikelihood(state_train, input_train)

4. NONLINEAR MODELS (if needed)
   ----------------------------
   - Fit bifurcation models if linear model is insufficient
   - Choose model based on theoretical considerations:
     * StratifiedNonLinear1: additive feedback (conscious ignition)
     * StratifiedGainModulation: multiplicative feedback (context modulation)
   
   Example:
   >>> from LEAD.fitting_tools import clever_fit_nonlinear1
   >>> bifurc_model = clever_fit_nonlinear1(linear_model, state_train, 
   ...                                      input_train, feedback=True)
   >>> bifurc_ll = bifurc_model.loglikelihood(state_train, input_train)

5. MODEL COMPARISON
   ----------------
   - Compare log-likelihoods (higher is better)
   - Consider model complexity (number of parameters)
   - Use BIC or AIC for formal comparison
   - Validate on held-out test data if available
   
   Example:
   >>> print(f"Linear LL: {baseline_ll:.2f}")
   >>> print(f"Bifurcation LL: {bifurc_ll:.2f}")
   >>> print(f"Improvement: {bifurc_ll - baseline_ll:.2f}")

6. VISUALIZATION AND INTERPRETATION
   --------------------------------
   - Plot observed data and model simulations
   - Identify bifurcation threshold from fitted parameters
   - Examine state distributions for bistability
   - Analyze ignition times (when trajectories cross threshold)
   
   Example:
   >>> from LEAD.visual import trajectories, densities
   >>> trajectories(state_train, avg=True, title="Observed Data")
   >>> 
   >>> simulated = bifurc_model.euler_maruyama(input_train, n_trials=100)
   >>> trajectories(simulated, avg=True, title="Model Predictions")
   >>> 
   >>> densities(state_train, tlims=(100, 150), title="State Distribution")

7. PARAMETER ANALYSIS
   ------------------
   - Extract fitted parameters
   - Compare across subjects or conditions
   - Relate parameters to behavioral or clinical measures
   
   Example:
   >>> params = bifurc_model.get_params()
   >>> print(f"Bifurcation threshold: {params['threshold']:.3f}")
   >>> print(f"Amplification gain: {params['gain']:.3f}")
   >>> bifurc_model.save_params('subject01_params.json')


================================================================================
EXAMPLE: COMPLETE ANALYSIS PIPELINE
================================================================================

# Import toolbox
from LEAD import STG
from LEAD.fitting_tools import clever_fit_linear, clever_fit_nonlinear1
from LEAD.visual import trajectories, densities
import numpy as np

# 1. Extract latent variables via MVPA
print("Extracting latent variables...")
latent_vars = STG('data/subject01_epochs.fif', tmin=75, tmax=100)

# 2. Prepare data
state_series = latent_vars
input_series = {cat: np.ones_like(states) for cat, states in state_series.items()}

# 3. Fit linear baseline
print("Fitting linear model...")
linear_model = clever_fit_linear(state_series, input_series, n_loops=3)
ll_linear = linear_model.loglikelihood(state_series, input_series)
print(f"Linear model log-likelihood: {ll_linear:.2f}")

# 4. Fit bifurcation model
print("Fitting bifurcation model...")
bifurc_model = clever_fit_nonlinear1(linear_model, state_series, input_series,
                                     n_loops=3, feedback=True)
ll_bifurc = bifurc_model.loglikelihood(state_series, input_series)
print(f"Bifurcation model log-likelihood: {ll_bifurc:.2f}")
print(f"Likelihood improvement: {ll_bifurc - ll_linear:.2f}")

# 5. Analyze parameters
params = bifurc_model.get_params()
print(f"\nFitted parameters:")
print(f"  Time constant (tau): {params['tau']:.1f} ms")
print(f"  Bifurcation threshold: {params['threshold']:.3f}")
print(f"  Amplification gain: {params['gain']:.3f}")
print(f"  Sharpness: {params['sharpness']:.3f}")

# 6. Visualize results
print("\nGenerating visualizations...")

# Observed data
trajectories(state_series, avg=True, cursor=[0, 75, 100],
            title="Observed Latent Dynamics")

# Model simulations
simulated = bifurc_model.euler_maruyama(input_series, n_trials=100)
trajectories(simulated, avg=True, cursor=[0, 75, 100],
            title="Bifurcation Model Predictions")

# State distributions
densities(state_series, tlims=(100, 150), xlims=(-2, 4),
         title="State Distribution (100-150ms)")

# 7. Save results
bifurc_model.save_params('results/subject01_bifurcation_params.json')
print("\nAnalysis complete!")


================================================================================
INTERPRETING BIFURCATION PARAMETERS
================================================================================

THRESHOLD
---------
The critical state value where the bifurcation occurs.

- Low threshold: Easy ignition, even weak stimuli trigger transitions
- High threshold: Difficult ignition, requires strong evidence
- Clinical relevance: May relate to conscious access threshold

GAIN
----
The strength of recurrent amplification above threshold.

- High gain: Strong self-amplification, rapid ignition
- Low gain: Weak amplification, gradual transitions
- Theoretical: Related to recurrent connectivity strength

SHARPNESS
---------
The steepness of the sigmoid transition.

- High sharpness: Abrupt all-or-none transitions (true bifurcation)
- Low sharpness: Gradual smooth transitions (no bifurcation)
- Interpretation: Degree of nonlinearity in recurrent dynamics

TAU
---
Time constant for leaky integration.

- Large tau: Slow dynamics, long memory
- Small tau: Fast dynamics, short memory
- Relates to: Temporal integration window for evidence accumulation

INPUT WEIGHTS (w0-w6)
---------------------
Category-specific sensitivities to bottom-up input.

- Should increase monotonically with stimulus intensity
- Reflects: Strength of feedforward sensory drive
- Category 0 (w0): Should be zero (resting state)


================================================================================
TECHNICAL NOTES
================================================================================

TIME DISCRETIZATION
-------------------
- Default dt = 0.01 (10ms time steps)
- Matches typical EEG/MEG sampling after decimation
- Euler-Maruyama scheme for stochastic integration

NOISE PARAMETERS
----------------
- Process noise: Intrinsic variability in latent dynamics
- Measurement noise: Variability in MVPA decoder output
- Both estimated from data during fitting

UKF IMPLEMENTATION
------------------
- Uses JulierSigmaPoints with kappa=0.0 for numerical stability
- Handles nonlinear dynamics via sigma point propagation
- Critical for accurate likelihood computation with bifurcations

PARALLEL PROCESSING
-------------------
- Uses joblib with 'loky' backend
- Default: 8 parallel jobs
- Batch size: 20 trials per batch
- Adjust based on available CPU cores

OPTIMIZATION
------------
- Algorithm: L-BFGS-B (handles box constraints)
- Maximizes log-likelihood (minimizes negative log-likelihood)
- Clever fitting strategies reduce local minima issues

CATEGORY CONVENTION
-------------------
- Category 0: Resting state (no stimulus)
- Categories 1-6: Increasing stimulus intensities
- Can be adapted to other experimental designs


================================================================================
DEPENDENCIES
================================================================================

Required packages:
- numpy              - Numerical computations
- scipy              - Optimization and signal processing
- matplotlib         - Plotting
- seaborn            - Statistical visualizations
- filterpy           - Unscented Kalman Filter implementation
- joblib             - Parallel processing
- scikit-learn       - MVPA classifiers and preprocessing
- mne                - EEG/MEG data handling
- pandas             - Data manipulation (dataprocess.py)

Install via:
    pip install numpy scipy matplotlib seaborn filterpy joblib scikit-learn mne pandas


================================================================================
REFERENCES
================================================================================

Multivariate Pattern Analysis:
    King, J. R., & Dehaene, S. (2014). Characterizing the dynamics of mental 
    representations: the temporal generalization method. Trends in Cognitive 
    Sciences, 18(4), 203-210.
    https://pubmed.ncbi.nlm.nih.gov/24239590/

Temporal Generalization:
    King, J. R., & Dehaene, S. (2014). A model of subjective report and 
    objective discrimination as categorical decisions in a vast representational 
    space. Philosophical Transactions of the Royal Society B, 369(1641).
    https://pubmed.ncbi.nlm.nih.gov/24593982/

Bifurcation Dynamics and Consciousness:
    Sergent, C., et al. (2021). Bifurcation in brain dynamics reveals a 
    signature of conscious processing independent of report. Nature 
    Communications, 12(1), 1149.
    https://pubmed.ncbi.nlm.nih.gov/33608533/


================================================================================
For questions, issues, or contributions, contact the development team.
================================================================================
