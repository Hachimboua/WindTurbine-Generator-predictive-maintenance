Wind Turbine Degradation Simulator Documentation
================================================

Overview
------------

This Python-based simulator models the degradation process of wind turbine components under various operational conditions. It calculates:

* Mechanical wear from torque and rotational forces
* Environmental impacts from temperature, humidity, and vibration
* Maintenance effects when degradation reaches critical thresholds

The tool processes time-series sensor data to predict component lifespan and visualize degradation patterns.

Key Features
----------------

* **Physics-Based Modeling**:
    * Calculates rotor RPM, torque, and power coefficient from wind speed
    * Models aerodynamic forces using tip-speed ratio and blade pitch

* **Degradation Algorithms**:
    * Combines mechanical, thermal, humidity, and cyclic stresses
    * Implements maintenance effects with configurable thresholds

* **Data Processing**:
    * Handles real or synthetic wind turbine data
    * Automatically estimates missing temperature data

* **Visualization**:
    * Generates 6-panel diagnostic plots
    * Highlights maintenance events on degradation curves

Code Structure
------------------

1. **Core Class: WindTurbineDegradationSimulator**

   **Initialization (__init__)**:
    * Sets 22 technical parameters (blade length, rated power, cut-in/cut-out speeds, etc.)
    * Initializes tracking variables:
        * ``degradation_value`` (0.0 to 1.0)
        * ``time_since_last_maintenance``
        * ``maintenance_events_count``

   **Key Methods**:

   .. list-table:: 
      :widths: 30 40 30
      :header-rows: 1

      * - Method
        - Purpose
        - Key Formulas
      * - ``calculate_rotor_rpm()``
        - Converts wind speed to rotor RPM
        - ``RPM = (tip_speed × 60) / (2π × blade_length)``
      * - ``calculate_cp()``
        - Computes power coefficient (efficiency)
        - Modified CP-lambda curve with pitch angle compensation
      * - ``calculate_system_torque()``
        - Determines mechanical torque
        - ``Torque = (0.5 × ρ × A × v³ × Cp) / ω``
      * - ``simple_cycle_factor()``
        - Quantifies thermal cycling damage
        - Weighted temperature differentials
      * - ``wind_turbine_generator_degradation()``
        - Updates degradation value
        - Combines 5 stress factors with Arrhenius temperature model

2. **Data Processing Pipeline**

   ``process_time_series()`` Workflow:

    * **Input Handling**:
        * Accepts CSV path or DataFrame
        * Calculates wind speed from (u,v) components if needed

    * **Default Values**:
        * Vibration: 5.0 (if missing)
        * Humidity: 50% (if missing)
        * Temperature: Synthetic daily cycle (if missing)

    * **Time-Step Processing**:
        * Calculates torque/RPM for each wind speed
        * Updates degradation using 5 stress factors
        * Triggers maintenance when threshold crossed

    * **Output**:
        * Original data + 6 new columns:
            * ``system_torque_knm``
            * ``generator_speed_rpm``
            * ``degradation``
            * ``cycle_factor``
            * ``maintenance_performed``
            * ``time_since_last_maintenance``

3. **Visualization & Reporting**

   ``plot_degradation_analysis()`` Outputs:

    * Degradation Timeline (with maintenance markers)
    * Wind Speed vs. Torque (operational envelope)
    * Temperature Profile (real/estimated)
    * Torque History (mechanical loading)
    * Humidity vs. Degradation Rate (correlation)
    * Generator Speed Distribution (RPM histogram)

   ``print_degradation_summary()``:

    * Final degradation level
    * Maintenance event count
    * Time since last maintenance
    * Cycle factor impact

Physics Models
------------------

**Power Coefficient (Cp) Calculation**:

.. code-block:: python

   Cp = 0.5176*(116/λ - 0.4*β - 5)*e^(-21/λ) + 0.0068*λ

Where:
* λ = Tip-speed ratio (TSR)
* β = Blade pitch angle (degrees)

**Degradation Rate**:

.. code-block:: python

   dD = (α1·mech_wear + α2·vibration² + α3·thermal_stress + α4·humidity_stress + α5·cycle_factor)

With temperature acceleration:

.. code-block:: python

   thermal_stress = exp[(Ea/kb)·(1/T_ref - 1/T_actual)]

**Maintenance Effect**:

.. code-block:: python

   new_degradation = current_degradation × (1 - maintenance_effectiveness)

Usage Examples
------------------

1. **With Real Data**

.. code-block:: python

   simulator = WindTurbineDegradationSimulator()
   df_results = simulator.process_time_series(csv_file_path="turbine_data.csv")
   simulator.plot_degradation_analysis(df_results)

2. **With Synthetic Data**

.. code-block:: python

   simulator = WindTurbineDegradationSimulator()
   df_synthetic = pd.DataFrame({
       'speed': np.random.weibull(2, 10000)*12,
       'temperature': 15 + 10*np.sin(np.linspace(0,20,10000))
   })
   df_results = simulator.process_time_series(df=df_synthetic)

3. **Parameter Customization**

.. code-block:: python

   simulator.params['maintenance_threshold'] = 0.7  # Change failure threshold
   simulator.params['air_density'] = 1.1  # High-altitude adjustment

Input/Output Specifications
-------------------------------

**Expected Input Columns**:

.. list-table:: 
   :widths: 20 15 40 25
   :header-rows: 1

   * - Column
     - Required
     - Description
     - Default if Missing
   * - ``speed`` or ``(u,v)``
     - Yes
     - Wind speed (m/s)
     - -
   * - ``temperature``
     - No
     - Ambient temp (°C)
     - Synthetic profile
   * - ``U``
     - No
     - Vibration (mm/s)
     - 5.0
   * - ``r``
     - No
     - Humidity (%)
     - 50%

**Output Columns Added**:

* ``system_torque_knm``: Shaft torque in kN·m
* ``generator_speed_rpm``: Output RPM
* ``degradation``: Cumulative damage (0-1)
* ``maintenance_performed``: Event counter
* ``time_since_last_maintenance``: Steps since last repair

Error Handling
------------------

* **Missing Wind Data**: Raises ValueError if no speed data detected
* **File Errors**: Falls back to synthetic data if CSV loading fails
* **Numerical Stability**: Handles division-by-zero in physics calculations
* **Visualization**: Gracefully handles missing data columns in plots

Maintenance Logic
---------------------

* **Triggers When**:
    * Degradation ≥ ``maintenance_threshold`` (default: 0.8)
    * Minimum ``time_between_maintenance`` (default: 200 steps) elapsed

* **Effect**:
    * Reduces degradation by ``maintenance_effectiveness`` (default: 90%)
    * Resets maintenance timer
    * Increments event counter