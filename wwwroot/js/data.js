    const projectsData = {
  "eee": {
    "name": "Electrical & Electronics Engineering (EEE)",
    "short": "EEE",
    "icon": "\u26a1",
    "color": "#f59e0b",
    "tagline": "Explore energy data, then build towards electrical-system AI.",
    "description": "Electrical principles remain the foundation. AI adds ways to explore recorded signals, compare forecasts, and investigate simulated control problems. Start with household appliance activity or thermal-image exploration; move to solar forecasting and motor signals; attempt grid stability or microgrid control after studying the prerequisites. Beginner means basic Python with guided examples; intermediate means independent data preparation and model evaluation; advanced means domain knowledge plus deeper modelling. Levels describe the starting scope, not a guaranteed completion time.",
    "projects": [
      {
        "id": "eee-01",
        "title": "Exploring Power Grid Stability with Physics-Informed AI",
        "difficulty": "Advanced",
        "summary": "Study how a small simulated power system responds to a disturbance. Start with a numerical simulation and a small neural-network baseline. When that works, add a physics-based loss and test unfamiliar disturbance cases. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a numerical simulation and a small neural-network baseline for this task? Keep the scope small enough to test fairly. Helpful background: Power-system dynamics, differential equations, and neural-network training.",
        "ai_technique": "Starting approach: a numerical simulation and a small neural-network baseline. Extension: add a physics-based loss and test unfamiliar disturbance cases.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Simulated trajectories from a documented small power-system model, with initial conditions and disturbance settings saved. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Prediction error, equation residuals, and runtime on the same test cases. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nclass GridPINN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(nn.Linear(39, 128), nn.GELU(), nn.Linear(128, 64), nn.Linear(64, 39))\n    def loss_physics(self, rotor_angle, omega, P_mech, P_elec, D, H):\n        # Residual of swing equation: 2H d(omega)/dt = P_m - P_e - D*omega\n        d_omega = torch.autograd.grad(omega, self.time, create_graph=True)[0]\n        return torch.mean((2*H*d_omega - (P_mech - P_elec - D*omega))**2)"
      },
      {
        "id": "eee-02",
        "title": "Learning to Manage Energy in a Simulated Microgrid",
        "difficulty": "Advanced",
        "summary": "Explore how a controller schedules a battery alongside solar power and household demand. Start with a rule-based battery schedule. When that works, train one reinforcement-learning agent before considering multiple agents. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a rule-based battery schedule for this task? Keep the scope small enough to test fairly. Helpful background: Python, energy balance, and reinforcement-learning basics.",
        "ai_technique": "Starting approach: a rule-based battery schedule. Extension: train one reinforcement-learning agent before considering multiple agents.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Time-aligned solar, demand, and tariff records, or clearly labelled synthetic profiles for a small simulation. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Simulated energy cost, unmet demand, and battery-limit violations. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom ray.rllib.algorithms.ppo import PPOConfig\nconfig = (PPOConfig().environment(MicrogridGymEnv)\n    .framework('torch')\n    .rollouts(num_rollout_workers=4)\n    .training(lr=3e-4, gamma=0.99, clip_param=0.2))\nalgo = config.build()"
      },
      {
        "id": "eee-03",
        "title": "Motor Bearing Fault Detection from Vibration Data",
        "difficulty": "Intermediate",
        "summary": "Use recorded vibration signals to distinguish healthy bearings from labelled faults. Start with signal features and a simple classifier. When that works, compare a small 1D convolutional network; try audio only if suitable labelled recordings are available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on signal features and a simple classifier for this task? Keep the scope small enough to test fairly. Helpful background: Python, basic signal processing, and classification.",
        "ai_technique": "Starting approach: signal features and a simple classifier. Extension: compare a small 1D convolutional network; try audio only if suitable labelled recordings are available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: A labelled bearing-vibration dataset; keep windows from the same recording together when splitting the data. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1, class-level recall, and errors on recordings from held-out machines or operating conditions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Edge feature extraction via FFT Spectrogram -> TinyCNN\ninput_data = apply_fft(mems_samples, window_size=256)\ninterpreter.set_tensor(input_index, input_data)\ninterpreter.invoke()\npredictions = interpreter.get_tensor(output_index) # [Healthy, InnerRace, OuterRace, BallFault]"
      },
      {
        "id": "eee-04",
        "title": "Solar Power Forecasting from Weather and Generation Data",
        "difficulty": "Intermediate",
        "summary": "Predict solar output for a clearly defined future time window. Start with a persistence forecast and a tree-based regressor. When that works, compare a temporal neural network or transformer if simpler models leave useful room for improvement. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a persistence forecast and a tree-based regressor for this task? Keep the scope small enough to test fairly. Helpful background: Pandas, regression, and time-based train/test splits.",
        "ai_technique": "Starting approach: a persistence forecast and a tree-based regressor. Extension: compare a temporal neural network or transformer if simpler models leave useful room for improvement.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Historical solar output and weather records with matching timestamps; include only information available at forecast time. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: MAE and RMSE across held-out dates, including cloudy periods. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet\ntraining_data = TimeSeriesDataSet(df, time_idx='time_step', target='power_mw', group_ids=['inverter_id'], max_encoder_length=96, max_prediction_length=24)"
      },
      {
        "id": "eee-05",
        "title": "Estimating Battery Health from Charging Records",
        "difficulty": "Intermediate",
        "summary": "Explore how charging measurements relate to a battery's remaining capacity. Start with cycle-level features and regression. When that works, compare a sequence model and investigate uncertainty. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on cycle-level features and regression for this task? Keep the scope small enough to test fairly. Helpful background: Regression, time-series preparation, and basic battery concepts.",
        "ai_technique": "Starting approach: cycle-level features and regression. Extension: compare a sequence model and investigate uncertainty.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Battery cycling records with measured capacity labels; split by battery rather than randomly mixing cycles. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Capacity-estimation error and performance on batteries excluded from training. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nmodel = Sequential([\n    Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, features)),\n    Dropout(0.2),\n    Bidirectional(LSTM(32)),\n    Dense(16, activation='relu'),\n    Dense(1, activation='linear') # Remaining Capacity / SOH\n])"
      },
      {
        "id": "eee-06",
        "title": "Neural Control for Harmonic Compensation in Simulation",
        "difficulty": "Advanced",
        "summary": "Investigate whether a neural controller can help reduce distortion in a simulated electrical system. Start with a documented conventional compensation method. When that works, compare a neural controller under changing simulated loads. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a documented conventional compensation method for this task? Keep the scope small enough to test fairly. Helpful background: Power electronics, control theory, and neural networks.",
        "ai_technique": "Starting approach: a documented conventional compensation method. Extension: compare a neural controller under changing simulated loads.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Simulated voltage and current waveforms from a documented model with known load and harmonic settings. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Total harmonic distortion, tracking error, and control-limit violations. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# High-frequency neural inference pipeline for FPGA synthesis (HLS)\n# void infer_harmonic_current(float i_load[3], float i_comp[3]);"
      },
      {
        "id": "eee-07",
        "title": "Finding Transmission-Line Defects in Inspection Images",
        "difficulty": "Intermediate",
        "summary": "Build an image-based prototype that highlights a small set of labelled line-component defects. Start with a transfer-learning image classifier. When that works, move to object detection when bounding-box annotations are available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a transfer-learning image classifier for this task? Keep the scope small enough to test fairly. Helpful background: Python, image preparation, and basic deep learning.",
        "ai_technique": "Starting approach: a transfer-learning image classifier. Extension: move to object detection when bounding-box annotations are available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted inspection images with clear class or bounding-box annotations; begin with a small labelled subset. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision, recall, missed defects, and performance across held-out inspection sessions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom ultralytics import YOLO\nmodel = YOLO('yolov8n-obb-powerlines.pt')\nresults = model.predict(source='drone_feed_rtsp', conf=0.6, stream=True)"
      },
      {
        "id": "eee-08",
        "title": "Exploring Household Appliance Use from Power Readings",
        "difficulty": "Beginner",
        "summary": "Learn to plot household energy data and estimate whether one selected appliance is active. Start with simple power features and a threshold or logistic-regression baseline. When that works, compare a sequence model for appliance-level power estimation after the basic task works. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on simple power features and a threshold or logistic-regression baseline for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and an introduction to classification; advanced energy disaggregation is optional.",
        "ai_technique": "Starting approach: simple power features and a threshold or logistic-regression baseline. Extension: compare a sequence model for appliance-level power estimation after the basic task works.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: Recorded household mains readings with matching appliance measurements; start with one appliance and hold out entire days. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision and recall for appliance activity; MAE if you extend to power estimation. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nclass Seq2Point(nn.Module):\n    def __init__(self, window_size=99):\n        super().__init__()\n        self.conv = nn.Sequential(nn.Conv1d(1, 30, 10, dilation=1), nn.ReLU(), nn.Conv1d(30, 40, 8, dilation=2), nn.ReLU())\n        self.fc = nn.Sequential(nn.Linear(40 * 79, 1024), nn.ReLU(), nn.Linear(1024, 1))"
      },
      {
        "id": "eee-09",
        "title": "Exploring Hotspots in Thermal Inspection Images",
        "difficulty": "Beginner",
        "summary": "Use recorded thermal images to mark unusually warm regions for a classroom inspection demo. Start with a documented threshold and connected-region analysis. When that works, compare an anomaly detector or segmentation model with labelled examples. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a documented threshold and connected-region analysis for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python, arrays, and image handling; no live electrical equipment is needed.",
        "ai_technique": "Starting approach: a documented threshold and connected-region analysis. Extension: compare an anomaly detector or segmentation model with labelled examples.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: Permitted radiometric images with calibration information; colour-only images support image-pattern analysis, not reliable temperature measurement. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Agreement with labelled regions and false alarms across different scenes. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Process radiometric matrix: Celsius = (RAW - B) / R\ntemp_matrix = flir_sdk.extract_temperature(thermal_frame)\nanomaly_mask = temp_matrix > (ambient_temp + 35.0)\ncontours, _ = cv2.findContours(anomaly_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"
      },
      {
        "id": "eee-10",
        "title": "Optimising EV Charging Schedules in Simulation",
        "difficulty": "Advanced",
        "summary": "Build a small simulated charging depot and compare ways to schedule vehicles before departure. Start with first-come scheduling and a simple optimisation baseline. When that works, compare a reinforcement-learning policy on held-out arrival patterns. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on first-come scheduling and a simple optimisation baseline for this task? Keep the scope small enough to test fairly. Helpful background: Optimisation, Python, and basic charging-system constraints.",
        "ai_technique": "Starting approach: first-come scheduling and a simple optimisation baseline. Extension: compare a reinforcement-learning policy on held-out arrival patterns.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Synthetic or permitted arrival times, departure times, battery capacities, and tariffs, with assumptions recorded. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Simulated cost, departure charge shortfalls, and capacity-limit violations. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom stable_baselines3 import DDPG\nmodel = DDPG('MlpPolicy', EVDepotEnv, learning_rate=1e-3, buffer_size=50000, batch_size=64)\nmodel.learn(total_timesteps=100000)"
      },
      {
        "id": "eee-11",
        "title": "Smart Grid Load Forecasting",
        "difficulty": "Beginner",
        "summary": "Forecast the next day's electricity demand from a small historical table. Start with seasonal-naive forecasting and linear regression. When that works, compare a tree model using calendar and past-load features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on seasonal-naive forecasting and linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: seasonal-naive forecasting and linear regression. Extension: compare a tree model using calendar and past-load features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: timestamped electricity demand with clear units. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and peak-period error on future dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Smart Grid Load Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with seasonal-naive forecasting and linear regression.\n# 4. Optional extension: compare a tree model using calendar and past-load features.\n# 5. Evaluate MAE and peak-period error on future dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-12",
        "title": "Wind Energy Output Prediction",
        "difficulty": "Beginner",
        "summary": "Estimate recorded wind-turbine output from weather measurements. Start with linear regression. When that works, compare a tree model and inspect errors at low and high wind speeds. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree model and inspect errors at low and high wind speeds.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: paired wind measurements and turbine power records. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and RMSE on held-out dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Wind Energy Output Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree model and inspect errors at low and high wind speeds.\n# 5. Evaluate MAE and RMSE on held-out dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-13",
        "title": "Battery State-of-Charge Estimation",
        "difficulty": "Intermediate",
        "summary": "Estimate charge level from recorded voltage, current, and temperature. Start with a documented conventional estimator. When that works, compare a regression model on batteries or cycles held out from training. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented conventional estimator for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a documented conventional estimator. Extension: compare a regression model on batteries or cycles held out from training.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: battery logs with independently defined state-of-charge reference values. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Estimation error, drift, and performance across operating conditions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Battery State-of-Charge Estimation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented conventional estimator.\n# 4. Optional extension: compare a regression model on batteries or cycles held out from training.\n# 5. Evaluate estimation error, drift, and performance across operating conditions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-14",
        "title": "Electricity-Use Anomaly Detection",
        "difficulty": "Beginner",
        "summary": "Find unusual patterns in a household or building electricity record. Start with a seasonal threshold rule. When that works, compare an isolation forest and review flagged examples. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal threshold rule for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a seasonal threshold rule. Extension: compare an isolation forest and review flagged examples.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: anonymised meter readings with known anomalies or transparently injected test events. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: False alarms and detection of labelled test events; an anomaly does not prove theft. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Electricity-Use Anomaly Detection\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal threshold rule.\n# 4. Optional extension: compare an isolation forest and review flagged examples.\n# 5. Evaluate false alarms and detection of labelled test events; an anomaly does not prove theft.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-15",
        "title": "Power Quality Event Classification",
        "difficulty": "Intermediate",
        "summary": "Classify a small set of labelled voltage disturbances. Start with signal features and a simple classifier. When that works, compare a small convolutional network. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on signal features and a simple classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: signal features and a simple classifier. Extension: compare a small convolutional network.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: recorded or simulated waveforms labelled as normal, sag, swell, or interruption. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on held-out event recordings. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Power Quality Event Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with signal features and a simple classifier.\n# 4. Optional extension: compare a small convolutional network.\n# 5. Evaluate macro-F1 and errors on held-out event recordings.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-16",
        "title": "Transformer Condition Classification from Recorded Measurements",
        "difficulty": "Intermediate",
        "summary": "Explore how recorded measurements relate to expert condition labels. Start with logistic regression or a small decision tree. When that works, compare a tree ensemble and inspect uncertain cases. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression or a small decision tree for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: logistic regression or a small decision tree. Extension: compare a tree ensemble and inspect uncertain cases.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted transformer measurements with documented condition labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and class-level recall on held-out equipment. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Transformer Condition Classification from Recorded Measurements\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression or a small decision tree.\n# 4. Optional extension: compare a tree ensemble and inspect uncertain cases.\n# 5. Evaluate macro-F1 and class-level recall on held-out equipment.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-17",
        "title": "Building Electricity Consumption Prediction",
        "difficulty": "Beginner",
        "summary": "Predict daily electricity use from weather and building schedules. Start with a historical-average baseline and regression. When that works, compare feature sets and a tree model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a historical-average baseline and regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a historical-average baseline and regression. Extension: compare feature sets and a tree model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: building meter data with weather and schedule features available at prediction time. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on later weeks and errors during unusual operating days. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Building Electricity Consumption Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a historical-average baseline and regression.\n# 4. Optional extension: compare feature sets and a tree model.\n# 5. Evaluate MAE on later weeks and errors during unusual operating days.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-18",
        "title": "Underground Cable Fault Location in Simulation",
        "difficulty": "Advanced",
        "summary": "Estimate a fault's position from a simplified cable simulation. Start with a documented signal-based location method. When that works, compare a learned regressor across simulated fault conditions. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented signal-based location method for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a documented signal-based location method. Extension: compare a learned regressor across simulated fault conditions.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: simulated cable signals with known fault positions and model parameters. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Location error and robustness to simulated noise and parameter changes. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Underground Cable Fault Location in Simulation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented signal-based location method.\n# 4. Optional extension: compare a learned regressor across simulated fault conditions.\n# 5. Evaluate location error and robustness to simulated noise and parameter changes.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-19",
        "title": "Automatic Voltage Regulation with Reinforcement Learning",
        "difficulty": "Advanced",
        "summary": "Compare control strategies for a simulated voltage regulator. Start with a tuned PID controller. When that works, train a bounded reinforcement-learning policy on separate scenarios. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a tuned PID controller for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a tuned PID controller. Extension: train a bounded reinforcement-learning policy on separate scenarios.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a documented AVR simulation with disturbance records and action limits. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Tracking error, overshoot, settling time, and constraint violations. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Automatic Voltage Regulation with Reinforcement Learning\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a tuned PID controller.\n# 4. Optional extension: train a bounded reinforcement-learning policy on separate scenarios.\n# 5. Evaluate tracking error, overshoot, settling time, and constraint violations.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "eee-20",
        "title": "Household Energy Dashboard and Usage Forecast",
        "difficulty": "Beginner",
        "summary": "Create a dashboard that explains usage patterns and adds a simple next-day forecast. Start with daily summaries and a moving-average forecast. When that works, compare a regression forecast and add an error view. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on daily summaries and a moving-average forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: daily summaries and a moving-average forecast. Extension: compare a regression forecast and add an error view.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: anonymised or synthetic household meter readings. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Forecast error, correct aggregation, and clarity of the displayed units. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Household Energy Dashboard and Usage Forecast\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with daily summaries and a moving-average forecast.\n# 4. Optional extension: compare a regression forecast and add an error view.\n# 5. Evaluate forecast error, correct aggregation, and clarity of the displayed units.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "ece": {
    "name": "Electronics & Communication Engineering (ECE)",
    "short": "ECE",
    "icon": "\ud83d\udce1",
    "color": "#3b82f6",
    "tagline": "Start with signals you can see, hear, and measure.",
    "description": "Signal processing and communication theory help you understand what a model is learning. Use AI as an approach to compare with conventional estimators and detectors. Start with voice activity detection; progress to radio-signal classification or PCB images; try MIMO estimation and learned equalisation once you know the signal theory. Beginner means basic Python with guided examples; intermediate means independent data preparation and model evaluation; advanced means domain knowledge plus deeper modelling. Levels describe the starting scope, not a guaranteed completion time.",
    "projects": [
      {
        "id": "ece-01",
        "title": "Learning-Based Channel Estimation in a MIMO Simulation",
        "difficulty": "Advanced",
        "summary": "Estimate a simulated wireless channel from noisy pilot observations. Start with a conventional least-squares estimator. When that works, compare a compact neural estimator across unseen noise and channel conditions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a conventional least-squares estimator for this task? Keep the scope small enough to test fairly. Helpful background: Communication theory, complex-valued signals, and neural networks.",
        "ai_technique": "Starting approach: a conventional least-squares estimator. Extension: compare a compact neural estimator across unseen noise and channel conditions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Channel simulations with saved parameters, pilot signals, noise levels, and known channel targets. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Normalised estimation error and inference time under identical conditions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport sionna\nfrom sionna.channel import RayTracing\n# Deep autoencoder compressing complex H channel tensor into compact latent codeword\nencoded = self.encoder(csi_real_imag)\nfeedback_bits = quantize(encoded, num_bits=16)\nrecovered_csi = self.decoder(dequantize(feedback_bits))"
      },
      {
        "id": "ece-02",
        "title": "Detecting Signal Activity in Recorded Radio Samples",
        "difficulty": "Intermediate",
        "summary": "Classify whether a recorded or simulated signal window contains a transmission. Start with an energy detector. When that works, compare a feature-based classifier or small CNN across noise conditions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on an energy detector for this task? Keep the scope small enough to test fairly. Helpful background: Signal processing, Python, and classification.",
        "ai_technique": "Starting approach: an energy detector. Extension: compare a feature-based classifier or small CNN across noise conditions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Synthetic or permitted receive-only I/Q recordings with reliable signal-present labels. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Detection probability and false-alarm rate across held-out signal conditions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# GNU Radio to Python IQ buffer stream\niq_samples = sdr.read_samples(1024)\nfeatures = np.stack([np.real(iq_samples), np.imag(iq_samples)], axis=-1)\npred = model.predict(np.expand_dims(features, 0)) # [Primary User Active vs Idle]"
      },
      {
        "id": "ece-03",
        "title": "PCB Defect Detection from Labelled Images",
        "difficulty": "Intermediate",
        "summary": "Identify a manageable set of visible component or solder defects in circuit-board images. Start with image classification using transfer learning. When that works, compare object detection when you have labelled defect locations. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on image classification using transfer learning for this task? Keep the scope small enough to test fairly. Helpful background: Image preprocessing and basic deep learning.",
        "ai_technique": "Starting approach: image classification using transfer learning. Extension: compare object detection when you have labelled defect locations.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Labelled PCB images with consistent defect definitions; keep images of the same board in one data split. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Per-class recall, precision, and mistakes on unseen boards. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nresults = pcb_detector.predict(board_image, imgsz=1280, conf=0.45)\nfor box in results[0].boxes:\n    defect_class = pcb_detector.names[int(box.cls)]\n    # e.g., 'solder_bridge', 'missing_capacitor', 'tombstone'"
      },
      {
        "id": "ece-04",
        "title": "Comparing Spiking and Conventional Networks for Keyword Recognition",
        "difficulty": "Advanced",
        "summary": "Study keyword classification with a small audio vocabulary and compare two network approaches. Start with a small conventional audio classifier. When that works, implement a spiking-network experiment using the same splits. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a small conventional audio classifier for this task? Keep the scope small enough to test fairly. Helpful background: Audio features, neural-network training, and interest in spiking models.",
        "ai_technique": "Starting approach: a small conventional audio classifier. Extension: implement a spiking-network experiment using the same splits.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: A permitted keyword-audio dataset with speaker-disjoint training and test sets. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1, latency, and measured resource use; do not infer energy savings from spike counts alone. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport snntorch as snn\nlif1 = snn.Leaky(beta=0.9, spike_grad=snn.surrogate.fast_sigmoid())\nmem = lif1.init_leaky()\nfor step in range(num_steps):\n    spk, mem = lif1(input_spike_train[step], mem)"
      },
      {
        "id": "ece-05",
        "title": "AI-Assisted Beam Selection in a Satellite-Link Simulation",
        "difficulty": "Advanced",
        "summary": "Explore how a model selects a beam in a simplified changing satellite-link scenario. Start with a small beam codebook and conventional selection rule. When that works, compare a learned selector under varied simulated conditions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a small beam codebook and conventional selection rule for this task? Keep the scope small enough to test fairly. Helpful background: Antenna concepts, communication models, and optimisation.",
        "ai_technique": "Starting approach: a small beam codebook and conventional selection rule. Extension: compare a learned selector under varied simulated conditions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn"
        ],
        "data_pipeline": "Suggested data: Simulated channel and beam-codebook examples with documented geometry and propagation assumptions. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Link-quality metrics, selection errors, and runtime against the same baseline. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Compute complex weights for N-element phased array\nphase_angles = actor_network(state_telemetry)\nw = torch.exp(1j * phase_angles)\narray_factor = torch.matmul(steering_vector, w)"
      },
      {
        "id": "ece-06",
        "title": "Recognising Radio Devices from Recorded Signal Features",
        "difficulty": "Intermediate",
        "summary": "Test whether signal features help distinguish a small set of labelled devices in a research dataset. Start with a feature-based classifier. When that works, compare a CNN and test sensitivity to changing recording conditions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a feature-based classifier for this task? Keep the scope small enough to test fairly. Helpful background: Signal features, classification, and careful data splitting.",
        "ai_technique": "Starting approach: a feature-based classifier. Extension: compare a CNN and test sensitivity to changing recording conditions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Permitted labelled device recordings with device and recording-session identifiers. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1 and performance on different sessions; discuss why classification alone is not secure authentication. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Silicon transient extraction\ntransient_slice = extract_preamble_turn_on(rf_signal)\nembedding = rf_vit_model(transient_slice)\nsimilarity = cosine_distance(embedding, enrolled_device_embedding)"
      },
      {
        "id": "ece-07",
        "title": "Neural Signal Recovery for a Simulated Optical Link",
        "difficulty": "Advanced",
        "summary": "Compare conventional and learned ways to recover a signal distorted by a simplified optical-link model. Start with a conventional equaliser. When that works, add a neural equaliser and vary the simulated distortion. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a conventional equaliser for this task? Keep the scope small enough to test fairly. Helpful background: Digital communications, numerical simulation, and neural networks.",
        "ai_technique": "Starting approach: a conventional equaliser. Extension: add a neural equaliser and vary the simulated distortion.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Documented optical-link simulations containing transmitted targets and corresponding received signals. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Bit-error rate and computation cost across held-out link conditions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Learned DBP layer replacing split-step Fourier method\nclass LearnedDBPBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.dispersion = nn.Conv1d(2, 2, kernel_size=15, padding=7)\n        self.kerr_net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))"
      },
      {
        "id": "ece-08",
        "title": "Voice Activity Detection from Audio Clips",
        "difficulty": "Beginner",
        "summary": "Build a small tool that marks speech and non-speech sections in recorded audio. Start with an energy-based speech detector. When that works, compare a small classifier; add denoising as a separate extension. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on an energy-based speech detector for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and simple audio features.",
        "ai_technique": "Starting approach: an energy-based speech detector. Extension: compare a small classifier; add denoising as a separate extension.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: Permitted speech and background-noise recordings with speech-activity labels. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Frame-level precision, recall, and performance across held-out speakers and noise clips. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n// C embedded inference loop running on ARM Cortex-M\ntflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);\ninterpreter.AllocateTensors();\n// Audio circular buffer DMA -> FFT -> Run Model -> IFFT Output"
      },
      {
        "id": "ece-09",
        "title": "Estimating Indoor Position from Wi-Fi Measurements",
        "difficulty": "Intermediate",
        "summary": "Explore whether recorded Wi-Fi features can predict a device's position within a measured room. Start with nearest-neighbour position matching. When that works, compare a regressor or temporal neural model. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on nearest-neighbour position matching for this task? Keep the scope small enough to test fairly. Helpful background: Python, regression, and introductory wireless concepts.",
        "ai_technique": "Starting approach: nearest-neighbour position matching. Extension: compare a regressor or temporal neural model.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Consented Wi-Fi measurements with position labels and session information; inspect dataset access before choosing the scope. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Median and high-percentile position error on held-out sessions or locations. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Process complex CSI matrix: 3 antennas x 56 subcarriers x T time stamps\ncsi_amplitude = np.abs(csi_matrix)\ncsi_phase = np.unwrap(np.angle(csi_matrix), axis=-1)\npredicted_xy = st_gcn_model(torch.tensor(csi_features))"
      },
      {
        "id": "ece-10",
        "title": "Learning to Search Circuit Parameters with Simulation",
        "difficulty": "Advanced",
        "summary": "Use a small amplifier simulation to explore how parameter choices affect circuit behaviour. Start with a documented parameter sweep or random search. When that works, compare Bayesian optimisation before considering a neural search method. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a documented parameter sweep or random search for this task? Keep the scope small enough to test fairly. Helpful background: Circuit analysis, simulation, and optimisation.",
        "ai_technique": "Starting approach: a documented parameter sweep or random search. Extension: compare Bayesian optimisation before considering a neural search method.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: A documented circuit model with permitted component parameters and reproducible simulator settings. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Simulated specification violations, simulation budget, and repeatability. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom torch_geometric.nn import GATConv\n# Graph representation of SPICE circuit netlist (nodes: transistors, edges: nets)\ngraph = circuit_to_graph(netlist_file)\npredicted_performance = gnn_surrogate(graph.x, graph.edge_index)"
      },
      {
        "id": "ece-11",
        "title": "Modulation Type Classification from Signal Features",
        "difficulty": "Beginner",
        "summary": "Recognise a few simulated modulation classes using prepared signal features. Start with a small decision tree. When that works, compare a neural classifier using raw signal windows. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a small decision tree for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a small decision tree. Extension: compare a neural classifier using raw signal windows.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: labelled simulated signals across documented noise levels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and performance across held-out noise conditions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Modulation Type Classification from Signal Features\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a small decision tree.\n# 4. Optional extension: compare a neural classifier using raw signal windows.\n# 5. Evaluate macro-F1 and performance across held-out noise conditions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-12",
        "title": "Network Traffic Volume Forecasting",
        "difficulty": "Beginner",
        "summary": "Predict traffic volume for the next interval from recorded network counts. Start with a persistence forecast. When that works, compare a tree regressor using past traffic and time features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a persistence forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a persistence forecast. Extension: compare a tree regressor using past traffic and time features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: anonymised timestamped traffic counts. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and peak-traffic error on later intervals. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Network Traffic Volume Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a persistence forecast.\n# 4. Optional extension: compare a tree regressor using past traffic and time features.\n# 5. Evaluate MAE and peak-traffic error on later intervals.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-13",
        "title": "Signal-to-Noise Ratio Estimation",
        "difficulty": "Beginner",
        "summary": "Estimate the noise level of a simulated signal using simple features. Start with a conventional estimator or linear regression. When that works, compare a small neural regressor. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a conventional estimator or linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a conventional estimator or linear regression. Extension: compare a small neural regressor.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic signals with known signal and noise powers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Estimation error across signal types excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Signal-to-Noise Ratio Estimation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a conventional estimator or linear regression.\n# 4. Optional extension: compare a small neural regressor.\n# 5. Evaluate estimation error across signal types excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-14",
        "title": "Bluetooth Signal-Based Distance Estimation",
        "difficulty": "Beginner",
        "summary": "Explore how recorded signal strength relates to measured distance. Start with a documented path-loss fit. When that works, compare regression and quantify the effect of obstacles. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented path-loss fit for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a documented path-loss fit. Extension: compare regression and quantify the effect of obstacles.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: consented signal-strength measurements with distance and session labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Median distance error and variation across sessions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Bluetooth Signal-Based Distance Estimation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented path-loss fit.\n# 4. Optional extension: compare regression and quantify the effect of obstacles.\n# 5. Evaluate median distance error and variation across sessions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-15",
        "title": "Audio Event Classification",
        "difficulty": "Intermediate",
        "summary": "Classify a small set of everyday sounds in a permitted audio dataset. Start with audio features and a linear classifier. When that works, compare a spectrogram-based neural network. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on audio features and a linear classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: audio features and a linear classifier. Extension: compare a spectrogram-based neural network.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: labelled audio clips grouped by recording source. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on unseen recording conditions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Audio Event Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with audio features and a linear classifier.\n# 4. Optional extension: compare a spectrogram-based neural network.\n# 5. Evaluate macro-F1 and errors on unseen recording conditions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-16",
        "title": "Wireless Link Quality Prediction",
        "difficulty": "Intermediate",
        "summary": "Predict a defined link-quality measure from earlier signal observations. Start with a moving average and regression. When that works, compare a tree model across simulated or recorded conditions. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a moving average and regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a moving average and regression. Extension: compare a tree model across simulated or recorded conditions.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: timestamped signal measurements and subsequent link-quality targets. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Prediction error and failures during changing conditions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Wireless Link Quality Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a moving average and regression.\n# 4. Optional extension: compare a tree model across simulated or recorded conditions.\n# 5. Evaluate prediction error and failures during changing conditions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-17",
        "title": "Electronics Component Condition Classification",
        "difficulty": "Beginner",
        "summary": "Use a small measurement table to recognise documented component conditions. Start with a decision tree. When that works, compare an ensemble and investigate feature importance. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a decision tree for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a decision tree. Extension: compare an ensemble and investigate feature importance.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted measurements or circuit simulations with clear component labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on held-out batches or simulation settings. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Electronics Component Condition Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a decision tree.\n# 4. Optional extension: compare an ensemble and investigate feature importance.\n# 5. Evaluate macro-F1 on held-out batches or simulation settings.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-18",
        "title": "Neural Error Correction in a Communication Simulation",
        "difficulty": "Advanced",
        "summary": "Compare learned decoding with a conventional decoder on a small coding task. Start with a conventional decoder. When that works, compare a compact neural decoder under the same channel assumptions. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a conventional decoder for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a conventional decoder. Extension: compare a compact neural decoder under the same channel assumptions.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic encoded messages and corresponding noisy received signals. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Bit-error rate and runtime across held-out noise levels. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Neural Error Correction in a Communication Simulation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a conventional decoder.\n# 4. Optional extension: compare a compact neural decoder under the same channel assumptions.\n# 5. Evaluate bit-error rate and runtime across held-out noise levels.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-19",
        "title": "Adaptive Network Resource Allocation in Simulation",
        "difficulty": "Advanced",
        "summary": "Explore how a controller allocates limited resources among simulated users. Start with round-robin or proportional allocation. When that works, compare a reinforcement-learning policy. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on round-robin or proportional allocation for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: round-robin or proportional allocation. Extension: compare a reinforcement-learning policy.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic user demand and link conditions in a documented simulator. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Throughput, delay, fairness, and constraint violations. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Adaptive Network Resource Allocation in Simulation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with round-robin or proportional allocation.\n# 4. Optional extension: compare a reinforcement-learning policy.\n# 5. Evaluate throughput, delay, fairness, and constraint violations.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "ece-20",
        "title": "Tiny Audio Classifier: Accuracy and Resource Trade-offs",
        "difficulty": "Intermediate",
        "summary": "Compare a small audio model before and after a size-reduction step. Start with a compact trained classifier. When that works, evaluate quantisation or pruning on recorded test clips. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a compact trained classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a compact trained classifier. Extension: evaluate quantisation or pruning on recorded test clips.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted labelled audio and a reproducible evaluation split. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Model size, accuracy change, and measured latency on the stated device. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Tiny Audio Classifier: Accuracy and Resource Trade-offs\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a compact trained classifier.\n# 4. Optional extension: evaluate quantisation or pruning on recorded test clips.\n# 5. Evaluate model size, accuracy change, and measured latency on the stated device.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "civil": {
    "name": "Civil & Structural Engineering",
    "short": "Civil",
    "icon": "\ud83c\udfd7\ufe0f",
    "color": "#10b981",
    "tagline": "Connect materials and infrastructure questions with data.",
    "description": "Engineering judgement, inspection, and physical models remain essential. Student AI projects can help explore recorded measurements and image patterns within a defined scope. Start with concrete strength, road images, or visible safety equipment; progress to crack or flood analysis; explore graph learning and traffic control as advanced projects. Beginner means basic Python with guided examples; intermediate means independent data preparation and model evaluation; advanced means domain knowledge plus deeper modelling. Levels describe the starting scope, not a guaranteed completion time.",
    "projects": [
      {
        "id": "civ-01",
        "title": "Detecting Concrete Cracks in Images",
        "difficulty": "Intermediate",
        "summary": "Find cracks in labelled concrete images and study where the model makes mistakes. Start with a transfer-learning crack classifier. When that works, compare pixel-level segmentation if masks are available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a transfer-learning crack classifier for this task? Keep the scope small enough to test fairly. Helpful background: Python, image preparation, and introductory deep learning.",
        "ai_technique": "Starting approach: a transfer-learning crack classifier. Extension: compare pixel-level segmentation if masks are available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted concrete images with class labels or crack masks and structure identifiers where available. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision, recall, and segmentation overlap where appropriate, using unseen structures for testing. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport segmentation_models_pytorch as smp\nmodel = smp.DeepLabV3Plus(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=3)\n# Classes: [Background, Crack, Spalling]"
      },
      {
        "id": "civ-02",
        "title": "Structural Monitoring with Graph-Based Learning",
        "difficulty": "Advanced",
        "summary": "Explore how connected sensor measurements describe the condition of a simulated or recorded structure. Start with sensor features and a conventional classifier or regressor. When that works, compare a graph neural network that reflects the structural connections. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on sensor features and a conventional classifier or regressor for this task? Keep the scope small enough to test fairly. Helpful background: Structural dynamics, time-series processing, and neural networks.",
        "ai_technique": "Starting approach: sensor features and a conventional classifier or regressor. Extension: compare a graph neural network that reflects the structural connections.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Documented structural simulations or permitted sensor recordings with known condition labels. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Prediction error, missed anomalies, and sensitivity to sensor noise on held-out scenarios. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Structural sensor network represented as spatial graph based on physical bridge girder topology\nclass BridgeGNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = GCNConv(in_features=8, out_features=32)\n        self.gru = nn.GRU(32, 16, batch_first=True)\n        self.classifier = nn.Linear(16, 2) # [Healthy, Damaged]"
      },
      {
        "id": "civ-03",
        "title": "Flood-Level Forecasting from Historical Records",
        "difficulty": "Intermediate",
        "summary": "Forecast a defined water-level or flood-related target from earlier rainfall and gauge measurements. Start with a persistence forecast and regression model. When that works, compare an LSTM when enough ordered observations are available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a persistence forecast and regression model for this task? Keep the scope small enough to test fairly. Helpful background: Time-series analysis, regression, and basic hydrology.",
        "ai_technique": "Starting approach: a persistence forecast and regression model. Extension: compare an LSTM when enough ordered observations are available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Historical rainfall and gauge records with aligned timestamps, clear units, and documented gaps. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: MAE, errors during high-water events, and performance on held-out time periods. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nmodel = Sequential([\n    ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(None, 128, 128, 4)),\n    BatchNormalization(),\n    Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same') # Output: Inundation Depth (m)\n])"
      },
      {
        "id": "civ-04",
        "title": "Exploring AI-Assisted Structural Topology Design",
        "difficulty": "Advanced",
        "summary": "Compare ways to propose material layouts for a small, clearly constrained structural design problem. Start with a conventional topology-optimisation example. When that works, evaluate a learned surrogate or generative proposal against the simulator. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a conventional topology-optimisation example for this task? Keep the scope small enough to test fairly. Helpful background: Structural mechanics, finite-element analysis, and optimisation.",
        "ai_technique": "Starting approach: a conventional topology-optimisation example. Extension: evaluate a learned surrogate or generative proposal against the simulator.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Small documented finite-element simulations with consistent loads, boundary conditions, and material assumptions. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Simulation-checked compliance, material use, constraint violations, and runtime. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Condition diffusion on boundary conditions: load vectors and fixed support nodes\npredicted_density_field = diffusion_model.sample(batch_size=1, condition=stress_tensor)"
      },
      {
        "id": "civ-05",
        "title": "Road Surface Classification from Images",
        "difficulty": "Beginner",
        "summary": "Classify a small set of road images as visibly damaged or undamaged. Start with a small transfer-learning image classifier. When that works, extend to pothole detection when location labels are available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a small transfer-learning image classifier for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and guided image-classification practice.",
        "ai_technique": "Starting approach: a small transfer-learning image classifier. Extension: extend to pothole detection when location labels are available.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: Permitted road images with checked labels; separate nearby frames to avoid near-duplicates across splits. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors on roads or recording sessions held out from training. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Live road damage parsing and GIS mapping\nresults = model(frame)\nfor r in results:\n    if r.boxes:\n        pothole_count = sum(1 for c in r.boxes.cls if model.names[int(c)] == 'pothole')\n        log_road_condition(lat, lon, pothole_count)"
      },
      {
        "id": "civ-06",
        "title": "Classifying Rock Conditions from Labelled Images",
        "difficulty": "Intermediate",
        "summary": "Study how visual rock features relate to a documented set of condition labels. Start with image features and a conventional classifier. When that works, compare transfer learning and discuss missing geotechnical information. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on image features and a conventional classifier for this task? Keep the scope small enough to test fairly. Helpful background: Classification and introductory geotechnical concepts.",
        "ai_technique": "Starting approach: image features and a conventional classifier. Extension: compare transfer learning and discuss missing geotechnical information.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Permitted rock-face images with documented expert labels and site identifiers. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors across held-out sites; do not treat image labels as tunnel-stability certification. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom transformers import ViTForImageClassification, ViTImageProcessor\nprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')\nmodel = ViTForImageClassification.from_pretrained('custom-tbm-rock-classifier')"
      },
      {
        "id": "civ-07",
        "title": "Traffic Signal Control in a Simulated Intersection",
        "difficulty": "Advanced",
        "summary": "Compare signal timing strategies in one simulated junction before extending to a network. Start with fixed timing and a simple demand-responsive policy. When that works, train a reinforcement-learning controller; consider multiple junctions later. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on fixed timing and a simple demand-responsive policy for this task? Keep the scope small enough to test fairly. Helpful background: Python, simulation, and reinforcement-learning basics.",
        "ai_technique": "Starting approach: fixed timing and a simple demand-responsive policy. Extension: train a reinforcement-learning controller; consider multiple junctions later.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Synthetic traffic arrivals or permitted counts in a documented traffic simulation. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Waiting time, queue length, and fairness between approaches under held-out demand. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport traci\n# Connect to SUMO simulator and execute RL phase control\ntraci.start(['sumo', '-c', 'city_network.sumocfg'])\nstate = get_intersection_queue_lengths(traci)\naction = agent.act(state)\ntraci.trafficlight.setPhase('Junction_4', action)"
      },
      {
        "id": "civ-08",
        "title": "Analysing Land Movement from Satellite Time Series",
        "difficulty": "Advanced",
        "summary": "Explore a prepared displacement time series and identify areas with changing movement patterns. Start with trend estimation and a documented anomaly rule. When that works, compare a forecasting model and test uncertainty across locations. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on trend estimation and a documented anomaly rule for this task? Keep the scope small enough to test fairly. Helpful background: GIS, time-series analysis, and remote-sensing fundamentals.",
        "ai_technique": "Starting approach: trend estimation and a documented anomaly rule. Extension: compare a forecasting model and test uncertainty across locations.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Prepared geospatial displacement products with quality flags, timestamps, and reference-frame information. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Forecast error, false alerts, and agreement with independent reference measurements where available. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Process unwrapped interferogram phase deformation\ndef compute_deformation_mm(unwrapped_phase, wavelength=55.46):\n    return (unwrapped_phase * wavelength) / (-4 * np.pi)"
      },
      {
        "id": "civ-09",
        "title": "Recognising Visible Safety Equipment in Site Images",
        "difficulty": "Beginner",
        "summary": "Create an offline demo that detects a limited set of visible items such as helmets in labelled images. Start with a pretrained detector and a small labelled test set. When that works, fine-tune the detector and test difficult lighting or occlusion. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a pretrained detector and a small labelled test set for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and a guided object-detection notebook.",
        "ai_technique": "Starting approach: a pretrained detector and a small labelled test set. Extension: fine-tune the detector and test difficult lighting or occlusion.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn"
        ],
        "data_pipeline": "Suggested data: Permitted or public site images with equipment annotations; avoid identifying or scoring individual workers. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision, missed items, and false detections; an image cannot establish overall site compliance. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom shapely.geometry import Point, Polygon\nzone_polygon = Polygon([(100, 100), (400, 100), (450, 400), (80, 400)])\nfor person_box in detected_workers:\n    center = Point(person_box.cx, person_box.cy)\n    if zone_polygon.contains(center) and excavator_active:\n        trigger_loudspeaker_alarm('Danger: Personnel in Heavy Equipment Swing Radius!')"
      },
      {
        "id": "civ-10",
        "title": "Predicting Concrete Compressive Strength",
        "difficulty": "Beginner",
        "summary": "Use mix and curing records to estimate measured compressive strength. Start with linear regression. When that works, compare a tree-based model and examine prediction errors. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on linear regression for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python, tabular data, and an introduction to regression.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree-based model and examine prediction errors.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: A documented concrete-strength table with mix inputs, curing age, units, and measured targets. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: MAE, RMSE, and errors for held-out mixes or batches. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport xgboost as xgb\nmodel = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)\nmodel.fit(X_train, y_train)\n# Explaining influence of water-cement ratio via SHAP\nexplainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_test)"
      },
      {
        "id": "civ-11",
        "title": "Construction Cost Estimation from Historical Projects",
        "difficulty": "Beginner",
        "summary": "Estimate a clearly defined construction cost from a small project table. Start with linear regression. When that works, compare a tree model and examine outliers. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree model and examine outliers.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted project records with consistent cost dates, units, and scope. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors across project types excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Construction Cost Estimation from Historical Projects\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree model and examine outliers.\n# 5. Evaluate MAE and errors across project types excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-12",
        "title": "Construction Delay Risk Classification",
        "difficulty": "Beginner",
        "summary": "Explore factors associated with recorded project delays. Start with logistic regression. When that works, compare a tree model using only information available at the planning date. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model using only information available at the planning date.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: anonymised project records with a documented delay definition. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and calibration on held-out projects. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Construction Delay Risk Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression.\n# 4. Optional extension: compare a tree model using only information available at the planning date.\n# 5. Evaluate precision, recall, and calibration on held-out projects.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-13",
        "title": "Soil Property Prediction from Laboratory Records",
        "difficulty": "Beginner",
        "summary": "Estimate one measured soil property from other available test features. Start with linear regression. When that works, compare a tree regressor and inspect uncertainty. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree regressor and inspect uncertainty.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: documented soil test records with units and sample identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and performance on held-out sites or samples. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Soil Property Prediction from Laboratory Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree regressor and inspect uncertainty.\n# 5. Evaluate MAE and performance on held-out sites or samples.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-14",
        "title": "Building Energy Consumption Forecasting",
        "difficulty": "Beginner",
        "summary": "Forecast daily building energy use from weather and calendar features. Start with a seasonal average and regression. When that works, compare a tree-based forecast. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal average and regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a seasonal average and regression. Extension: compare a tree-based forecast.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: timestamped building energy readings and matching weather observations. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and peak-period error on later dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Building Energy Consumption Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal average and regression.\n# 4. Optional extension: compare a tree-based forecast.\n# 5. Evaluate MAE and peak-period error on later dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-15",
        "title": "Water Quality Indicator Prediction",
        "difficulty": "Intermediate",
        "summary": "Estimate a measured water-quality indicator from related monitoring variables. Start with linear regression. When that works, compare a tree model and examine station-level differences. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree model and examine station-level differences.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted monitoring records with laboratory reference values and units. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Prediction error on held-out stations or periods; results do not certify drinking-water safety. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Water Quality Indicator Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree model and examine station-level differences.\n# 5. Evaluate prediction error on held-out stations or periods; results do not certify drinking-water safety.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-16",
        "title": "Traffic Flow Forecasting from Road Sensor Records",
        "difficulty": "Intermediate",
        "summary": "Forecast vehicle counts at a small set of monitoring points. Start with a seasonal-naive forecast. When that works, compare a sequence model after evaluating regression. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal-naive forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a seasonal-naive forecast. Extension: compare a sequence model after evaluating regression.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: timestamped traffic counts with location identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE during normal and congested periods on future data. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Traffic Flow Forecasting from Road Sensor Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal-naive forecast.\n# 4. Optional extension: compare a sequence model after evaluating regression.\n# 5. Evaluate MAE during normal and congested periods on future data.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-17",
        "title": "Construction Waste Quantity Estimation",
        "difficulty": "Beginner",
        "summary": "Estimate recorded waste quantities from project characteristics. Start with linear regression. When that works, compare a tree model and analyse errors by project category. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree model and analyse errors by project category.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted records with waste categories, quantities, and consistent units. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and bias across held-out project types. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Construction Waste Quantity Estimation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree model and analyse errors by project category.\n# 5. Evaluate MAE and bias across held-out project types.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-18",
        "title": "Bridge Condition Trend Analysis",
        "difficulty": "Intermediate",
        "summary": "Explore how recorded inspection scores change over time. Start with a last-observation baseline. When that works, compare regression with inspection-history features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a last-observation baseline for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a last-observation baseline. Extension: compare regression with inspection-history features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted bridge inspection histories with documented scoring definitions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Forecast error and uncertainty on bridges held out from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Bridge Condition Trend Analysis\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a last-observation baseline.\n# 4. Optional extension: compare regression with inspection-history features.\n# 5. Evaluate forecast error and uncertainty on bridges held out from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-19",
        "title": "Urban Growth Mapping from Satellite Images",
        "difficulty": "Advanced",
        "summary": "Compare land-cover changes across a small study area using labelled imagery. Start with a simple land-cover classifier. When that works, compare segmentation and check change consistency across dates. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a simple land-cover classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a simple land-cover classifier. Extension: compare segmentation and check change consistency across dates.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted georeferenced images with comparable resolution and checked labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Class-level mapping accuracy and change errors in held-out areas. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Urban Growth Mapping from Satellite Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a simple land-cover classifier.\n# 4. Optional extension: compare segmentation and check change consistency across dates.\n# 5. Evaluate class-level mapping accuracy and change errors in held-out areas.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "civ-20",
        "title": "Exploring Building Design Trade-offs with Surrogate Models",
        "difficulty": "Advanced",
        "summary": "Use simulated design examples to compare energy and material trade-offs. Start with a small parameter sweep. When that works, fit a surrogate and validate selected design proposals with the simulator. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a small parameter sweep for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a small parameter sweep. Extension: fit a surrogate and validate selected design proposals with the simulator.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: documented building simulations with bounded design variables. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Surrogate error, verified objective values, and constraint violations. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Exploring Building Design Trade-offs with Surrogate Models\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a small parameter sweep.\n# 4. Optional extension: fit a surrogate and validate selected design proposals with the simulator.\n# 5. Evaluate surrogate error, verified objective values, and constraint violations.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "agriculture": {
    "name": "Agriculture & Smart Farming",
    "short": "Agriculture",
    "icon": "\ud83c\udf31",
    "color": "#84cc16",
    "tagline": "Use familiar farming questions to begin learning AI.",
    "description": "Crop knowledge and field observations give meaning to agricultural data. Models can help you explore patterns, but results from one dataset may not transfer to another farm or season. Start with soil moisture, leaf images, or grain appearance; progress to yield and hive audio; try simulated robotics or greenhouse control when ready. Beginner means basic Python with guided examples; intermediate means independent data preparation and model evaluation; advanced means domain knowledge plus deeper modelling. Levels describe the starting scope, not a guaranteed completion time.",
    "projects": [
      {
        "id": "agr-01",
        "title": "Exploring Crop Stress in Multispectral Images",
        "difficulty": "Intermediate",
        "summary": "Study whether labelled image features can distinguish selected crop-stress conditions. Start with spectral features and a simple classifier. When that works, compare a small neural model if the dataset supports it. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on spectral features and a simple classifier for this task? Keep the scope small enough to test fairly. Helpful background: Image features, classification, and basic crop science.",
        "ai_technique": "Starting approach: spectral features and a simple classifier. Extension: compare a small neural model if the dataset supports it.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted multispectral or hyperspectral images with ground-reference labels and band information. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1, class-level recall, and performance on held-out fields or dates. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Compute Normalized Difference Red Edge index for early chlorophyll stress\nndre = (nir_band - red_edge_band) / (nir_band + red_edge_band + 1e-6)\n# Feed 3D spectral datacube into spatial-spectral CNN\nstress_map = model(torch.tensor(spectral_cube))"
      },
      {
        "id": "agr-02",
        "title": "Crop and Weed Recognition for a Simulated Farm Robot",
        "difficulty": "Advanced",
        "summary": "Build the vision component of a simulated weeding task using labelled crop and weed images. Start with a labelled image classifier or detector. When that works, add segmentation and evaluate target localisation in simulation. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a labelled image classifier or detector for this task? Keep the scope small enough to test fairly. Helpful background: Computer vision, geometry, and robotics simulation.",
        "ai_technique": "Starting approach: a labelled image classifier or detector. Extension: add segmentation and evaluate target localisation in simulation.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted field images with crop and weed labels; robot testing remains simulated for this capstone. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Detection precision, missed weeds, crop misidentification, and simulated localisation error. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Real-time ROS 2 weed tracker node\ndef image_callback(self, msg):\n    frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')\n    results = self.model(frame, conf=0.7)\n    for weed in results.weeds:\n        self.publish_laser_target(weed.centroid_x, weed.centroid_y)"
      },
      {
        "id": "agr-03",
        "title": "Forecasting Soil Moisture from Sensor Records",
        "difficulty": "Beginner",
        "summary": "Predict the next recorded soil-moisture value and explore a simple simulated irrigation rule. Start with a persistence forecast and linear regression. When that works, compare a tree model and test the rule on held-out days. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a persistence forecast and linear regression for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python, plotting, and introductory regression.",
        "ai_technique": "Starting approach: a persistence forecast and linear regression. Extension: compare a tree model and test the rule on held-out days.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: Recorded moisture and weather data, or clearly labelled synthetic observations with consistent time intervals. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: MAE and the behaviour of the simulated rule under clearly stated assumptions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# ESP32 Python/MicroPython control loop\npredicted_moisture_24h = lstm_predict(sensor_history)\nif predicted_moisture_24h < wilting_point_threshold:\n    trigger_solenoid_valve(duration_minutes=calculate_water_volume(field_capacity))"
      },
      {
        "id": "agr-04",
        "title": "Crop Yield Prediction from Weather and Satellite Features",
        "difficulty": "Intermediate",
        "summary": "Estimate a defined crop-yield target from prepared seasonal and location features. Start with a historical-average baseline and regression. When that works, compare feature groups or a neural model if data volume supports it. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a historical-average baseline and regression for this task? Keep the scope small enough to test fairly. Helpful background: Regression, feature preparation, and basic agricultural statistics.",
        "ai_technique": "Starting approach: a historical-average baseline and regression. Extension: compare feature groups or a neural model if data volume supports it.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Permitted yield labels matched to weather and prepared satellite features at a consistent geographic scale. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: MAE and performance across held-out years or regions. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport ee\nee.Initialize()\nsentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2026-05-01', '2026-09-01').map(mask_clouds)\n# Feature extraction and multi-modal tensor assembly"
      },
      {
        "id": "agr-05",
        "title": "Exploring Beehive Audio Classification",
        "difficulty": "Intermediate",
        "summary": "Use labelled recordings to study whether sound patterns distinguish selected hive conditions. Start with audio features and a simple classifier. When that works, compare a small spectrogram-based network. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on audio features and a simple classifier for this task? Keep the scope small enough to test fairly. Helpful background: Audio processing and classification.",
        "ai_technique": "Starting approach: audio features and a simple classifier. Extension: compare a small spectrogram-based network.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Permitted hive recordings with hive identifiers, recording conditions, and documented labels. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1 and performance on hives excluded from training; discuss the reliability of condition labels. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Audio feature transformation\nmfcc = librosa.feature.mfcc(y=audio_signal, sr=16000, n_mfcc=40)\nstatus = tflite_model.predict(mfcc) # [Healthy Queen, Queenless, Pre-Swarm Buzz]"
      },
      {
        "id": "agr-06",
        "title": "Studying Cattle Movement from Recorded Video",
        "difficulty": "Advanced",
        "summary": "Explore a limited movement-analysis task using labelled, permitted cattle videos. Start with a documented motion-feature baseline. When that works, compare pose or sequence models; keep individual identification as a separate optional task. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a documented motion-feature baseline for this task? Keep the scope small enough to test fairly. Helpful background: Computer vision, sequence modelling, and animal-science supervision.",
        "ai_technique": "Starting approach: a documented motion-feature baseline. Extension: compare pose or sequence models; keep individual identification as a separate optional task.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Consented or permitted videos with animal identifiers and expert movement annotations. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Agreement with movement labels and performance on held-out animals; avoid veterinary diagnostic claims. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Extract cow posture keypoints: spine curvature, head bobbing, step asymmetry\nkeypoints = pose_model(cow_gait_frame)\nspine_curvature_angle = calculate_angle(keypoints['shoulder'], keypoints['mid_back'], keypoints['pelvis'])\nif spine_curvature_angle > 15.0:\n    flag_early_lameness_warning(cow_id)"
      },
      {
        "id": "agr-07",
        "title": "Plant Leaf Image Classification",
        "difficulty": "Beginner",
        "summary": "Build a small classroom demo that classifies leaf images into a few labelled categories. Start with transfer learning with a small labelled dataset. When that works, test field photographs and examine uncertain predictions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on transfer learning with a small labelled dataset for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and guided image-classification practice.",
        "ai_technique": "Starting approach: transfer learning with a small labelled dataset. Extension: test field photographs and examine uncertain predictions.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn"
        ],
        "data_pipeline": "Suggested data: Permitted leaf images with documented labels; group images by plant or source where identifiers are available. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1 and differences between curated and field images; predictions are not confirmed diagnoses. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n// Flutter / TFLite Flutter implementation\nvar recognitions = await Tflite.runModelOnImage(\n  path: imageFile.path,\n  numResults: 3,\n  threshold: 0.5,\n  imageMean: 127.5,\n  imageStd: 127.5\n);"
      },
      {
        "id": "agr-08",
        "title": "Fruit Ripeness and Picking-Point Estimation in Simulation",
        "difficulty": "Advanced",
        "summary": "Study how a vision model identifies fruit and proposes a picking location in a simulated scene. Start with a fruit detector and simple geometric selection rule. When that works, compare segmentation or depth-based localisation in simulation. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a fruit detector and simple geometric selection rule for this task? Keep the scope small enough to test fairly. Helpful background: Computer vision, geometry, and robotics simulation.",
        "ai_technique": "Starting approach: a fruit detector and simple geometric selection rule. Extension: compare segmentation or depth-based localisation in simulation.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Labelled fruit images and, if needed, permitted depth data or synthetic scenes with known locations. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Detection errors, picking-point error, and failures under occlusion. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Estimate 3D grasp center and surface normal from depth point cloud\npoint_cloud = depth_to_pointcloud(rgbd_frame)\nfruits = segment_ripe_fruits(rgb_frame)\ngrasp_poses = pointnet_grasp_evaluator(point_cloud, fruits)"
      },
      {
        "id": "agr-09",
        "title": "Greenhouse Climate Control in Simulation",
        "difficulty": "Advanced",
        "summary": "Compare control strategies for a simplified greenhouse temperature model. Start with a thermostat-style rule or conventional controller. When that works, compare reinforcement learning after validating the simulation. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a thermostat-style rule or conventional controller for this task? Keep the scope small enough to test fairly. Helpful background: Control concepts, Python simulation, and reinforcement learning.",
        "ai_technique": "Starting approach: a thermostat-style rule or conventional controller. Extension: compare reinforcement learning after validating the simulation.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Recorded climate observations or a documented greenhouse simulation with clearly stated assumptions. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Time outside target bounds, simulated energy use, and constraint violations. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nclass GreenhouseEnv(gym.Env):\n    def step(self, action):\n        # Action: [ventilation_pct, led_intensity, misting_active, co2_ppm]\n        next_state, reward, done, info = self.simulate_physics(action)\n        return next_state, reward, done, info"
      },
      {
        "id": "agr-10",
        "title": "Grain Appearance Classification from Images",
        "difficulty": "Beginner",
        "summary": "Classify visible grain qualities such as intact and visibly damaged samples. Start with image features and a simple classifier. When that works, compare transfer learning and test new batches. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on image features and a simple classifier for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and introductory classification.",
        "ai_technique": "Starting approach: image features and a simple classifier. Extension: compare transfer learning and test new batches.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn"
        ],
        "data_pipeline": "Suggested data: Permitted grain images with checked appearance labels and batch identifiers where available. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on held-out batches; visual appearance does not establish mycotoxin safety. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Kernel classification on industrial conveyor belt\nspectrum = swir_sensor.read_spectrum(kernel_trigger)\nprob_aflatoxin = mycotoxin_classifier.predict_proba([spectrum])[0][1]\nif prob_aflatoxin > 0.05:\n    fire_pneumatic_nozzle(delay_ms=12)"
      },
      {
        "id": "agr-11",
        "title": "Crop Recommendation from Soil and Weather Records",
        "difficulty": "Beginner",
        "summary": "Explore which crop labels are associated with a prepared soil-and-weather table. Start with a decision tree. When that works, compare an ensemble and discuss limits of transferring labels to real farms. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a decision tree for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a decision tree. Extension: compare an ensemble and discuss limits of transferring labels to real farms.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a documented dataset with crop labels and feature definitions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on held-out locations; recommendations need local agronomic review. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Crop Recommendation from Soil and Weather Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a decision tree.\n# 4. Optional extension: compare an ensemble and discuss limits of transferring labels to real farms.\n# 5. Evaluate macro-F1 on held-out locations; recommendations need local agronomic review.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-12",
        "title": "Agricultural Market Price Forecasting",
        "difficulty": "Beginner",
        "summary": "Forecast the price of one crop in one market over a defined horizon. Start with a seasonal-naive forecast. When that works, compare regression with past prices and calendar features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal-naive forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a seasonal-naive forecast. Extension: compare regression with past prices and calendar features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: timestamped market prices with consistent grades, units, and currency. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors during price changes on future dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Agricultural Market Price Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal-naive forecast.\n# 4. Optional extension: compare regression with past prices and calendar features.\n# 5. Evaluate MAE and errors during price changes on future dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-13",
        "title": "Seed Image Classification",
        "difficulty": "Beginner",
        "summary": "Classify a small number of seed categories from labelled photographs. Start with shape features and a simple classifier. When that works, compare transfer learning. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on shape features and a simple classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: shape features and a simple classifier. Extension: compare transfer learning.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted seed images with class labels and batch identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and performance on unseen batches. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Seed Image Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with shape features and a simple classifier.\n# 4. Optional extension: compare transfer learning.\n# 5. Evaluate macro-F1 and performance on unseen batches.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-14",
        "title": "Farm Water-Use Forecasting",
        "difficulty": "Beginner",
        "summary": "Estimate the next day's water use from recorded farm conditions. Start with a moving average and regression. When that works, compare a tree model with weather and schedule features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a moving average and regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a moving average and regression. Extension: compare a tree model with weather and schedule features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted meter records and time-aligned farm observations. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors on high-use days. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Farm Water-Use Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a moving average and regression.\n# 4. Optional extension: compare a tree model with weather and schedule features.\n# 5. Evaluate MAE and errors on high-use days.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-15",
        "title": "Fruit Quality Classification from Images",
        "difficulty": "Intermediate",
        "summary": "Recognise documented visible fruit-quality categories. Start with a transfer-learning classifier. When that works, compare image segmentation where masks are available. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a transfer-learning classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a transfer-learning classifier. Extension: compare image segmentation where masks are available.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted fruit images grouped by batch and collection session. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors across lighting conditions; appearance alone does not establish food safety. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Fruit Quality Classification from Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a transfer-learning classifier.\n# 4. Optional extension: compare image segmentation where masks are available.\n# 5. Evaluate macro-F1 and errors across lighting conditions; appearance alone does not establish food safety.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-16",
        "title": "Livestock Activity Classification from Sensor Data",
        "difficulty": "Intermediate",
        "summary": "Classify recorded activity labels such as resting and walking. Start with window features and a simple classifier. When that works, compare a sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on window features and a simple classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: window features and a simple classifier. Extension: compare a sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted animal sensor recordings with time-aligned activity labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on animals excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Livestock Activity Classification from Sensor Data\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with window features and a simple classifier.\n# 4. Optional extension: compare a sequence model.\n# 5. Evaluate macro-F1 on animals excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-17",
        "title": "Crop Canopy Segmentation from Aerial Images",
        "difficulty": "Intermediate",
        "summary": "Separate crop canopy from background in a small annotated image set. Start with a colour or vegetation-index threshold. When that works, compare a compact segmentation network. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a colour or vegetation-index threshold for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a colour or vegetation-index threshold. Extension: compare a compact segmentation network.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted aerial images with checked canopy masks. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Segmentation overlap and errors across held-out fields. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Crop Canopy Segmentation from Aerial Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a colour or vegetation-index threshold.\n# 4. Optional extension: compare a compact segmentation network.\n# 5. Evaluate segmentation overlap and errors across held-out fields.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-18",
        "title": "Farm Equipment Maintenance Risk Prediction",
        "difficulty": "Intermediate",
        "summary": "Explore whether recorded operating features help flag labelled maintenance events. Start with logistic regression. When that works, compare a tree model and analyse false alarms. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model and analyse false alarms.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted equipment histories with event times and measurements collected beforehand. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision-recall and lead-time-dependent errors on held-out equipment. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Farm Equipment Maintenance Risk Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression.\n# 4. Optional extension: compare a tree model and analyse false alarms.\n# 5. Evaluate precision-recall and lead-time-dependent errors on held-out equipment.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-19",
        "title": "Irrigation Scheduling with Forecast Uncertainty",
        "difficulty": "Advanced",
        "summary": "Compare irrigation schedules in a documented soil-water simulation. Start with a threshold-based schedule. When that works, compare optimisation using uncertain moisture forecasts. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a threshold-based schedule for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a threshold-based schedule. Extension: compare optimisation using uncertain moisture forecasts.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a soil-water simulation or approved historical data with stated assumptions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Simulated water use, time outside target bounds, and sensitivity to forecast error. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Irrigation Scheduling with Forecast Uncertainty\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a threshold-based schedule.\n# 4. Optional extension: compare optimisation using uncertain moisture forecasts.\n# 5. Evaluate simulated water use, time outside target bounds, and sensitivity to forecast error.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "agr-20",
        "title": "Crop Monitoring Model Transfer Across Farms",
        "difficulty": "Advanced",
        "summary": "Test how an image model trained on one farm behaves on another. Start with a fixed image-classification model. When that works, compare limited fine-tuning while keeping an independent target-farm test set. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a fixed image-classification model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a fixed image-classification model. Extension: compare limited fine-tuning while keeping an independent target-farm test set.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted labelled crop images with farm and season identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Performance change, calibration, and errors across farms and seasons. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Crop Monitoring Model Transfer Across Farms\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a fixed image-classification model.\n# 4. Optional extension: compare limited fine-tuning while keeping an independent target-farm test set.\n# 5. Evaluate performance change, calibration, and errors across farms and seasons.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "business": {
    "name": "Business, Finance & Management",
    "short": "Business",
    "icon": "\ud83d\udcc8",
    "color": "#8b5cf6",
    "tagline": "Turn a business question into a measurable experiment.",
    "description": "Business knowledge helps define useful targets and costs. Spreadsheets, statistics, and simple rules remain useful baselines when you evaluate a more complex model. Start with churn prediction or invoice fields; progress to demand, reviews, and recommendations; attempt document assistants or simulated pricing after learning evaluation methods. Beginner means basic Python with guided examples; intermediate means independent data preparation and model evaluation; advanced means domain knowledge plus deeper modelling. Levels describe the starting scope, not a guaranteed completion time.",
    "projects": [
      {
        "id": "biz-01",
        "title": "Fraud Detection in Historical Transaction Data",
        "difficulty": "Intermediate",
        "summary": "Explore how a model flags potentially fraudulent transactions in a labelled research dataset. Start with logistic regression and a simple decision rule. When that works, compare a tree model; consider graph learning only if relationship data is available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on logistic regression and a simple decision rule for this task? Keep the scope small enough to test fairly. Helpful background: Classification, imbalanced datasets, and basic statistics.",
        "ai_technique": "Starting approach: logistic regression and a simple decision rule. Extension: compare a tree model; consider graph learning only if relationship data is available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: An anonymised, permitted transaction dataset with documented labels and a time-aware split where timestamps exist. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision-recall AUC, recall at a chosen review rate, and false positives on later data. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport torch_geometric.transforms as T\nfrom torch_geometric.nn import HGTConv\nclass FraudHGT(nn.Module):\n    def __init__(self, metadata):\n        super().__init__()\n        self.conv1 = HGTConv(in_channels=64, out_channels=32, metadata=metadata, heads=4)\n        self.classifier = nn.Linear(32, 2)"
      },
      {
        "id": "biz-02",
        "title": "Financial Report Question Answering with Source References",
        "difficulty": "Advanced",
        "summary": "Create a research demo that answers a small set of questions from supplied public company reports. Start with document search and extractive answers. When that works, compare retrieval-augmented generation with answer checks and source references. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on document search and extractive answers for this task? Keep the scope small enough to test fairly. Helpful background: Python, text retrieval, and language-model evaluation.",
        "ai_technique": "Starting approach: document search and extractive answers. Extension: compare retrieval-augmented generation with answer checks and source references.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted public reports and a manually checked question-and-answer set; exclude private account information. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Citation support, factual correctness, unanswered questions, and latency on a fixed test set. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader\nfrom llama_index.core.tools import QueryEngineTool\n# Build hierarchical query engine comparing year-over-year balance sheet line items\nengine = index.as_query_engine(similarity_top_k=5, response_mode='tree_summarize')"
      },
      {
        "id": "biz-03",
        "title": "Comparing Trading Policies in a Market Simulation",
        "difficulty": "Advanced",
        "summary": "Study how simple and learned trading policies behave in a constrained classroom simulation. Start with a documented rule-based policy. When that works, compare reinforcement learning with realistic transaction-cost assumptions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a documented rule-based policy for this task? Keep the scope small enough to test fairly. Helpful background: Time-series evaluation, probability, and reinforcement learning.",
        "ai_technique": "Starting approach: a documented rule-based policy. Extension: compare reinforcement learning with realistic transaction-cost assumptions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Historical research data or synthetic market sequences with timestamps, fees, and execution assumptions recorded. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Risk, drawdown, simulated return, and sensitivity to costs across held-out periods. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Process LOB Level 2 depth: 10 bids, 10 asks (price and volume)\nlob_state = extract_order_book_features(market_feed)\naction = rl_agent.get_action(lob_state) # [bid_offset, ask_offset, cancel_threshold]"
      },
      {
        "id": "biz-04",
        "title": "Forecasting Product Demand from Sales History",
        "difficulty": "Intermediate",
        "summary": "Predict demand for a small set of products over a defined future period. Start with a seasonal-naive forecast. When that works, compare a tree-based model or sequence model before trying a transformer. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a seasonal-naive forecast for this task? Keep the scope small enough to test fairly. Helpful background: Pandas, regression, and time-series splits.",
        "ai_technique": "Starting approach: a seasonal-naive forecast. Extension: compare a tree-based model or sequence model before trying a transformer.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted sales histories with timestamps and product identifiers; use promotion information only if known at forecast time. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: MAE or an appropriate scale-adjusted error across future dates and different products. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom pytorch_forecasting.models import TemporalFusionTransformer\ntft = TemporalFusionTransformer.from_dataset(\n    training_dataset, learning_rate=0.03, hidden_size=64, attention_head_size=4, dropout=0.2, loss=QuantileLoss()\n)"
      },
      {
        "id": "biz-05",
        "title": "Understanding Customer Churn with Machine Learning",
        "difficulty": "Beginner",
        "summary": "Predict a clearly defined churn label from an anonymised customer table and explain the main errors. Start with logistic regression. When that works, compare a tree model and discuss possible follow-up questions rather than automatically assigning incentives. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on logistic regression for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python, tabular data, and introductory classification.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model and discuss possible follow-up questions rather than automatically assigning incentives.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn",
          "Matplotlib"
        ],
        "data_pipeline": "Suggested data: An anonymised customer dataset with a documented churn definition and features recorded before the outcome. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision, recall, and calibration on held-out customers or later periods. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom causalml.inference.meta import BaseXLearner\n# Estimate Individual Treatment Effect (ITE) of a 20% discount offer\nxlearner = BaseXLearner(learner=LGBMRegressor())\nite = xlearner.fit_predict(X=features, Treatment=coupon_sent, y=retained_next_month)"
      },
      {
        "id": "biz-06",
        "title": "Finding Sentiment in Product Reviews",
        "difficulty": "Intermediate",
        "summary": "Classify review sentiment and investigate comments about a small set of product aspects. Start with TF-IDF features and a linear classifier. When that works, compare a pretrained language model; add languages only when labelled evaluation data is available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on TF-IDF features and a linear classifier for this task? Keep the scope small enough to test fairly. Helpful background: Python, text preprocessing, and classification.",
        "ai_technique": "Starting approach: TF-IDF features and a linear classifier. Extension: compare a pretrained language model; add languages only when labelled evaluation data is available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted review text with sentiment or aspect labels; group duplicates and near-duplicates before splitting. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors across products, languages, or review sources. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom transformers import pipeline\nabsa_classifier = pipeline('sentiment-analysis', model='yangheng/deberta-v3-base-absa-v1.1')\n# 'The battery life is stellar but the screen scratches easily'\n# -> {'Aspect': 'battery life', 'Sentiment': 'Positive'}, {'Aspect': 'screen', 'Sentiment': 'Negative'}"
      },
      {
        "id": "biz-07",
        "title": "Exploring Pricing Decisions in a Simulated Market",
        "difficulty": "Advanced",
        "summary": "Compare pricing strategies under a clearly documented classroom demand model. Start with a fixed-price rule. When that works, compare a contextual bandit while testing sensitivity to the simulator assumptions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a fixed-price rule for this task? Keep the scope small enough to test fairly. Helpful background: Probability, optimisation, and bandit methods.",
        "ai_technique": "Starting approach: a fixed-price rule. Extension: compare a contextual bandit while testing sensitivity to the simulator assumptions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Synthetic demand and price-response data with assumptions stated explicitly; no live customer experimentation is required. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Simulated revenue, exploration cost, and performance across changing demand settings. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Linear Upper Confidence Bound (LinUCB) for price tier recommendation\narm_scores = [np.dot(theta[arm], context_vector) + alpha * np.sqrt(np.dot(context_vector, A_inv[arm]).dot(context_vector)) for arm in price_tiers]\nchosen_price_tier = np.argmax(arm_scores)"
      },
      {
        "id": "biz-08",
        "title": "Extracting Fields from Sample Invoices",
        "difficulty": "Beginner",
        "summary": "Build a small tool that extracts a few fields from synthetic or redacted invoice images. Start with OCR and simple field-matching rules. When that works, compare a document model after preparing a labelled test set. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on OCR and simple field-matching rules for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python, text processing, and an OCR introduction.",
        "ai_technique": "Starting approach: OCR and simple field-matching rules. Extension: compare a document model after preparing a labelled test set.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn"
        ],
        "data_pipeline": "Suggested data: Synthetic or properly redacted invoices with checked text and field labels. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Field-level accuracy, missing fields, and errors on unseen invoice layouts. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor\nprocessor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)\nencoding = processor(image, words, boxes=boxes, word_labels=labels, return_tensors='pt')"
      },
      {
        "id": "biz-09",
        "title": "Comparing Sustainability Claims with Documented Evidence",
        "difficulty": "Intermediate",
        "summary": "Create a tool that retrieves report passages related to a small set of sustainability claims. Start with keyword search and a manually labelled relevance set. When that works, compare text embeddings and highlight claims needing human review. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on keyword search and a manually labelled relevance set for this task? Keep the scope small enough to test fairly. Helpful background: Text processing, information retrieval, and careful interpretation.",
        "ai_technique": "Starting approach: keyword search and a manually labelled relevance set. Extension: compare text embeddings and highlight claims needing human review.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted public reports with source locations and a transparent, manually checked relevance rubric. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Retrieval precision and agreement with a documented review rubric; do not label organisations deceptive from a model score. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nnli_model = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')\nclaim = 'We reduced scope 1 greenhouse gas emissions by 40% in our manufacturing sites.'\nlabels = ['verifiable empirical claim', 'vague marketing fluff', 'contradicted by regulatory data']"
      },
      {
        "id": "biz-10",
        "title": "Building a Small Product Recommendation System",
        "difficulty": "Intermediate",
        "summary": "Recommend items from a permitted interaction dataset and compare against a simple popularity list. Start with popularity ranking and collaborative filtering. When that works, compare embeddings or a two-tower model if sufficient interaction data is available. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on popularity ranking and collaborative filtering for this task? Keep the scope small enough to test fairly. Helpful background: Python, matrix operations, and introductory recommendation methods.",
        "ai_technique": "Starting approach: popularity ranking and collaborative filtering. Extension: compare embeddings or a two-tower model if sufficient interaction data is available.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: An anonymised interaction dataset with user/item identifiers and a time-aware evaluation split. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision or recall at a chosen list length, coverage, and cold-start performance. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport tensorflow_recommenders as tfrs\nclass TwoTowerModel(tfrs.Model):\n    def __init__(self, user_model, item_model):\n        super().__init__()\n        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=items.batch(128)))\n        self.user_model = user_model\n        self.item_model = item_model"
      },
      {
        "id": "biz-11",
        "title": "Customer Segmentation from Purchase Summaries",
        "difficulty": "Beginner",
        "summary": "Group anonymised customer purchase patterns and explain each group's characteristics. Start with simple summary statistics and k-means. When that works, compare cluster stability under different features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on simple summary statistics and k-means for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: simple summary statistics and k-means. Extension: compare cluster stability under different features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: anonymised customer-level purchase summaries. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Cluster stability and interpretability; groups are exploratory rather than fixed customer identities. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Customer Segmentation from Purchase Summaries\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with simple summary statistics and k-means.\n# 4. Optional extension: compare cluster stability under different features.\n# 5. Evaluate cluster stability and interpretability; groups are exploratory rather than fixed customer identities.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-12",
        "title": "Sales Revenue Forecasting for a Small Business",
        "difficulty": "Beginner",
        "summary": "Forecast revenue for a defined period from a small sales history. Start with a seasonal-naive forecast. When that works, compare a regression model with known calendar features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal-naive forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a seasonal-naive forecast. Extension: compare a regression model with known calendar features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or synthetic timestamped sales totals. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and error during seasonal peaks on future periods. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Sales Revenue Forecasting for a Small Business\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal-naive forecast.\n# 4. Optional extension: compare a regression model with known calendar features.\n# 5. Evaluate MAE and error during seasonal peaks on future periods.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-13",
        "title": "Support Ticket Category Classification",
        "difficulty": "Beginner",
        "summary": "Route sample support messages into a small set of labelled categories. Start with TF-IDF and logistic regression. When that works, compare embeddings and add an uncertain-case review option. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on TF-IDF and logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: TF-IDF and logistic regression. Extension: compare embeddings and add an uncertain-case review option.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: redacted or synthetic support tickets with checked labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1, misrouting rate, and performance on later tickets. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Support Ticket Category Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with TF-IDF and logistic regression.\n# 4. Optional extension: compare embeddings and add an uncertain-case review option.\n# 5. Evaluate macro-F1, misrouting rate, and performance on later tickets.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-14",
        "title": "Expense Category Classification",
        "difficulty": "Beginner",
        "summary": "Assign recorded expense descriptions to a defined set of accounting categories. Start with keyword rules. When that works, compare a text classifier and flag low-confidence cases. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: keyword rules. Extension: compare a text classifier and flag low-confidence cases.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or anonymised expense descriptions with category labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on new suppliers or descriptions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Expense Category Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword rules.\n# 4. Optional extension: compare a text classifier and flag low-confidence cases.\n# 5. Evaluate macro-F1 and errors on new suppliers or descriptions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-15",
        "title": "Inventory Stockout Risk Prediction",
        "difficulty": "Intermediate",
        "summary": "Estimate whether a product may run out within a defined planning window. Start with a reorder-threshold rule. When that works, compare a classifier using only available demand and stock information. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a reorder-threshold rule for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a reorder-threshold rule. Extension: compare a classifier using only available demand and stock information.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted inventory histories with dated stock levels and observed outcomes. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Recall, false alarms, and errors across held-out products. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Inventory Stockout Risk Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a reorder-threshold rule.\n# 4. Optional extension: compare a classifier using only available demand and stock information.\n# 5. Evaluate recall, false alarms, and errors across held-out products.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-16",
        "title": "Loan Default Modelling with Fairness Checks",
        "difficulty": "Intermediate",
        "summary": "Study a documented historical default label in a permitted research dataset. Start with logistic regression. When that works, compare a tree model and examine calibration and subgroup errors. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model and examine calibration and subgroup errors.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: de-identified research data with a defined outcome and lawful permitted use. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Calibration, precision-recall, and subgroup error; the prototype must not make lending decisions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Loan Default Modelling with Fairness Checks\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression.\n# 4. Optional extension: compare a tree model and examine calibration and subgroup errors.\n# 5. Evaluate calibration, precision-recall, and subgroup error; the prototype must not make lending decisions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-17",
        "title": "Cash-Flow Forecasting from Business Records",
        "difficulty": "Intermediate",
        "summary": "Forecast a small organisation's cash inflow and outflow separately. Start with historical averages. When that works, compare regression and combine forecasts into a clearly labelled projection. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on historical averages for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: historical averages. Extension: compare regression and combine forecasts into a clearly labelled projection.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or permitted cash-flow records with timestamps. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and sensitivity to unusually large transactions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Cash-Flow Forecasting from Business Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with historical averages.\n# 4. Optional extension: compare regression and combine forecasts into a clearly labelled projection.\n# 5. Evaluate MAE and sensitivity to unusually large transactions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-18",
        "title": "Invoice Anomaly Detection for Human Review",
        "difficulty": "Intermediate",
        "summary": "Highlight unusual invoice records for a reviewer to inspect. Start with documented amount and duplication rules. When that works, compare an isolation forest. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on documented amount and duplication rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: documented amount and duplication rules. Extension: compare an isolation forest.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or redacted invoices with known or transparently injected anomalies. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Review burden, detection rate, and false alarms; unusual invoices are not proof of wrongdoing. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Invoice Anomaly Detection for Human Review\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with documented amount and duplication rules.\n# 4. Optional extension: compare an isolation forest.\n# 5. Evaluate review burden, detection rate, and false alarms; unusual invoices are not proof of wrongdoing.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-19",
        "title": "Supply Chain Disruption Simulation and Planning",
        "difficulty": "Advanced",
        "summary": "Compare inventory strategies under simulated delivery disruptions. Start with a fixed replenishment policy. When that works, compare forecast-informed optimisation. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a fixed replenishment policy for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a fixed replenishment policy. Extension: compare forecast-informed optimisation.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic demand and lead-time scenarios with recorded assumptions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Simulated service level, holding cost, and sensitivity to disruption settings. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Supply Chain Disruption Simulation and Planning\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a fixed replenishment policy.\n# 4. Optional extension: compare forecast-informed optimisation.\n# 5. Evaluate simulated service level, holding cost, and sensitivity to disruption settings.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "biz-20",
        "title": "Evaluating a Business Document Assistant",
        "difficulty": "Advanced",
        "summary": "Test a retrieval assistant on a fixed set of questions about permitted internal-style documents. Start with keyword retrieval and quoted answers. When that works, compare retrieval-augmented generation with source checks. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword retrieval and quoted answers for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: keyword retrieval and quoted answers. Extension: compare retrieval-augmented generation with source checks.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or approved business documents and manually checked questions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Answer support, retrieval recall, abstention quality, and response time. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Evaluating a Business Document Assistant\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword retrieval and quoted answers.\n# 4. Optional extension: compare retrieval-augmented generation with source checks.\n# 5. Evaluate answer support, retrieval recall, abstention quality, and response time.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "healthcare": {
    "name": "Healthcare & Biomedical Sciences",
    "short": "Healthcare",
    "icon": "\ud83c\udfe5",
    "color": "#ef4444",
    "tagline": "Learn from health research data with care and curiosity.",
    "description": "Healthcare AI learning starts with approved data, well-defined labels, and careful evaluation. Models support research questions; a student result does not establish clinical usefulness. Start with prepared ECG features; move to retinal images or recorded instrument detection; attempt segmentation, federated learning, or retrospective risk models with suitable supervision. Beginner means basic Python with guided examples; intermediate means independent data preparation and model evaluation; advanced means domain knowledge plus deeper modelling. Levels describe the starting scope, not a guaranteed completion time.",
    "projects": [
      {
        "id": "hlth-01",
        "title": "Retinal Image Classification for a Research Prototype",
        "difficulty": "Intermediate",
        "summary": "Explore classification of labelled retinal images using a small research dataset. Start with a pretrained image model. When that works, compare fine-tuning choices and inspect errors across image quality levels. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a pretrained image model for this task? Keep the scope small enough to test fairly. Helpful background: Image classification and awareness of research-data permissions.",
        "ai_technique": "Starting approach: a pretrained image model. Extension: compare fine-tuning choices and inspect errors across image quality levels.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: A permitted, de-identified retinal-image dataset with documented labels and patient identifiers for splitting where available. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Class-level sensitivity, specificity, and uncertainty on patient-separated test data. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Generate Grad-CAM attribution heatmap for ophthalmologist review\nheatmap = grad_cam(model, retinal_image, target_layer='features.8')\noverlay = cv2.addWeighted(retinal_image, 0.6, heatmap, 0.4, 0)\n# Highlights microaneurysms, hemorrhages, and hard exudates"
      },
      {
        "id": "hlth-02",
        "title": "Brain MRI Segmentation as a Research Project",
        "difficulty": "Advanced",
        "summary": "Segment an annotated region in a permitted MRI research dataset and compare the output with reference masks. Start with a small segmentation baseline or reduced 2D task. When that works, compare a 3D network when data and computing resources permit. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a small segmentation baseline or reduced 2D task for this task? Keep the scope small enough to test fairly. Helpful background: Deep learning, medical-image formats, and segmentation.",
        "ai_technique": "Starting approach: a small segmentation baseline or reduced 2D task. Extension: compare a 3D network when data and computing resources permit.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: A de-identified, permitted MRI dataset with matching masks and documented image preprocessing. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Dice overlap, boundary errors, and variation across held-out patients. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom monai.networks.nets import UNet\nmodel = UNet(spatial_dims=3, in_channels=4, out_channels=3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))\n# Input: 4-channel 3D volume (T1, T1c, T2, FLAIR) -> Output: [Edema, Non-Enhancing Core, Enhancing Tumor]"
      },
      {
        "id": "hlth-03",
        "title": "Summarising Synthetic Clinical Notes with Source Checks",
        "difficulty": "Advanced",
        "summary": "Study whether a model can produce faithful short summaries of synthetic or permitted de-identified notes. Start with extractive summarisation. When that works, compare a language model; treat coding suggestions as a separate, expert-reviewed extension. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on extractive summarisation for this task? Keep the scope small enough to test fairly. Helpful background: Text processing, language-model evaluation, and data governance.",
        "ai_technique": "Starting approach: extractive summarisation. Extension: compare a language model; treat coding suggestions as a separate, expert-reviewed extension.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Synthetic notes or approved de-identified notes with manually checked summaries; do not upload identifiable records to external services. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Unsupported statements, omitted information, and agreement with a checked reference set. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom transformers import AutoTokenizer, AutoModelForSequenceClassification\ntokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')\n# Extracts ICD-10 diagnosis codes from 4,096-token patient doctor progress notes"
      },
      {
        "id": "hlth-04",
        "title": "Exploring ECG Beat Classification",
        "difficulty": "Beginner",
        "summary": "Learn to plot prepared ECG segments and classify a small set of existing research labels. Start with provided signal features and a simple classifier. When that works, compare a small 1D CNN after the baseline works. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on provided signal features and a simple classifier for this task? Keep the scope small enough to test fairly. Helpful background: Basic Python and guided classification; clinical interpretation needs qualified supervision.",
        "ai_technique": "Starting approach: provided signal features and a simple classifier. Extension: compare a small 1D CNN after the baseline works.",
        "tech_stack": [
          "Python",
          "Pandas",
          "NumPy",
          "Scikit-learn"
        ],
        "data_pipeline": "Suggested data: A permitted ECG research dataset with prepared segments, checked labels, and patient identifiers. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not required for the starting task. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Macro-F1, per-class recall, and errors on patients excluded from training. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n// Lightweight 1D convolution kernel on wearable embedded microcontroller\nint8_t ecg_quantized[250]; // 10 seconds sampled at 25 Hz\ntf_interpreter->Invoke();\nint8_t afib_prob = tf_interpreter->output(0)->data.int8[AFIB_INDEX];"
      },
      {
        "id": "hlth-05",
        "title": "Evaluating Models on Published Molecular Property Benchmarks",
        "difficulty": "Advanced",
        "summary": "Compare models on a fixed, published molecular-property benchmark as a computational evaluation exercise. Start with provided descriptors and a simple regressor. When that works, compare a graph model using the same fixed benchmark without proposing new molecules. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on provided descriptors and a simple regressor for this task? Keep the scope small enough to test fairly. Helpful background: Regression, graph learning, and research-methods supervision.",
        "ai_technique": "Starting approach: provided descriptors and a simple regressor. Extension: compare a graph model using the same fixed benchmark without proposing new molecules.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: An approved public benchmark with fixed molecular descriptors or graphs, labels, and documented splits. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Held-out prediction error, leakage checks, and limits of transferring benchmark results to experiments. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nfrom rdkit import Chem\nfrom torch_geometric.data import Data\n# Transform SMILES to 3D atomic graph with electrostatic and covalent edge features\nmol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')\npredicted_affinity = egnn_model(protein_graph, ligand_graph)"
      },
      {
        "id": "hlth-06",
        "title": "Histology Image Classification with Multiple Instance Learning",
        "difficulty": "Advanced",
        "summary": "Study slide-level labels using prepared image patches from a permitted research dataset. Start with a simple patch-feature aggregation method. When that works, compare attention-based multiple instance learning. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a simple patch-feature aggregation method for this task? Keep the scope small enough to test fairly. Helpful background: Computer vision, large-image handling, and deep learning.",
        "ai_technique": "Starting approach: a simple patch-feature aggregation method. Extension: compare attention-based multiple instance learning.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted de-identified histology data with slide labels, patient grouping, and prepared patches where available. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Slide-level metrics and errors on patient-separated test sets, including stain variation. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Extract 256x256 tiles, embed via ResNet50, then aggregate via attention gating\nfeatures = resnet50_feature_extractor(wsi_patches)\nattention_weights, slide_prediction = clam_model(features)\n# Generates gigapixel attention heatmap overlay"
      },
      {
        "id": "hlth-07",
        "title": "Studying Blood Pressure Estimation from PPG Signals",
        "difficulty": "Advanced",
        "summary": "Explore the relationship between paired PPG recordings and reference blood-pressure measurements in a research dataset. Start with a documented regression baseline. When that works, compare a sequence model and investigate calibration sensitivity. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a documented regression baseline for this task? Keep the scope small enough to test fairly. Helpful background: Signal processing, regression, and physiological measurement concepts.",
        "ai_technique": "Starting approach: a documented regression baseline. Extension: compare a sequence model and investigate calibration sensitivity.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Permitted paired PPG and reference measurements with timestamps and participant identifiers. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Error and variation across held-out participants; a low average error does not establish clinical validity. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Calculate pulse arrival time (PAT) and pulse transit time (PTT) from PPG derivatives\nfirst_derivative = np.diff(ppg_signal)\nsecond_derivative = np.diff(first_derivative)\nfeatures = np.stack([ppg_signal[2:], first_derivative[1:], second_derivative], axis=-1)\nsbp, dbp = model(torch.tensor(features))"
      },
      {
        "id": "hlth-08",
        "title": "Simulating Federated Learning for Health-Data Research",
        "difficulty": "Advanced",
        "summary": "Divide an approved dataset into simulated sites and compare local, centralised, and federated training. Start with separate local models and a centralised reference. When that works, implement a federated experiment and vary the simulated site distributions. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on separate local models and a centralised reference for this task? Keep the scope small enough to test fairly. Helpful background: Neural-network training, distributed-learning concepts, and privacy fundamentals.",
        "ai_technique": "Starting approach: separate local models and a centralised reference. Extension: implement a federated experiment and vary the simulated site distributions.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: A permitted de-identified or synthetic dataset partitioned into simulated sites with documented assumptions. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Predictive performance, communication cost, and site-level variation; federated training alone does not guarantee privacy. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\nimport flwr as fl\nclass HospitalClient(fl.client.NumPyClient):\n    def fit(self, parameters, config):\n        self.set_parameters(parameters)\n        train_with_differential_privacy(self.model, self.train_loader, epochs=1)\n        return self.get_parameters(), len(self.train_loader), {}"
      },
      {
        "id": "hlth-09",
        "title": "Evaluating a Retrospective ICU Risk Model",
        "difficulty": "Advanced",
        "summary": "Study a clearly defined historical deterioration label using approved research records. Start with a simple risk model using only data available before the prediction time. When that works, compare a more flexible model and examine calibration and subgroup errors. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a simple risk model using only data available before the prediction time for this task? Keep the scope small enough to test fairly. Helpful background: Time-series evaluation, clinical-research supervision, and leakage prevention.",
        "ai_technique": "Starting approach: a simple risk model using only data available before the prediction time. Extension: compare a more flexible model and examine calibration and subgroup errors.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)",
          "Jupyter / Colab"
        ],
        "data_pipeline": "Suggested data: Approved de-identified ICU data with a documented outcome definition, time windows, and access conditions. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision-recall, calibration, false-alert burden, and patient-separated temporal evaluation. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Real-time bedside vital feature windowing (HR, MAP, Temp, SpO2, WBC, Lactate)\npatient_vitals_6h = fetch_icu_telemetry(bed_id, window_hours=6)\nrisk_score = sepsis_model.predict_proba(patient_vitals_6h)[0][1]\nif risk_score > 0.65:\n    dispatch_icu_pager_alert(bed_id, 'High Sepsis Risk: Check Lactate & Blood Cultures')"
      },
      {
        "id": "hlth-10",
        "title": "Recognising Instruments in Recorded Surgical Training Video",
        "difficulty": "Intermediate",
        "summary": "Detect a limited set of labelled instruments in permitted recorded training footage. Start with a pretrained detector. When that works, compare fine-tuning or add phase classification as a separate extension. Finish with a notebook, comparison plots, and a short explanation of what you learned.",
        "problem": "Your project question: can your chosen approach improve on a pretrained detector for this task? Keep the scope small enough to test fairly. Helpful background: Object detection and careful video-data preparation.",
        "ai_technique": "Starting approach: a pretrained detector. Extension: compare fine-tuning or add phase classification as a separate extension.",
        "tech_stack": [
          "Python",
          "NumPy",
          "Scikit-learn",
          "PyTorch (optional neural extension)"
        ],
        "data_pipeline": "Suggested data: Permitted de-identified training footage with instrument annotations and procedure-level grouping. Confirm access and permitted use before committing. Clean the data, document the split, fit preprocessing on training data only, and keep the final test set separate. Data is not bundled with this description.",
        "hardware_target": "Begin with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with the neural-network extension; check memory and runtime before scaling up. The tool list is a suggestion, not a required installation list. Device deployment is optional and outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors on held-out procedures with occlusion or motion blur. Report the results you actually obtain, including cases where the baseline works better. No accuracy, savings, or deployment outcome is guaranteed. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Optional advanced illustration from the original catalog, not a complete runnable project.\n# This excerpt may need corrections, missing definitions, and compatible dependencies.\n# Build the starting baseline described above before attempting this extension.\n# Process laparoscopic endoscopic video feed\nframe_features = spatial_cnn(video_frame)\nphase_logits = temporal_tcn(frame_features)\ncurrent_surgical_phase = phases[torch.argmax(phase_logits)]\n# e.g., 'CalotTriangleDissection', 'ClippingAndCutting', 'GallbladderPackaging'"
      },
      {
        "id": "hlth-11",
        "title": "Appointment Attendance Prediction from Synthetic Records",
        "difficulty": "Beginner",
        "summary": "Explore factors associated with attendance in a synthetic or approved scheduling table. Start with logistic regression. When that works, compare a tree model and investigate calibration. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model and investigate calibration.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or de-identified appointment records with outcome labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision-recall, calibration, and subgroup errors; do not use the model to restrict care. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Appointment Attendance Prediction from Synthetic Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression.\n# 4. Optional extension: compare a tree model and investigate calibration.\n# 5. Evaluate precision-recall, calibration, and subgroup errors; do not use the model to restrict care.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-12",
        "title": "Hospital Visit Volume Forecasting",
        "difficulty": "Beginner",
        "summary": "Forecast aggregate daily visit counts using past counts and calendar features. Start with a seasonal-naive forecast. When that works, compare regression. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal-naive forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a seasonal-naive forecast. Extension: compare regression.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: public aggregate or synthetic daily visit counts without patient identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and peak-day error on future dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Hospital Visit Volume Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal-naive forecast.\n# 4. Optional extension: compare regression.\n# 5. Evaluate MAE and peak-day error on future dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-13",
        "title": "Physical Activity Classification from Wearable Records",
        "difficulty": "Beginner",
        "summary": "Recognise a few labelled activity types from prepared wearable features. Start with a decision tree. When that works, compare a small sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a decision tree for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a decision tree. Extension: compare a small sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted activity recordings with participant identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on participants excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Physical Activity Classification from Wearable Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a decision tree.\n# 4. Optional extension: compare a small sequence model.\n# 5. Evaluate macro-F1 on participants excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-14",
        "title": "Healthcare Data Quality Issue Detection",
        "difficulty": "Beginner",
        "summary": "Build a notebook that flags missing, inconsistent, or out-of-range values in a synthetic table. Start with documented validation rules. When that works, compare an anomaly detector on injected errors. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on documented validation rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: documented validation rules. Extension: compare an anomaly detector on injected errors.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic records with known units and explicitly injected data-quality issues. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and false alarms for the known errors. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Healthcare Data Quality Issue Detection\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with documented validation rules.\n# 4. Optional extension: compare an anomaly detector on injected errors.\n# 5. Evaluate precision, recall, and false alarms for the known errors.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-15",
        "title": "Rehabilitation Exercise Recognition from Recorded Poses",
        "difficulty": "Intermediate",
        "summary": "Classify a small set of exercise movements from permitted pose sequences. Start with motion features and a simple classifier. When that works, compare a sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on motion features and a simple classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: motion features and a simple classifier. Extension: compare a sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: consented or public research pose data with movement labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on held-out participants; labels do not determine whether exercise is medically appropriate. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Rehabilitation Exercise Recognition from Recorded Poses\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with motion features and a simple classifier.\n# 4. Optional extension: compare a sequence model.\n# 5. Evaluate macro-F1 and errors on held-out participants; labels do not determine whether exercise is medically appropriate.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-16",
        "title": "Sleep Stage Classification from Prepared Signal Features",
        "difficulty": "Intermediate",
        "summary": "Explore sleep-stage labels using a documented research dataset. Start with a conventional classifier on provided features. When that works, compare a sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a conventional classifier on provided features for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a conventional classifier on provided features. Extension: compare a sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted sleep recordings with expert labels and participant grouping. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and class-level recall on held-out participants. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Sleep Stage Classification from Prepared Signal Features\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a conventional classifier on provided features.\n# 4. Optional extension: compare a sequence model.\n# 5. Evaluate macro-F1 and class-level recall on held-out participants.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-17",
        "title": "Medical Image Quality Classification",
        "difficulty": "Intermediate",
        "summary": "Identify a small set of labelled image-quality issues in research images. Start with simple image-quality features. When that works, compare a compact image classifier. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on simple image-quality features for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: simple image-quality features. Extension: compare a compact image classifier.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted de-identified images with checked quality labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors across held-out imaging sources. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Medical Image Quality Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with simple image-quality features.\n# 4. Optional extension: compare a compact image classifier.\n# 5. Evaluate precision, recall, and errors across held-out imaging sources.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-18",
        "title": "Laboratory Sample Turnaround-Time Forecasting",
        "difficulty": "Beginner",
        "summary": "Estimate a defined processing duration from anonymised operational records. Start with a historical median and regression. When that works, compare a tree model with features available at intake. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a historical median and regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a historical median and regression. Extension: compare a tree model with features available at intake.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or approved operational records without patient-identifying information. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and error on long-duration cases in later periods. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Laboratory Sample Turnaround-Time Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a historical median and regression.\n# 4. Optional extension: compare a tree model with features available at intake.\n# 5. Evaluate MAE and error on long-duration cases in later periods.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-19",
        "title": "Calibration of a Research Classification Model",
        "difficulty": "Advanced",
        "summary": "Study whether a health-research model's confidence matches its observed correctness. Start with an existing model with a fixed evaluation split. When that works, compare a calibration method fitted on separate validation data. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on an existing model with a fixed evaluation split for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: an existing model with a fixed evaluation split. Extension: compare a calibration method fitted on separate validation data.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a permitted research dataset and saved model scores with true labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Calibration error, discrimination, and variation across held-out groups. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Calibration of a Research Classification Model\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with an existing model with a fixed evaluation split.\n# 4. Optional extension: compare a calibration method fitted on separate validation data.\n# 5. Evaluate calibration error, discrimination, and variation across held-out groups.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "hlth-20",
        "title": "Health Model Robustness Across Data Sources",
        "difficulty": "Advanced",
        "summary": "Compare one research model's behaviour across two approved data sources. Start with a fixed baseline model. When that works, test limited adaptation using a separate evaluation set. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a fixed baseline model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a fixed baseline model. Extension: test limited adaptation using a separate evaluation set.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: compatible de-identified research datasets with documented label definitions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Performance change, missing-data sensitivity, and source-specific errors. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance. This is a research-learning prototype, not a tool for patient-care decisions.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Health Model Robustness Across Data Sources\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a fixed baseline model.\n# 4. Optional extension: test limited adaptation using a separate evaluation set.\n# 5. Evaluate performance change, missing-data sensitivity, and source-specific errors.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "cs": {
    "name": "Computer Science & Software Systems",
    "short": "CS",
    "icon": "\ud83d\udcbb",
    "color": "#7c3aed",
    "tagline": "Build useful software while learning how to evaluate AI.",
    "description": "Reliable AI software depends on sound programming, testing, and data handling. Start with text classification or notes search; progress to recommendations and logs; explore grounded assistants and model monitoring when ready. The level describes the starting scope: beginner projects use guided baselines, intermediate projects compare methods independently, and advanced projects need deeper domain and modelling skills.",
    "projects": [
      {
        "id": "cs-01",
        "title": "Email Spam Classification",
        "difficulty": "Beginner",
        "summary": "Classify sample messages as spam or non-spam using a small labelled dataset. Start with TF-IDF and logistic regression. When that works, compare a second classifier and inspect difficult messages. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on TF-IDF and logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: TF-IDF and logistic regression. Extension: compare a second classifier and inspect difficult messages.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted, redacted message text with spam labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and false positives on held-out sources. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Email Spam Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with TF-IDF and logistic regression.\n# 4. Optional extension: compare a second classifier and inspect difficult messages.\n# 5. Evaluate precision, recall, and false positives on held-out sources.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-02",
        "title": "Movie Review Sentiment Classification",
        "difficulty": "Beginner",
        "summary": "Predict positive or negative sentiment from labelled reviews. Start with a bag-of-words linear classifier. When that works, compare pretrained text embeddings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a bag-of-words linear classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a bag-of-words linear classifier. Extension: compare pretrained text embeddings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted review text with sentiment labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on reviews from unseen products or sources. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Movie Review Sentiment Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a bag-of-words linear classifier.\n# 4. Optional extension: compare pretrained text embeddings.\n# 5. Evaluate macro-F1 and errors on reviews from unseen products or sources.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-03",
        "title": "Handwritten Digit Recognition",
        "difficulty": "Beginner",
        "summary": "Build an image classifier for a small, clearly labelled digit task. Start with a linear image classifier. When that works, compare a small convolutional network. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a linear image classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a linear image classifier. Extension: compare a small convolutional network.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a permitted labelled digit-image dataset. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Accuracy, per-class recall, and a confusion matrix on held-out images. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Handwritten Digit Recognition\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a linear image classifier.\n# 4. Optional extension: compare a small convolutional network.\n# 5. Evaluate accuracy, per-class recall, and a confusion matrix on held-out images.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-04",
        "title": "News Topic Classification",
        "difficulty": "Beginner",
        "summary": "Organise sample news articles into a few labelled topics. Start with TF-IDF and a linear classifier. When that works, compare embeddings while checking duplicate articles. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on TF-IDF and a linear classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: TF-IDF and a linear classifier. Extension: compare embeddings while checking duplicate articles.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted article text with topic labels and dates. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and performance on later articles. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# News Topic Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with TF-IDF and a linear classifier.\n# 4. Optional extension: compare embeddings while checking duplicate articles.\n# 5. Evaluate macro-F1 and performance on later articles.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-05",
        "title": "Personal Notes Semantic Search",
        "difficulty": "Beginner",
        "summary": "Search a small collection of your own notes using a set of checked example queries. Start with keyword search. When that works, compare sentence embeddings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword search for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: keyword search. Extension: compare sentence embeddings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: your own or synthetic notes and manually judged query relevance. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Retrieval precision and recall at a chosen result count. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Personal Notes Semantic Search\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword search.\n# 4. Optional extension: compare sentence embeddings.\n# 5. Evaluate retrieval precision and recall at a chosen result count.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-06",
        "title": "Simple Content Recommendation System",
        "difficulty": "Beginner",
        "summary": "Recommend items from a small anonymised interaction table. Start with a popularity list. When that works, compare item-based collaborative filtering. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a popularity list for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a popularity list. Extension: compare item-based collaborative filtering.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted item interactions with user identifiers anonymised. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision at a chosen list length and coverage of less-popular items. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Simple Content Recommendation System\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a popularity list.\n# 4. Optional extension: compare item-based collaborative filtering.\n# 5. Evaluate precision at a chosen list length and coverage of less-popular items.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-07",
        "title": "Application Log Anomaly Detection",
        "difficulty": "Intermediate",
        "summary": "Flag unusual patterns in synthetic or approved application logs. Start with documented frequency thresholds. When that works, compare an isolation forest on aggregate features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on documented frequency thresholds for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: documented frequency thresholds. Extension: compare an isolation forest on aggregate features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: redacted logs or synthetic events with labelled test anomalies. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: False-alarm rate and detection of known events. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Application Log Anomaly Detection\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with documented frequency thresholds.\n# 4. Optional extension: compare an isolation forest on aggregate features.\n# 5. Evaluate false-alarm rate and detection of known events.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-08",
        "title": "Image Similarity Search",
        "difficulty": "Intermediate",
        "summary": "Retrieve visually similar images from a permitted collection. Start with simple colour features. When that works, compare pretrained image embeddings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on simple colour features for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: simple colour features. Extension: compare pretrained image embeddings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted images and manually checked query-relevance pairs. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Retrieval precision and errors across image categories. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Image Similarity Search\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with simple colour features.\n# 4. Optional extension: compare pretrained image embeddings.\n# 5. Evaluate retrieval precision and errors across image categories.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-09",
        "title": "Duplicate Question Detection",
        "difficulty": "Intermediate",
        "summary": "Identify pairs of questions that express the same intent. Start with text similarity with TF-IDF. When that works, compare sentence embeddings or a small pair classifier. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on text similarity with TF-IDF for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: text similarity with TF-IDF. Extension: compare sentence embeddings or a small pair classifier.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted question pairs with checked duplicate labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors on unfamiliar topics. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Duplicate Question Detection\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with text similarity with TF-IDF.\n# 4. Optional extension: compare sentence embeddings or a small pair classifier.\n# 5. Evaluate precision, recall, and errors on unfamiliar topics.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-10",
        "title": "Software Issue Category Prediction",
        "difficulty": "Intermediate",
        "summary": "Suggest categories for redacted software issue reports. Start with a text classifier. When that works, compare an embedding model and include an uncertain-case route. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a text classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a text classifier. Extension: compare an embedding model and include an uncertain-case route.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted issue text with historical labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on later issues and errors across repositories. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Software Issue Category Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a text classifier.\n# 4. Optional extension: compare an embedding model and include an uncertain-case route.\n# 5. Evaluate macro-F1 on later issues and errors across repositories.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-11",
        "title": "API Response-Time Prediction",
        "difficulty": "Intermediate",
        "summary": "Predict request latency from recorded workload features in a controlled test. Start with linear regression. When that works, compare a tree regressor. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree regressor.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or approved benchmark logs with known request and workload features. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors at the slow end of the latency distribution. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# API Response-Time Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree regressor.\n# 5. Evaluate MAE and errors at the slow end of the latency distribution.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-12",
        "title": "Document Layout Classification",
        "difficulty": "Intermediate",
        "summary": "Classify sample document pages into a few layout categories. Start with image features and a simple classifier. When that works, compare a pretrained vision model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on image features and a simple classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: image features and a simple classifier. Extension: compare a pretrained vision model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or permitted page images with category labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on unseen document templates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Document Layout Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with image features and a simple classifier.\n# 4. Optional extension: compare a pretrained vision model.\n# 5. Evaluate macro-F1 on unseen document templates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-13",
        "title": "Evaluating a Retrieval-Augmented Study Assistant",
        "difficulty": "Advanced",
        "summary": "Build and evaluate an assistant that answers from a fixed set of permitted study documents. Start with retrieval with direct quoted passages. When that works, compare grounded language-model answers with source references. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on retrieval with direct quoted passages for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: retrieval with direct quoted passages. Extension: compare grounded language-model answers with source references.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: approved study materials and manually checked questions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Source support, retrieval quality, and appropriate abstention. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Evaluating a Retrieval-Augmented Study Assistant\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with retrieval with direct quoted passages.\n# 4. Optional extension: compare grounded language-model answers with source references.\n# 5. Evaluate source support, retrieval quality, and appropriate abstention.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-14",
        "title": "Model Drift Monitoring in a Simulated Data Stream",
        "difficulty": "Advanced",
        "summary": "Study how model performance changes when a controlled data stream shifts. Start with a fixed model and simple distribution checks. When that works, compare drift detectors with delayed labelled evaluation. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a fixed model and simple distribution checks for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a fixed model and simple distribution checks. Extension: compare drift detectors with delayed labelled evaluation.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a documented synthetic stream or permitted historical data with controlled splits. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: False alerts, detection delay, and measured performance after changes. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Model Drift Monitoring in a Simulated Data Stream\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a fixed model and simple distribution checks.\n# 4. Optional extension: compare drift detectors with delayed labelled evaluation.\n# 5. Evaluate false alerts, detection delay, and measured performance after changes.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-15",
        "title": "Federated Learning Across Simulated Clients",
        "difficulty": "Advanced",
        "summary": "Compare local and shared training across synthetic client partitions. Start with local models and a centralised reference. When that works, implement federated averaging and vary client distributions. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on local models and a centralised reference for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: local models and a centralised reference. Extension: implement federated averaging and vary client distributions.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a permitted non-sensitive dataset divided into documented simulated clients. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Accuracy, communication volume, and variation between clients. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Federated Learning Across Simulated Clients\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with local models and a centralised reference.\n# 4. Optional extension: implement federated averaging and vary client distributions.\n# 5. Evaluate accuracy, communication volume, and variation between clients.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-16",
        "title": "Language Model Answer Reliability Evaluation",
        "difficulty": "Advanced",
        "summary": "Evaluate answers to a fixed set of factual questions grounded in supplied documents. Start with extractive answers from reference text. When that works, compare model prompts or retrieval settings without changing the final test set. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on extractive answers from reference text for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: extractive answers from reference text. Extension: compare model prompts or retrieval settings without changing the final test set.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted reference documents and checked answers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Unsupported-claim rate, omissions, and abstention quality. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Language Model Answer Reliability Evaluation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with extractive answers from reference text.\n# 4. Optional extension: compare model prompts or retrieval settings without changing the final test set.\n# 5. Evaluate unsupported-claim rate, omissions, and abstention quality.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-17",
        "title": "Model Compression for a Small Image Application",
        "difficulty": "Advanced",
        "summary": "Compare an image classifier before and after a compression step. Start with an evaluated uncompressed model. When that works, test quantisation or pruning with the same test images. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on an evaluated uncompressed model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: an evaluated uncompressed model. Extension: test quantisation or pruning with the same test images.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a permitted image dataset and documented hardware measurements. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Accuracy change, model size, and measured inference latency. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Model Compression for a Small Image Application\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with an evaluated uncompressed model.\n# 4. Optional extension: test quantisation or pruning with the same test images.\n# 5. Evaluate accuracy change, model size, and measured inference latency.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-18",
        "title": "Reproducible Machine Learning Experiment Tracker",
        "difficulty": "Intermediate",
        "summary": "Build a small tool that records experiments for one classification task. Start with a structured CSV log and saved configurations. When that works, add a dashboard comparing runs and held-out metrics. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a structured CSV log and saved configurations for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a structured CSV log and saved configurations. Extension: add a dashboard comparing runs and held-out metrics.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a permitted example dataset and your own experiment records. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Successful reruns, configuration completeness, and correct metric comparisons. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Reproducible Machine Learning Experiment Tracker\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a structured CSV log and saved configurations.\n# 4. Optional extension: add a dashboard comparing runs and held-out metrics.\n# 5. Evaluate successful reruns, configuration completeness, and correct metric comparisons.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-19",
        "title": "Uncertainty-Aware Image Classification",
        "difficulty": "Advanced",
        "summary": "Study when a small image classifier should defer a prediction. Start with a calibrated classifier with a confidence threshold. When that works, compare an uncertainty method using a separate validation set. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a calibrated classifier with a confidence threshold for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a calibrated classifier with a confidence threshold. Extension: compare an uncertainty method using a separate validation set.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted labelled images including clearly separated out-of-distribution examples. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Error versus coverage and calibration on untouched test data. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Uncertainty-Aware Image Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a calibrated classifier with a confidence threshold.\n# 4. Optional extension: compare an uncertainty method using a separate validation set.\n# 5. Evaluate error versus coverage and calibration on untouched test data.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "cs-20",
        "title": "Accessibility Issue Classification from Website Reports",
        "difficulty": "Intermediate",
        "summary": "Classify synthetic audit findings into a small set of accessibility issue categories. Start with keyword rules. When that works, compare a text classifier and review ambiguous findings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: keyword rules. Extension: compare a text classifier and review ambiguous findings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or permitted audit descriptions with checked categories. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors across sites; automated labels do not establish accessibility compliance. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Accessibility Issue Classification from Website Reports\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword rules.\n# 4. Optional extension: compare a text classifier and review ambiguous findings.\n# 5. Evaluate macro-F1 and errors across sites; automated labels do not establish accessibility compliance.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "mechanical": {
    "name": "Mechanical Engineering",
    "short": "ME",
    "icon": "\u2699\ufe0f",
    "color": "#64748b",
    "tagline": "Connect machines, measurements, and models.",
    "description": "Mechanical principles and measurement quality remain the foundation. These projects use recorded data or simulation to investigate a manageable engineering question. Start with temperature forecasts or part images; progress to tool wear and vibration; explore simulated control and design optimisation after building the prerequisites. The level describes the starting scope: beginner projects use guided baselines, intermediate projects compare methods independently, and advanced projects need deeper domain and modelling skills.",
    "projects": [
      {
        "id": "mech-01",
        "title": "Machine Temperature Forecasting",
        "difficulty": "Beginner",
        "summary": "Forecast a machine's next recorded temperature from recent measurements. Start with a persistence forecast. When that works, compare a regression model with load features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a persistence forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a persistence forecast. Extension: compare a regression model with load features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or simulated timestamped temperature and load records. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors during heating or cooling transitions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Machine Temperature Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a persistence forecast.\n# 4. Optional extension: compare a regression model with load features.\n# 5. Evaluate MAE and errors during heating or cooling transitions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-02",
        "title": "Tool Wear Estimation from Prepared Features",
        "difficulty": "Beginner",
        "summary": "Estimate measured tool wear from a small table of machining features. Start with linear regression. When that works, compare a tree model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted machining measurements with wear reference values. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on held-out tools or runs. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Tool Wear Estimation from Prepared Features\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree model.\n# 5. Evaluate MAE on held-out tools or runs.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-03",
        "title": "Manufactured Part Image Classification",
        "difficulty": "Beginner",
        "summary": "Classify a limited set of visible part conditions in labelled images. Start with transfer learning with a small image classifier. When that works, compare a second model and inspect false rejects. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on transfer learning with a small image classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: transfer learning with a small image classifier. Extension: compare a second model and inspect false rejects.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted part images grouped by manufacturing batch. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors on unseen batches. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Manufactured Part Image Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with transfer learning with a small image classifier.\n# 4. Optional extension: compare a second model and inspect false rejects.\n# 5. Evaluate precision, recall, and errors on unseen batches.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-04",
        "title": "Fuel Consumption Prediction from Vehicle Records",
        "difficulty": "Beginner",
        "summary": "Estimate recorded fuel use from trip and vehicle features. Start with linear regression. When that works, compare a tree regressor. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree regressor.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or synthetic trip summaries with consistent consumption units. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on vehicles or trips excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Fuel Consumption Prediction from Vehicle Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree regressor.\n# 5. Evaluate MAE on vehicles or trips excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-05",
        "title": "Machine Operating-State Classification",
        "difficulty": "Beginner",
        "summary": "Identify recorded operating states from prepared sensor features. Start with a decision tree. When that works, compare an ensemble and inspect confused states. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a decision tree for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a decision tree. Extension: compare an ensemble and inspect confused states.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or simulated sensor records with state labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on held-out operating sessions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Machine Operating-State Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a decision tree.\n# 4. Optional extension: compare an ensemble and inspect confused states.\n# 5. Evaluate macro-F1 on held-out operating sessions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-06",
        "title": "Heat Exchanger Performance Estimation",
        "difficulty": "Beginner",
        "summary": "Estimate one measured heat-exchanger output from a small input table. Start with a documented engineering baseline or regression. When that works, compare a tree model within the measured operating range. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented engineering baseline or regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a documented engineering baseline or regression. Extension: compare a tree model within the measured operating range.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted laboratory records or clearly labelled simulation data. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Prediction error and consistency with stated physical bounds. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Heat Exchanger Performance Estimation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented engineering baseline or regression.\n# 4. Optional extension: compare a tree model within the measured operating range.\n# 5. Evaluate prediction error and consistency with stated physical bounds.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-07",
        "title": "Rotating Machinery Fault Classification",
        "difficulty": "Intermediate",
        "summary": "Classify labelled machine conditions from recorded vibration signals. Start with signal features and a conventional classifier. When that works, compare a compact sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on signal features and a conventional classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: signal features and a conventional classifier. Extension: compare a compact sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted vibration recordings with machine and session identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on unseen machines or operating conditions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Rotating Machinery Fault Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with signal features and a conventional classifier.\n# 4. Optional extension: compare a compact sequence model.\n# 5. Evaluate macro-F1 and errors on unseen machines or operating conditions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-08",
        "title": "Remaining Useful Life Prediction from Simulated Run-to-Failure Data",
        "difficulty": "Intermediate",
        "summary": "Estimate a defined remaining-life target in a research dataset. Start with a simple regression model. When that works, compare a sequence model and analyse uncertainty. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a simple regression model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a simple regression model. Extension: compare a sequence model and analyse uncertainty.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted run-to-failure simulations or research records with unit identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Prediction error on held-out units and errors near end of life. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Remaining Useful Life Prediction from Simulated Run-to-Failure Data\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a simple regression model.\n# 4. Optional extension: compare a sequence model and analyse uncertainty.\n# 5. Evaluate prediction error on held-out units and errors near end of life.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-09",
        "title": "Weld Defect Detection in Images",
        "difficulty": "Intermediate",
        "summary": "Detect a small set of labelled weld-image defects. Start with transfer-learning classification. When that works, compare object detection or segmentation when annotations support it. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on transfer-learning classification for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: transfer-learning classification. Extension: compare object detection or segmentation when annotations support it.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted inspection images with checked defect labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Class-level recall and false positives on unseen welds. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Weld Defect Detection in Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with transfer-learning classification.\n# 4. Optional extension: compare object detection or segmentation when annotations support it.\n# 5. Evaluate class-level recall and false positives on unseen welds.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-10",
        "title": "Machining Energy Use Prediction",
        "difficulty": "Intermediate",
        "summary": "Estimate energy use for a defined machining operation. Start with linear regression. When that works, compare a tree ensemble across material or speed settings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree ensemble across material or speed settings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or simulated process records with measured energy. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors on held-out operating settings. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Machining Energy Use Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree ensemble across material or speed settings.\n# 5. Evaluate MAE and errors on held-out operating settings.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-11",
        "title": "Additive Manufacturing Quality Classification",
        "difficulty": "Intermediate",
        "summary": "Relate recorded process features to labelled part-quality outcomes. Start with a conventional classifier. When that works, compare a multimodal model only if aligned images and sensor data exist. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a conventional classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a conventional classifier. Extension: compare a multimodal model only if aligned images and sensor data exist.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted print records with batch identifiers and measured quality labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 on print jobs excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Additive Manufacturing Quality Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a conventional classifier.\n# 4. Optional extension: compare a multimodal model only if aligned images and sensor data exist.\n# 5. Evaluate macro-F1 on print jobs excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-12",
        "title": "Aerodynamic Coefficient Surrogate Modelling",
        "difficulty": "Intermediate",
        "summary": "Estimate a coefficient from a small, documented set of simulation cases. Start with regression on bounded design variables. When that works, compare a small neural model and validate difficult cases with the simulator. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on regression on bounded design variables for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: regression on bounded design variables. Extension: compare a small neural model and validate difficult cases with the simulator.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: documented simulation inputs and coefficient outputs within a stated design range. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Held-out error and behaviour near the boundaries of the training range. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Aerodynamic Coefficient Surrogate Modelling\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with regression on bounded design variables.\n# 4. Optional extension: compare a small neural model and validate difficult cases with the simulator.\n# 5. Evaluate held-out error and behaviour near the boundaries of the training range.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-13",
        "title": "Thermal System Control with Reinforcement Learning",
        "difficulty": "Advanced",
        "summary": "Compare controllers in a simple thermal-process simulation. Start with a tuned PID controller. When that works, train a bounded reinforcement-learning policy. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a tuned PID controller for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a tuned PID controller. Extension: train a bounded reinforcement-learning policy.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: a documented thermal simulator with saved initial conditions and disturbances. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Tracking error, simulated energy use, and limit violations. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Thermal System Control with Reinforcement Learning\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a tuned PID controller.\n# 4. Optional extension: train a bounded reinforcement-learning policy.\n# 5. Evaluate tracking error, simulated energy use, and limit violations.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-14",
        "title": "Robot Grasp Point Prediction in Simulation",
        "difficulty": "Advanced",
        "summary": "Predict picking points for a limited set of objects in simulated scenes. Start with a geometric selection rule. When that works, compare a vision model using synthetic ground truth. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a geometric selection rule for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a geometric selection rule. Extension: compare a vision model using synthetic ground truth.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic object scenes with known poses and picking targets. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Localisation error and simulated success across unseen object poses. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Robot Grasp Point Prediction in Simulation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a geometric selection rule.\n# 4. Optional extension: compare a vision model using synthetic ground truth.\n# 5. Evaluate localisation error and simulated success across unseen object poses.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-15",
        "title": "Digital Twin Residual Analysis for a Small Machine Model",
        "difficulty": "Advanced",
        "summary": "Compare a physical simulation with recorded or generated observations. Start with a calibrated physics model. When that works, learn residual errors while checking the underlying assumptions. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a calibrated physics model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a calibrated physics model. Extension: learn residual errors while checking the underlying assumptions.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: paired simulator outputs and permitted measurements with operating conditions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Prediction error, residual structure, and stability under unseen settings. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Digital Twin Residual Analysis for a Small Machine Model\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a calibrated physics model.\n# 4. Optional extension: learn residual errors while checking the underlying assumptions.\n# 5. Evaluate prediction error, residual structure, and stability under unseen settings.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-16",
        "title": "Topology Optimisation with a Learned Surrogate",
        "difficulty": "Advanced",
        "summary": "Explore material layouts for a small constrained mechanical design problem. Start with a conventional optimisation routine. When that works, fit a surrogate and verify proposals with finite-element simulation. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a conventional optimisation routine for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a conventional optimisation routine. Extension: fit a surrogate and verify proposals with finite-element simulation.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: documented structural simulations with fixed loads and boundary conditions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Verified compliance, material usage, and constraint violations. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Topology Optimisation with a Learned Surrogate\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a conventional optimisation routine.\n# 4. Optional extension: fit a surrogate and verify proposals with finite-element simulation.\n# 5. Evaluate verified compliance, material usage, and constraint violations.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-17",
        "title": "Multi-Sensor Machine Condition Modelling",
        "difficulty": "Advanced",
        "summary": "Compare individual sensor models with a combined model for a defined condition label. Start with one-sensor classifiers. When that works, compare feature fusion and missing-sensor robustness. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on one-sensor classifiers for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: one-sensor classifiers. Extension: compare feature fusion and missing-sensor robustness.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: aligned permitted sensor recordings with unit identifiers and checked labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1, sensor-ablation results, and errors on unseen machines. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Multi-Sensor Machine Condition Modelling\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with one-sensor classifiers.\n# 4. Optional extension: compare feature fusion and missing-sensor robustness.\n# 5. Evaluate macro-F1, sensor-ablation results, and errors on unseen machines.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-18",
        "title": "Pump Efficiency Estimation from Operating Records",
        "difficulty": "Intermediate",
        "summary": "Estimate recorded pump efficiency within a documented operating range. Start with a conventional curve fit. When that works, compare a tree regressor. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a conventional curve fit for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a conventional curve fit. Extension: compare a tree regressor.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted flow, pressure, power, and reference-efficiency records. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and physically inconsistent predictions on held-out runs. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Pump Efficiency Estimation from Operating Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a conventional curve fit.\n# 4. Optional extension: compare a tree regressor.\n# 5. Evaluate MAE and physically inconsistent predictions on held-out runs.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-19",
        "title": "Predictive Maintenance Alert Evaluation",
        "difficulty": "Advanced",
        "summary": "Study the trade-off between early warnings and false alarms for one machine dataset. Start with a threshold-based alert. When that works, compare a learned risk model using time-aware evaluation. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a threshold-based alert for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a threshold-based alert. Extension: compare a learned risk model using time-aware evaluation.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted maintenance histories with event times and pre-event measurements. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: False-alert burden, missed events, and lead-time-dependent performance. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Predictive Maintenance Alert Evaluation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a threshold-based alert.\n# 4. Optional extension: compare a learned risk model using time-aware evaluation.\n# 5. Evaluate false-alert burden, missed events, and lead-time-dependent performance.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "mech-20",
        "title": "Mechanical Test Curve Classification",
        "difficulty": "Intermediate",
        "summary": "Classify a few labelled material-test curve types. Start with curve features and a conventional classifier. When that works, compare a small sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on curve features and a conventional classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: curve features and a conventional classifier. Extension: compare a small sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted stress-strain or other test curves with documented labels and specimen grouping. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on specimens excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Mechanical Test Curve Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with curve features and a conventional classifier.\n# 4. Optional extension: compare a small sequence model.\n# 5. Evaluate macro-F1 and errors on specimens excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "environment": {
    "name": "Environmental Science",
    "short": "ENV",
    "icon": "\ud83c\udf0d",
    "color": "#059669",
    "tagline": "Use data to explore environmental questions with care.",
    "description": "Environmental models depend on measurement quality, time, and location. Explore the evidence and its uncertainty before drawing conclusions beyond the study area. Start with pollution records, waste images, or rainfall; progress to satellite mapping; explore spatial uncertainty and planning trade-offs with suitable background. The level describes the starting scope: beginner projects use guided baselines, intermediate projects compare methods independently, and advanced projects need deeper domain and modelling skills.",
    "projects": [
      {
        "id": "env-01",
        "title": "Air Pollution Concentration Forecasting",
        "difficulty": "Beginner",
        "summary": "Forecast one measured pollutant for a defined future interval. Start with a persistence forecast and regression. When that works, compare a tree model using weather and past observations. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a persistence forecast and regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a persistence forecast and regression. Extension: compare a tree model using weather and past observations.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted timestamped pollutant and weather records with units. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on later dates and errors during high-concentration periods. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Air Pollution Concentration Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a persistence forecast and regression.\n# 4. Optional extension: compare a tree model using weather and past observations.\n# 5. Evaluate MAE on later dates and errors during high-concentration periods.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-02",
        "title": "Rainfall Prediction from Weather Records",
        "difficulty": "Beginner",
        "summary": "Predict whether rain occurs in a clearly defined future period. Start with logistic regression. When that works, compare a tree model and calibration. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model and calibration.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted historical weather records with a documented rainfall threshold. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and calibration on held-out dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Rainfall Prediction from Weather Records\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression.\n# 4. Optional extension: compare a tree model and calibration.\n# 5. Evaluate precision, recall, and calibration on held-out dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-03",
        "title": "Waste Image Classification",
        "difficulty": "Beginner",
        "summary": "Classify a small set of labelled waste items in photographs. Start with transfer learning with a small image model. When that works, test mixed backgrounds and uncertain cases. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on transfer learning with a small image model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: transfer learning with a small image model. Extension: test mixed backgrounds and uncertain cases.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted waste images with category labels and source grouping. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on unfamiliar image sources. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Waste Image Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with transfer learning with a small image model.\n# 4. Optional extension: test mixed backgrounds and uncertain cases.\n# 5. Evaluate macro-F1 and errors on unfamiliar image sources.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-04",
        "title": "Noise Level Prediction",
        "difficulty": "Beginner",
        "summary": "Estimate recorded noise level from a small set of environmental features. Start with linear regression. When that works, compare a tree regressor. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: linear regression. Extension: compare a tree regressor.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted monitoring records with measurement units, locations, and times. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on held-out locations or periods. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Noise Level Prediction\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with linear regression.\n# 4. Optional extension: compare a tree regressor.\n# 5. Evaluate MAE on held-out locations or periods.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-05",
        "title": "Carbon Emission Estimate Modelling",
        "difficulty": "Beginner",
        "summary": "Predict a documented emissions estimate from activity data and compare it with its calculation method. Start with a transparent factor-based calculation. When that works, compare regression while explaining label assumptions. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a transparent factor-based calculation for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a transparent factor-based calculation. Extension: compare regression while explaining label assumptions.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: activity records with documented emission factors and units. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Agreement with reference estimates and sensitivity to factor assumptions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Carbon Emission Estimate Modelling\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a transparent factor-based calculation.\n# 4. Optional extension: compare regression while explaining label assumptions.\n# 5. Evaluate agreement with reference estimates and sensitivity to factor assumptions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-06",
        "title": "Water Quality Indicator Trend Analysis",
        "difficulty": "Beginner",
        "summary": "Explore and forecast one measured indicator in a monitoring time series. Start with a moving average or persistence forecast. When that works, compare regression and inspect seasonal patterns. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a moving average or persistence forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: a moving average or persistence forecast. Extension: compare regression and inspect seasonal patterns.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted water-monitoring data with units and station identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Forecast error on later periods; no drinking-water safety conclusion is implied. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Water Quality Indicator Trend Analysis\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a moving average or persistence forecast.\n# 4. Optional extension: compare regression and inspect seasonal patterns.\n# 5. Evaluate forecast error on later periods; no drinking-water safety conclusion is implied.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-07",
        "title": "Flood-Level Forecasting for a Monitoring Station",
        "difficulty": "Intermediate",
        "summary": "Predict a defined water-level horizon from historical rainfall and gauge readings. Start with a persistence forecast. When that works, compare a sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a persistence forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a persistence forecast. Extension: compare a sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: aligned permitted rainfall and gauge histories. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and errors during high-water events on future periods. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Flood-Level Forecasting for a Monitoring Station\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a persistence forecast.\n# 4. Optional extension: compare a sequence model.\n# 5. Evaluate MAE and errors during high-water events on future periods.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-08",
        "title": "Deforestation Mapping from Satellite Images",
        "difficulty": "Intermediate",
        "summary": "Classify forest-cover change within a small labelled study area. Start with a simple vegetation-feature classifier. When that works, compare image segmentation across dates. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a simple vegetation-feature classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a simple vegetation-feature classifier. Extension: compare image segmentation across dates.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted georeferenced imagery with checked forest and change labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Class-level accuracy and change errors on held-out regions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Deforestation Mapping from Satellite Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a simple vegetation-feature classifier.\n# 4. Optional extension: compare image segmentation across dates.\n# 5. Evaluate class-level accuracy and change errors on held-out regions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-09",
        "title": "Wildfire Susceptibility Modelling from Historical Features",
        "difficulty": "Intermediate",
        "summary": "Explore associations between environmental features and recorded fire occurrence. Start with logistic regression. When that works, compare a tree model with spatially separated evaluation. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: logistic regression. Extension: compare a tree model with spatially separated evaluation.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted historical fire labels and environmental features available before the event. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision-recall and geographic transfer; scores are not emergency warnings. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Wildfire Susceptibility Modelling from Historical Features\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with logistic regression.\n# 4. Optional extension: compare a tree model with spatially separated evaluation.\n# 5. Evaluate precision-recall and geographic transfer; scores are not emergency warnings.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-10",
        "title": "Urban Heat Pattern Mapping",
        "difficulty": "Intermediate",
        "summary": "Relate prepared land-surface temperature observations to land-cover features. Start with regression. When that works, compare a spatial model while controlling leakage. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: regression. Extension: compare a spatial model while controlling leakage.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted temperature and land-cover products with dates and resolution recorded. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Prediction error on held-out areas and sensitivity to data resolution. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Urban Heat Pattern Mapping\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with regression.\n# 4. Optional extension: compare a spatial model while controlling leakage.\n# 5. Evaluate prediction error on held-out areas and sensitivity to data resolution.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-11",
        "title": "Biodiversity Audio Classification",
        "difficulty": "Intermediate",
        "summary": "Recognise a small set of labelled species sounds in permitted recordings. Start with audio features and a conventional classifier. When that works, compare a spectrogram network. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on audio features and a conventional classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: audio features and a conventional classifier. Extension: compare a spectrogram network.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted labelled recordings with site and session identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on sites excluded from training. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Biodiversity Audio Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with audio features and a conventional classifier.\n# 4. Optional extension: compare a spectrogram network.\n# 5. Evaluate macro-F1 and errors on sites excluded from training.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-12",
        "title": "Renewable Energy Forecasting for Environmental Planning",
        "difficulty": "Intermediate",
        "summary": "Forecast recorded renewable output and examine uncertainty over a chosen horizon. Start with a persistence or seasonal-naive forecast. When that works, compare regression or a sequence model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a persistence or seasonal-naive forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a persistence or seasonal-naive forecast. Extension: compare regression or a sequence model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: aligned permitted generation and weather histories. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE and uncertainty coverage on future dates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Renewable Energy Forecasting for Environmental Planning\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a persistence or seasonal-naive forecast.\n# 4. Optional extension: compare regression or a sequence model.\n# 5. Evaluate MAE and uncertainty coverage on future dates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-13",
        "title": "Climate Time-Series Scenario Exploration",
        "difficulty": "Advanced",
        "summary": "Compare statistical forecasts for one climate indicator and discuss their limits. Start with a documented trend and seasonal model. When that works, compare a learned model and test sensitivity to the chosen period. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented trend and seasonal model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a documented trend and seasonal model. Extension: compare a learned model and test sensitivity to the chosen period.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: documented historical climate observations with units and source uncertainty. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Hindcast error and robustness across periods; this is not a substitute for physical climate projections. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Climate Time-Series Scenario Exploration\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented trend and seasonal model.\n# 4. Optional extension: compare a learned model and test sensitivity to the chosen period.\n# 5. Evaluate hindcast error and robustness across periods; this is not a substitute for physical climate projections.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-14",
        "title": "Glacier Area Change from Satellite Images",
        "difficulty": "Advanced",
        "summary": "Estimate changes in glacier extent for a small annotated region. Start with a documented image-threshold method. When that works, compare segmentation with careful geospatial alignment. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented image-threshold method for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a documented image-threshold method. Extension: compare segmentation with careful geospatial alignment.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted multi-date imagery with cloud masks and reference outlines. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Area error and sensitivity to snow, clouds, and boundary uncertainty. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Glacier Area Change from Satellite Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented image-threshold method.\n# 4. Optional extension: compare segmentation with careful geospatial alignment.\n# 5. Evaluate area error and sensitivity to snow, clouds, and boundary uncertainty.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-15",
        "title": "Ocean Litter Detection in Recorded Images",
        "difficulty": "Intermediate",
        "summary": "Detect a limited set of visible litter categories in permitted coastal or marine images. Start with a pretrained image detector. When that works, fine-tune on checked labels and test new scenes. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a pretrained image detector for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a pretrained image detector. Extension: fine-tune on checked labels and test new scenes.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted images with litter annotations and recording-source identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors under glare or occlusion. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Ocean Litter Detection in Recorded Images\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a pretrained image detector.\n# 4. Optional extension: fine-tune on checked labels and test new scenes.\n# 5. Evaluate precision, recall, and errors under glare or occlusion.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-16",
        "title": "Waste Collection Route Planning in Simulation",
        "difficulty": "Advanced",
        "summary": "Compare collection routes under a small, documented demand scenario. Start with a fixed route or nearest-neighbour heuristic. When that works, compare optimisation using uncertain fill-level forecasts. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a fixed route or nearest-neighbour heuristic for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a fixed route or nearest-neighbour heuristic. Extension: compare optimisation using uncertain fill-level forecasts.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic locations, distances, vehicle capacities, and bin-demand records. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Simulated distance, missed collections, and capacity violations. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Waste Collection Route Planning in Simulation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a fixed route or nearest-neighbour heuristic.\n# 4. Optional extension: compare optimisation using uncertain fill-level forecasts.\n# 5. Evaluate simulated distance, missed collections, and capacity violations.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-17",
        "title": "Environmental Risk Mapping with Uncertainty",
        "difficulty": "Advanced",
        "summary": "Build an exploratory map for one clearly defined historical environmental outcome. Start with a transparent statistical model. When that works, compare a spatial learner and evaluate uncertainty. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a transparent statistical model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a transparent statistical model. Extension: compare a spatial learner and evaluate uncertainty.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted outcome labels and environmental predictors with known spatial resolution. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Spatially held-out error, calibration, and sensitivity to incomplete data. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Environmental Risk Mapping with Uncertainty\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a transparent statistical model.\n# 4. Optional extension: compare a spatial learner and evaluate uncertainty.\n# 5. Evaluate spatially held-out error, calibration, and sensitivity to incomplete data.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-18",
        "title": "Sensor Network Anomaly Detection",
        "difficulty": "Intermediate",
        "summary": "Flag unusual environmental sensor readings for review. Start with range and rate-of-change rules. When that works, compare an anomaly detector on known test events. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on range and rate-of-change rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: range and rate-of-change rules. Extension: compare an anomaly detector on known test events.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or synthetic sensor streams with documented units and labelled faults. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: False alarms and detection of known sensor faults. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Sensor Network Anomaly Detection\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with range and rate-of-change rules.\n# 4. Optional extension: compare an anomaly detector on known test events.\n# 5. Evaluate false alarms and detection of known sensor faults.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-19",
        "title": "Multi-Hazard Scenario Comparison",
        "difficulty": "Advanced",
        "summary": "Explore how several separately modelled hazards overlap in a small study area. Start with a documented overlay with explicit weights. When that works, test sensitivity to weights and data uncertainty. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a documented overlay with explicit weights for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a documented overlay with explicit weights. Extension: test sensitivity to weights and data uncertainty.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted prepared hazard layers with dates, definitions, and spatial resolution. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Consistency, sensitivity, and uncertainty; the map is exploratory rather than an operational forecast. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Multi-Hazard Scenario Comparison\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a documented overlay with explicit weights.\n# 4. Optional extension: test sensitivity to weights and data uncertainty.\n# 5. Evaluate consistency, sensitivity, and uncertainty; the map is exploratory rather than an operational forecast.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "env-20",
        "title": "Sustainable Planning Trade-off Exploration",
        "difficulty": "Advanced",
        "summary": "Compare a small set of planning options across documented environmental objectives. Start with a transparent weighted score. When that works, compare multi-objective optimisation and inspect alternative rankings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a transparent weighted score for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a transparent weighted score. Extension: compare multi-objective optimisation and inspect alternative rankings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or permitted planning indicators with units and assumptions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Constraint satisfaction, trade-off stability, and sensitivity to chosen weights. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Sustainable Planning Trade-off Exploration\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a transparent weighted score.\n# 4. Optional extension: compare multi-objective optimisation and inspect alternative rankings.\n# 5. Evaluate constraint satisfaction, trade-off stability, and sensitivity to chosen weights.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  },
  "education": {
    "name": "Education & Learning Sciences",
    "short": "EDU",
    "icon": "\ud83d\udcda",
    "color": "#e11d48",
    "tagline": "Build learning tools that support teachers and students.",
    "description": "Educational judgement stays with teachers and learners. Begin with synthetic or approved de-identified data and evaluate usefulness without making consequential decisions about students. Start with resource tagging, FAQ search, or synthetic quiz data; progress to feedback evaluation; explore adaptive learning in simulation with teacher review. The level describes the starting scope: beginner projects use guided baselines, intermediate projects compare methods independently, and advanced projects need deeper domain and modelling skills.",
    "projects": [
      {
        "id": "edu-01",
        "title": "Study-Time and Quiz-Score Exploration",
        "difficulty": "Beginner",
        "summary": "Explore patterns between study records and later quiz scores in synthetic data. Start with simple summaries and linear regression. When that works, compare a small tree model while discussing confounding. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on simple summaries and linear regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: simple summaries and linear regression. Extension: compare a small tree model while discussing confounding.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or appropriately consented study records without identifying details. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on held-out learners; association does not establish causation. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Study-Time and Quiz-Score Exploration\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with simple summaries and linear regression.\n# 4. Optional extension: compare a small tree model while discussing confounding.\n# 5. Evaluate MAE on held-out learners; association does not establish causation.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-02",
        "title": "Course Feedback Sentiment Classification",
        "difficulty": "Beginner",
        "summary": "Classify anonymised feedback into a small set of sentiment labels. Start with TF-IDF and logistic regression. When that works, compare embeddings and review ambiguous comments. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on TF-IDF and logistic regression for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: TF-IDF and logistic regression. Extension: compare embeddings and review ambiguous comments.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: redacted or synthetic feedback with checked labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors across courses; avoid identifying individual students. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Course Feedback Sentiment Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with TF-IDF and logistic regression.\n# 4. Optional extension: compare embeddings and review ambiguous comments.\n# 5. Evaluate macro-F1 and errors across courses; avoid identifying individual students.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-03",
        "title": "Learning Resource Topic Classification",
        "difficulty": "Beginner",
        "summary": "Organise permitted learning-material descriptions into subject categories. Start with keyword rules. When that works, compare a text classifier. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: keyword rules. Extension: compare a text classifier.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted resource descriptions with checked topic labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 and errors on unfamiliar resources. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Learning Resource Topic Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword rules.\n# 4. Optional extension: compare a text classifier.\n# 5. Evaluate macro-F1 and errors on unfamiliar resources.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-04",
        "title": "Student FAQ Search Tool",
        "difficulty": "Beginner",
        "summary": "Retrieve relevant answers from an approved set of course FAQs. Start with keyword search. When that works, compare sentence embeddings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword search for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: keyword search. Extension: compare sentence embeddings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: approved FAQ text and manually judged sample queries. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Retrieval precision and recall at a chosen result count. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Student FAQ Search Tool\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword search.\n# 4. Optional extension: compare sentence embeddings.\n# 5. Evaluate retrieval precision and recall at a chosen result count.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-05",
        "title": "Synthetic Attendance Pattern Dashboard",
        "difficulty": "Beginner",
        "summary": "Create a dashboard that summarises synthetic attendance patterns over time. Start with transparent aggregations and trend rules. When that works, add a small aggregate attendance forecast. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on transparent aggregations and trend rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: transparent aggregations and trend rules. Extension: add a small aggregate attendance forecast.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic class-level attendance records without personal identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Correct summaries and forecast error on later periods. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Synthetic Attendance Pattern Dashboard\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with transparent aggregations and trend rules.\n# 4. Optional extension: add a small aggregate attendance forecast.\n# 5. Evaluate correct summaries and forecast error on later periods.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-06",
        "title": "Quiz Question Difficulty Estimation",
        "difficulty": "Beginner",
        "summary": "Estimate observed item difficulty from a prepared response table. Start with the proportion of correct responses. When that works, compare regression using permitted item features. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on the proportion of correct responses for this task? Keep your first version small enough to evaluate fairly. Helpful background: Basic Python and guided practice with tables, text, or images.",
        "ai_technique": "Starting approach: the proportion of correct responses. Extension: compare regression using permitted item features.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or de-identified item responses with a clearly defined target. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. Specialist hardware is not needed for the starting task. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Estimation error on held-out questions and uncertainty for rarely answered items. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Quiz Question Difficulty Estimation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with the proportion of correct responses.\n# 4. Optional extension: compare regression using permitted item features.\n# 5. Evaluate estimation error on held-out questions and uncertainty for rarely answered items.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-07",
        "title": "Learning Resource Recommendation",
        "difficulty": "Intermediate",
        "summary": "Recommend permitted study resources using anonymised interaction data. Start with a popularity list. When that works, compare content similarity or collaborative filtering. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a popularity list for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a popularity list. Extension: compare content similarity or collaborative filtering.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or consented resource interactions with anonymised identifiers. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision at a chosen list length, coverage, and cold-start behaviour. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Learning Resource Recommendation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a popularity list.\n# 4. Optional extension: compare content similarity or collaborative filtering.\n# 5. Evaluate precision at a chosen list length, coverage, and cold-start behaviour.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-08",
        "title": "Short-Answer Feedback Prototype",
        "difficulty": "Intermediate",
        "summary": "Compare a student's synthetic answer with a rubric and suggest evidence for review. Start with keyword or concept matching. When that works, compare embeddings without assigning final grades automatically. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword or concept matching for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: keyword or concept matching. Extension: compare embeddings without assigning final grades automatically.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic answers with teacher-checked rubric annotations. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Agreement with checked concepts and rates of misleading feedback. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Short-Answer Feedback Prototype\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword or concept matching.\n# 4. Optional extension: compare embeddings without assigning final grades automatically.\n# 5. Evaluate agreement with checked concepts and rates of misleading feedback.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-09",
        "title": "Course Engagement Trend Forecasting",
        "difficulty": "Intermediate",
        "summary": "Forecast aggregate learning-platform activity for a defined period. Start with a seasonal-naive forecast. When that works, compare a regression model. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a seasonal-naive forecast for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: a seasonal-naive forecast. Extension: compare a regression model.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or consented aggregate activity counts. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: MAE on future periods and sensitivity to schedule changes. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Course Engagement Trend Forecasting\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a seasonal-naive forecast.\n# 4. Optional extension: compare a regression model.\n# 5. Evaluate MAE on future periods and sensitivity to schedule changes.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-10",
        "title": "Educational Video Topic Segmentation",
        "difficulty": "Intermediate",
        "summary": "Split permitted lecture transcripts into a few topic sections. Start with keyword changes and fixed windows. When that works, compare text-embedding similarity. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on keyword changes and fixed windows for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: keyword changes and fixed windows. Extension: compare text-embedding similarity.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted transcripts with manually checked topic boundaries. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Boundary error and agreement with reference topic labels. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Educational Video Topic Segmentation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with keyword changes and fixed windows.\n# 4. Optional extension: compare text-embedding similarity.\n# 5. Evaluate boundary error and agreement with reference topic labels.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-11",
        "title": "Question Similarity and Duplicate Detection",
        "difficulty": "Intermediate",
        "summary": "Find question pairs that assess similar concepts. Start with TF-IDF similarity. When that works, compare sentence embeddings. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on TF-IDF similarity for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: TF-IDF similarity. Extension: compare sentence embeddings.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted or synthetic question pairs with teacher-reviewed labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and errors across subject areas. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Question Similarity and Duplicate Detection\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with TF-IDF similarity.\n# 4. Optional extension: compare sentence embeddings.\n# 5. Evaluate precision, recall, and errors across subject areas.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-12",
        "title": "Reading Resource Level Classification",
        "difficulty": "Intermediate",
        "summary": "Classify text resources into a documented set of reading-demand categories. Start with transparent text features and a classifier. When that works, compare embeddings and examine domain differences. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on transparent text features and a classifier for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: transparent text features and a classifier. Extension: compare embeddings and examine domain differences.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted passages with reviewed category labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Macro-F1 across text sources; labels should not be used to categorise a learner's ability. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Reading Resource Level Classification\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with transparent text features and a classifier.\n# 4. Optional extension: compare embeddings and examine domain differences.\n# 5. Evaluate macro-F1 across text sources; labels should not be used to categorise a learner's ability.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-13",
        "title": "Knowledge Tracing on Synthetic Learning Sequences",
        "difficulty": "Advanced",
        "summary": "Study how a model predicts the next response in a synthetic learning sequence. Start with a simple statistical learner model. When that works, compare a sequence model with learner-separated testing. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a simple statistical learner model for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a simple statistical learner model. Extension: compare a sequence model with learner-separated testing.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic item-response sequences with concept labels. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Log loss, calibration, and generalisation to held-out learners. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Knowledge Tracing on Synthetic Learning Sequences\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a simple statistical learner model.\n# 4. Optional extension: compare a sequence model with learner-separated testing.\n# 5. Evaluate log loss, calibration, and generalisation to held-out learners.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-14",
        "title": "Evaluating an AI Tutor's Grounded Explanations",
        "difficulty": "Advanced",
        "summary": "Test explanations against a fixed set of approved lesson materials. Start with retrieval with source excerpts. When that works, compare generated explanations using a teacher-reviewed rubric. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on retrieval with source excerpts for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: retrieval with source excerpts. Extension: compare generated explanations using a teacher-reviewed rubric.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted lessons and checked questions with reference explanations. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Factual support, clarity, omissions, and appropriate uncertainty. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Evaluating an AI Tutor's Grounded Explanations\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with retrieval with source excerpts.\n# 4. Optional extension: compare generated explanations using a teacher-reviewed rubric.\n# 5. Evaluate factual support, clarity, omissions, and appropriate uncertainty.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-15",
        "title": "Adaptive Quiz Selection in Simulation",
        "difficulty": "Advanced",
        "summary": "Compare ways to choose the next question in a simulated learning task. Start with a fixed question order. When that works, compare a bandit or simple adaptive strategy. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a fixed question order for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a fixed question order. Extension: compare a bandit or simple adaptive strategy.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic learner responses and documented simulator assumptions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Simulated learning proxy, question coverage, and sensitivity to learner assumptions. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Adaptive Quiz Selection in Simulation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a fixed question order.\n# 4. Optional extension: compare a bandit or simple adaptive strategy.\n# 5. Evaluate simulated learning proxy, question coverage, and sensitivity to learner assumptions.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-16",
        "title": "Fairness Audit of an Educational Prediction Model",
        "difficulty": "Advanced",
        "summary": "Investigate how a fixed research model performs across permitted, adequately represented groups. Start with a transparent baseline with group-level error summaries. When that works, compare calibration or error-mitigation approaches on a separate validation set. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on a transparent baseline with group-level error summaries for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: a transparent baseline with group-level error summaries. Extension: compare calibration or error-mitigation approaches on a separate validation set.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or approved de-identified data with justified group definitions. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Group-level error and uncertainty; results must not determine admissions or access. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Fairness Audit of an Educational Prediction Model\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with a transparent baseline with group-level error summaries.\n# 4. Optional extension: compare calibration or error-mitigation approaches on a separate validation set.\n# 5. Evaluate group-level error and uncertainty; results must not determine admissions or access.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-17",
        "title": "Rubric-Based Essay Feedback Evaluation",
        "difficulty": "Advanced",
        "summary": "Evaluate whether a prototype identifies rubric-relevant evidence in synthetic essays. Start with transparent feature rules. When that works, compare a language model with teacher review and source-supported feedback. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on transparent feature rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: transparent feature rules. Extension: compare a language model with teacher review and source-supported feedback.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or permitted essays with checked rubric annotations. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Agreement with rubric evidence and harmful or unsupported feedback rates. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Rubric-Based Essay Feedback Evaluation\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with transparent feature rules.\n# 4. Optional extension: compare a language model with teacher review and source-supported feedback.\n# 5. Evaluate agreement with rubric evidence and harmful or unsupported feedback rates.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-18",
        "title": "Accessible Learning Material Tagging",
        "difficulty": "Intermediate",
        "summary": "Tag permitted resource descriptions by available formats and accessibility features. Start with explicit metadata rules. When that works, compare a text classifier and flag uncertain records. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on explicit metadata rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: explicit metadata rules. Extension: compare a text classifier and flag uncertain records.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic or approved resource metadata with verified tags. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Tag precision and missed features; tags do not establish full accessibility compliance. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Accessible Learning Material Tagging\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with explicit metadata rules.\n# 4. Optional extension: compare a text classifier and flag uncertain records.\n# 5. Evaluate tag precision and missed features; tags do not establish full accessibility compliance.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-19",
        "title": "Teacher-Reviewed Question Generation Study",
        "difficulty": "Advanced",
        "summary": "Generate draft questions from approved learning materials and evaluate their quality. Start with manually written template questions. When that works, compare a language model with teacher review before use. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on manually written template questions for this task? Keep your first version small enough to evaluate fairly. Helpful background: Independent model evaluation, relevant subject knowledge, and experience with the starting method.",
        "ai_technique": "Starting approach: manually written template questions. Extension: compare a language model with teacher review before use.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: permitted lessons and a rubric for relevance, answerability, and factual support. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Rubric agreement, unsupported answers, and duplicate-question rate. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Teacher-Reviewed Question Generation Study\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with manually written template questions.\n# 4. Optional extension: compare a language model with teacher review before use.\n# 5. Evaluate rubric agreement, unsupported answers, and duplicate-question rate.\n# 6. Save the notebook, settings, results, and limitations."
      },
      {
        "id": "edu-20",
        "title": "Learning Analytics Data Quality Monitor",
        "difficulty": "Intermediate",
        "summary": "Detect missing, inconsistent, or duplicate events in synthetic learning-platform logs. Start with documented validation rules. When that works, compare anomaly detection on intentionally inserted errors. Finish with a notebook, a comparison of results, and a short explanation of what you learned.",
        "problem": "Your question: can a more detailed approach improve on documented validation rules for this task? Keep your first version small enough to evaluate fairly. Helpful background: Comfort with data preparation, a working baseline, and train/validation/test splits.",
        "ai_technique": "Starting approach: documented validation rules. Extension: compare anomaly detection on intentionally inserted errors.",
        "tech_stack": [
          "Python",
          "Pandas / NumPy",
          "Scikit-learn",
          "Jupyter / Colab",
          "Task-specific libraries as needed"
        ],
        "data_pipeline": "Suggested data: synthetic event logs with known injected issues. Confirm access, permissions, units, and labels before choosing your final scope. Fit preprocessing on training data only and keep related records together when splitting. Data is not bundled with this description.",
        "hardware_target": "Start with a small dataset in a Python notebook on your laptop or Colab. A GPU may help with a neural extension; check the workload before scaling up. The tool list is illustrative. Device deployment and live-system use are outside the starting scope.",
        "impact": "What you can show: Precision, recall, and review burden for the known data-quality errors. Report your measured results, including cases where the baseline works better. These are learning goals, not promised performance.",
        "code_snippet": "# Project planning outline \u2014 not an executable implementation.\n# Learning Analytics Data Quality Monitor\n# 1. Define the target and check the data.\n# 2. Reserve a final test set before fitting any preprocessing.\n# 3. Start with documented validation rules.\n# 4. Optional extension: compare anomaly detection on intentionally inserted errors.\n# 5. Evaluate precision, recall, and review burden for the known data-quality errors.\n# 6. Save the notebook, settings, results, and limitations."
      }
    ]
  }
};

    const deptMetadata = {
      eee: {
        name: "Electrical & Electronics Engineering (EEE)",
        shortName: "Electrical (EEE)",
        icon: "⚡",
        color: "#3b82f6",
        bg: "rgba(59, 130, 246, 0.2)",
        glow: "rgba(59, 130, 246, 0.45)",
        synergy: "Energy forecasting, recorded-signal analysis, and simulated control."
      },
      ece: {
        name: "Electronics & Communication Engineering (ECE)",
        shortName: "Telecom & ECE",
        icon: "📡",
        color: "#0284c7",
        bg: "rgba(2, 132, 199, 0.2)",
        glow: "rgba(2, 132, 199, 0.45)",
        synergy: "Audio activity, PCB inspection, and wireless-signal experiments."
      },
      civil: {
        name: "Civil & Structural Engineering",
        shortName: "Civil Engineering",
        icon: "🏗️",
        color: "#0ea5e9",
        bg: "rgba(14, 165, 233, 0.2)",
        glow: "rgba(14, 165, 233, 0.45)",
        synergy: "Concrete strength, inspection images, and traffic simulations."
      },
      agriculture: {
        name: "Agriculture & Smart Farming",
        shortName: "Smart Agriculture",
        icon: "🌱",
        color: "#2563eb",
        bg: "rgba(37, 99, 235, 0.2)",
        glow: "rgba(37, 99, 235, 0.45)",
        synergy: "Moisture forecasting, crop images, and yield analysis."
      },
      business: {
        name: "Business, Finance & Management",
        shortName: "FinTech & Business",
        icon: "📈",
        color: "#6366f1",
        bg: "rgba(99, 102, 241, 0.2)",
        glow: "rgba(99, 102, 241, 0.45)",
        synergy: "Customer churn, demand forecasting, and document analysis."
      },
      healthcare: {
        name: "Healthcare & Biomedical Sciences",
        shortName: "Healthcare & Bio",
        icon: "🏥",
        color: "#38bdf8",
        bg: "rgba(56, 189, 248, 0.2)",
        glow: "rgba(56, 189, 248, 0.45)",
        synergy: "ECG learning, retinal images, and research-model evaluation."
      },
  "cs": {
    "name": "Computer Science & Software Systems",
    "shortName": "Computer Science",
    "icon": "\ud83d\udcbb",
    "color": "#7c3aed",
    "bg": "rgba(124, 58, 237, 0.2)",
    "glow": "rgba(124, 58, 237, 0.45)",
    "synergy": "Testable Python tools, search, model evaluation, and small AI applications."
  },
  "mechanical": {
    "name": "Mechanical Engineering",
    "shortName": "Mechanical Engineering",
    "icon": "\u2699\ufe0f",
    "color": "#64748b",
    "bg": "rgba(100, 116, 139, 0.2)",
    "glow": "rgba(100, 116, 139, 0.45)",
    "synergy": "Machine measurements, manufacturing images, and simulated design experiments."
  },
  "environment": {
    "name": "Environmental Science",
    "shortName": "Environmental Science",
    "icon": "\ud83c\udf0d",
    "color": "#059669",
    "bg": "rgba(5, 150, 105, 0.2)",
    "glow": "rgba(5, 150, 105, 0.45)",
    "synergy": "Environmental records, satellite images, forecasting, and uncertainty."
  },
  "education": {
    "name": "Education & Learning Sciences",
    "shortName": "Education",
    "icon": "\ud83d\udcda",
    "color": "#e11d48",
    "bg": "rgba(225, 29, 72, 0.2)",
    "glow": "rgba(225, 29, 72, 0.45)",
    "synergy": "Resource search, synthetic learning analytics, and teacher-reviewed AI feedback."
  }
    };

    // WHY COURSES ARE ESSENTIAL FOR EACH DISCIPLINE
    const essentialDisciplineData = {
      eee: {
        name: "Electrical & Electronics Engineering (EEE)",
        icon: "⚡",
        deptKey: "eee",
        shiftSummary: "Electrical principles remain the foundation. AI adds ways to explore recorded signals, compare forecasts, and investigate simulated control problems. Start with household appliance activity or thermal-image exploration; move to solar forecasting and motor signals; attempt grid stability or microgrid control after studying the prerequisites.",
        python: {
          title: "Python: Start with the Data",
          desc: "Plot recorded measurements, clean time series, and automate small calculations.",
          keySkills: "Signal plots, timestamps, reproducible notebooks."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Compare simple models for energy use, solar output, or battery measurements.",
          keySkills: "Regression, held-out time periods, baseline comparisons."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Explore sequence models, image analysis, and simulated control after building a baseline.",
          keySkills: "Neural training, simulation checks, uncertainty."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
      ece: {
        name: "Electronics & Communication Engineering (ECE / Telecom)",
        icon: "📡",
        deptKey: "ece",
        shiftSummary: "Signal processing and communication theory help you understand what a model is learning. Use AI as an approach to compare with conventional estimators and detectors. Start with voice activity detection; progress to radio-signal classification or PCB images; try MIMO estimation and learned equalisation once you know the signal theory.",
        python: {
          title: "Python: Start with the Data",
          desc: "Load recorded audio or radio samples and visualise their patterns.",
          keySkills: "Arrays, sampling, signal visualisation."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Build and evaluate small classifiers for signal activity or device labels.",
          keySkills: "Feature extraction, class-level metrics, session-based splits."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Compare neural estimators and sequence models in documented simulations.",
          keySkills: "Neural estimators, simulated channels, robustness."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
      civil: {
        name: "Civil & Structural Engineering",
        icon: "🏗️",
        deptKey: "civil",
        shiftSummary: "Engineering judgement, inspection, and physical models remain essential. Student AI projects can help explore recorded measurements and image patterns within a defined scope. Start with concrete strength, road images, or visible safety equipment; progress to crack or flood analysis; explore graph learning and traffic control as advanced projects.",
        python: {
          title: "Python: Start with the Data",
          desc: "Clean material-test tables, organise sensor records, and plot results.",
          keySkills: "Tables, units, plots, and data quality."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Compare models for measured strength or historical water-level data.",
          keySkills: "Regression, meaningful errors, site-aware validation."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Explore crack segmentation and structural signals with carefully separated test data.",
          keySkills: "Segmentation, time series, physical interpretation."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
      agriculture: {
        name: "Agriculture & Smart Farming",
        icon: "🌱",
        deptKey: "agriculture",
        shiftSummary: "Crop knowledge and field observations give meaning to agricultural data. Models can help you explore patterns, but results from one dataset may not transfer to another farm or season. Start with soil moisture, leaf images, or grain appearance; progress to yield and hive audio; try simulated robotics or greenhouse control when ready.",
        python: {
          title: "Python: Start with the Data",
          desc: "Work with weather tables, sensor readings, and prepared image data.",
          keySkills: "Sensor records, missing readings, time-series plots."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Compare moisture forecasts and yield estimates with simple baselines.",
          keySkills: "Forecasting, season-based splits, feature comparisons."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Explore crop images and simulated robotic tasks with field-aware evaluation.",
          keySkills: "Image classification, segmentation, simulation."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
      business: {
        name: "Business, Finance & Quantitative Management",
        icon: "📈",
        deptKey: "business",
        shiftSummary: "Business knowledge helps define useful targets and costs. Spreadsheets, statistics, and simple rules remain useful baselines when you evaluate a more complex model. Start with churn prediction or invoice fields; progress to demand, reviews, and recommendations; attempt document assistants or simulated pricing after learning evaluation methods.",
        python: {
          title: "Python: Start with the Data",
          desc: "Clean tables, work with dates, and produce repeatable analyses.",
          keySkills: "Data cleaning, dates, summaries, notebooks."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Build models for churn, demand, or suspicious historical transactions.",
          keySkills: "Classification, forecasting, leakage prevention."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Explore text retrieval and recommendation models when a simpler approach is insufficient.",
          keySkills: "Text evaluation, embeddings, recommendation metrics."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
      healthcare: {
        name: "Healthcare, Biomedical & Life Sciences",
        icon: "🏥",
        deptKey: "healthcare",
        shiftSummary: "Healthcare AI learning starts with approved data, well-defined labels, and careful evaluation. Models support research questions; a student result does not establish clinical usefulness. Start with prepared ECG features; move to retinal images or recorded instrument detection; attempt segmentation, federated learning, or retrospective risk models with suitable supervision.",
        python: {
          title: "Python: Start with the Data",
          desc: "Plot de-identified signals and check units, timestamps, and missing data.",
          keySkills: "Signal plots, de-identification, dataset documentation."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Compare research-label predictions with transparent statistical baselines.",
          keySkills: "Patient-level splits, calibration, class-level error."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Study imaging and sequence models while separating patients between data splits.",
          keySkills: "Segmentation, bias analysis, research reproducibility."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
      cs: {
        name: "Computer Science & Software Systems",
        icon: "💻",
        deptKey: "cs",
        shiftSummary: "Reliable AI software still depends on sound programming, testing, and data handling. Start with a clear user task and add a model only where it helps. Start with Python and a small tabular model; progress to APIs and retrieval; explore transformers and model monitoring after learning how to evaluate outputs.",
        python: {
          title: "Python: Start with the Data",
          desc: "Write readable functions, organise projects, and test small components.",
          keySkills: "Functions, tests, APIs, and versioned environments."
        },
        ml: {
          title: "Machine Learning: Build and Compare",
          desc: "Compare ranking, classification, or anomaly-detection models with clear baselines.",
          keySkills: "Train/test splits, metrics, reproducible inference."
        },
        dl: {
          title: "Deep Learning: Extend When Ready",
          desc: "Explore neural retrieval and language-model outputs with fixed evaluation examples.",
          keySkills: "Retrieval, output checks, model limitations."
        },
        roiMetric: "Your evidence of progress: a runnable notebook, a fair baseline comparison, and an explanation of errors and limitations. Outcomes depend on your data and experiments."
      },
  "mechanical": {
    "name": "Mechanical Engineering",
    "icon": "\u2699\ufe0f",
    "deptKey": "mechanical",
    "shiftSummary": "Mechanical principles and measurement quality remain the foundation. These projects use recorded data or simulation to investigate a manageable engineering question. Start with temperature forecasts or part images; progress to tool wear and vibration; explore simulated control and design optimisation after building the prerequisites.",
    "python": {
      "title": "Python: Start with the Data",
      "desc": "Load a small permitted dataset, check its units and labels, and create plots that help you understand the question.",
      "keySkills": "Data cleaning, plots, reusable functions, and reproducible notebooks."
    },
    "ml": {
      "title": "Machine Learning: Build and Compare",
      "desc": "Try a simple baseline, compare one improvement, and test on examples that were not used to choose your model.",
      "keySkills": "Feature preparation, appropriate splits, meaningful metrics, and error analysis."
    },
    "dl": {
      "title": "Deep Learning: Extend When Ready",
      "desc": "Consider a neural method when the task, labels, and computing budget support it. Compare its results with the simpler baseline.",
      "keySkills": "Neural training, validation curves, robustness checks, and clear limitations."
    },
    "roiMetric": "Your evidence of progress: a runnable notebook, a fair comparison, and a clear explanation of results. Benefits depend on the dataset and the experiment."
  },
  "environment": {
    "name": "Environmental Science",
    "icon": "\ud83c\udf0d",
    "deptKey": "environment",
    "shiftSummary": "Environmental models depend on measurement quality, time, and location. Explore the evidence and its uncertainty before drawing conclusions beyond the study area. Start with pollution records, waste images, or rainfall; progress to satellite mapping; explore spatial uncertainty and planning trade-offs with suitable background.",
    "python": {
      "title": "Python: Start with the Data",
      "desc": "Load a small permitted dataset, check its units and labels, and create plots that help you understand the question.",
      "keySkills": "Data cleaning, plots, reusable functions, and reproducible notebooks."
    },
    "ml": {
      "title": "Machine Learning: Build and Compare",
      "desc": "Try a simple baseline, compare one improvement, and test on examples that were not used to choose your model.",
      "keySkills": "Feature preparation, appropriate splits, meaningful metrics, and error analysis."
    },
    "dl": {
      "title": "Deep Learning: Extend When Ready",
      "desc": "Consider a neural method when the task, labels, and computing budget support it. Compare its results with the simpler baseline.",
      "keySkills": "Neural training, validation curves, robustness checks, and clear limitations."
    },
    "roiMetric": "Your evidence of progress: a runnable notebook, a fair comparison, and a clear explanation of results. Benefits depend on the dataset and the experiment."
  },
  "education": {
    "name": "Education & Learning Sciences",
    "icon": "\ud83d\udcda",
    "deptKey": "education",
    "shiftSummary": "Educational judgement stays with teachers and learners. Begin with synthetic or approved de-identified data and evaluate usefulness without making consequential decisions about students. Start with resource tagging, FAQ search, or synthetic quiz data; progress to feedback evaluation; explore adaptive learning in simulation with teacher review.",
    "python": {
      "title": "Python: Start with the Data",
      "desc": "Load a small permitted dataset, check its units and labels, and create plots that help you understand the question.",
      "keySkills": "Data cleaning, plots, reusable functions, and reproducible notebooks."
    },
    "ml": {
      "title": "Machine Learning: Build and Compare",
      "desc": "Try a simple baseline, compare one improvement, and test on examples that were not used to choose your model.",
      "keySkills": "Feature preparation, appropriate splits, meaningful metrics, and error analysis."
    },
    "dl": {
      "title": "Deep Learning: Extend When Ready",
      "desc": "Consider a neural method when the task, labels, and computing budget support it. Compare its results with the simpler baseline.",
      "keySkills": "Neural training, validation curves, robustness checks, and clear limitations."
    },
    "roiMetric": "Your evidence of progress: a runnable notebook, a fair comparison, and a clear explanation of results. Benefits depend on the dataset and the experiment."
  }
    };

    // 3 FLAGSHIP COURSES DATA
    const coursesData = {
      python: {
        id: "python",
        title: "Python for AI & Scientific Computing",
        short: "Python for AI",
        icon: "🐍",
        color: "#38bdf8",
        glow: "rgba(56, 189, 248, 0.4)",
        price: "$29",
        badge: "Beginner foundations, then practical extensions",
        freeDesc: "New to programming? Start with variables, conditions, loops, and functions. Build towards data analysis with NumPy and Pandas, then return to advanced Python topics when you need them.",
        paidDesc: "Explore practical Python project work after the basics. Check the selected course details for available notebooks, review support, and access terms before enrolling.",
        // playlistUrl: "https:www.youtube.com/watch?v=kqtD5dpn9C8&list=PLKnIA16_Rmvb1m_BqV_eE9dJ3pW3_j_z1",
        // embedUrl: "https:www.youtube.com/embed/kqtD5dpn9C8",
        playlistName: "Python for Data Science & AI Foundations",
        freeFeatures: [
          "Beginner Python: variables, conditions, loops, and functions",
          "Organise your work with reusable functions and simple classes",
          "Use NumPy arrays and understand basic matrix operations",
          "Clean and explore tables with Pandas",
          "Explain your findings with clear plots",
          "Practise debugging and simple problem solving"
        ],
        paidFeatures: [
          "Practical project topics for applying your Python skills",
          "Data collection and cleaning exercises with permitted sources",
          "Optional performance profiling after your code works correctly",
          "Testing, type hints, and project organisation practice",
          "Check the course details for review and support arrangements",
          "Check completion requirements and any certificate terms before enrolling"
        ],
        modules: [
          { num: "MODULE 01", title: "Python Basics: Your First Working Program", desc: "Practise variables, data types, conditions, loops, and functions. Finish with a small script you can explain and debug.", tier: "free" },
          { num: "MODULE 02", title: "Reusable Code and Simple Classes", desc: "Organise repeated work into functions and classes. Explore inheritance and special methods after the basic design is clear.", tier: "free" },
          { num: "MODULE 03", title: "NumPy: Working with Numerical Data", desc: "Explore arrays, shapes, indexing, and matrix operations. Compare loops with array operations on small examples.", tier: "free" },
          { num: "MODULE 04", title: "Pandas: Cleaning and Exploring Tables", desc: "Read a table, check missing values, group records, and create plots. Add time-based operations when your project needs them.", tier: "free" },
          { num: "MODULE 05", title: "Practical Projects and Testing", desc: "Build a small data workflow, record dependencies, and add meaningful tests. Extend the project only after the first version works.", tier: "paid" },
          { num: "MODULE 06", title: "Advanced Python Extensions", desc: "Explore profiling, asynchronous work, packaging, or containers when there is a clear need. You do not need all of these for your first capstone.", tier: "paid" }
        ]
      },
      ml: {
        id: "ml",
        title: "Machine Learning: From Your First Model to Capstone Practice",
        short: "Machine Learning",
        icon: "🤖",
        color: "#2563eb",
        glow: "rgba(37, 99, 235, 0.4)",
        price: "$49",
        badge: "Learn with the CampusX 100 Days of ML series",
        freeDesc: "Use the free CampusX video series alongside this suggested learning path. Start with one clear prediction task, then learn how to prepare data, compare models, and interpret errors. The steps below organise your learning; they are not a verified day-by-day map of the playlist.",
        paidDesc: "Explore 200 capstone project outlines across 10 disciplines, from guided beginner baselines to advanced investigations. Choose one manageable scope, keep an experiment log, and build evidence you can explain. Check which teaching resources and support are available for your selected track.",
        // playlistUrl: "https:www.youtube.com/watch?v=ZftI2fEz0Fw&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH",
        // embedUrl: "https:www.youtube.com/embed/ZftI2fEz0Fw?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH",
        playlistName: "100 Days of Machine Learning by CampusX",
        freeFeatures: [
          "Explore the CampusX 100 Days of Machine Learning videos",
          "Understand loss, model fitting, and why a baseline matters",
          "Prepare data with transformations fitted on training data only",
          "Start with regression and classification before exploring ensembles",
          "Explore clustering and dimensionality reduction when useful",
          "Use suitable metrics, cross-validation, and a separate final test set"
        ],
        paidFeatures: [
          "200 project outlines across 10 disciplines, with beginner, intermediate, and advanced scopes",
          "Project-specific data suggestions; check access and permitted use",
          "Optional dashboards and APIs after the model has been evaluated",
          "Use a documented notebook to record your experiments and results",
          "Build evidence for your portfolio; confirm any review or certificate offering",
          "Check available learner-support channels and response arrangements"
        ],
        modules: [
          { num: "STEP 01", title: "Start with a Clear Question", desc: "Choose a target, identify who the work is for, and define a simple baseline. Learn the difference between regression, classification, and clustering.", tier: "free" },
          { num: "STEP 02", title: "Get to Know Your Data", desc: "Load permitted data, check units and labels, inspect missing values, and make a few useful plots.", tier: "free" },
          { num: "STEP 03", title: "Prepare Features without Leakage", desc: "Separate training and test data. Learn scaling, encoding, and pipelines fitted using training data only.", tier: "free" },
          { num: "STEP 04", title: "Handle Missing Values and Unusual Records", desc: "Compare simple data-cleaning choices. Record what you changed and why, rather than automatically deleting difficult examples.", tier: "free" },
          { num: "STEP 05", title: "Build and Understand a Regression Model", desc: "Start with linear regression, inspect errors, and explore regularisation or dimensionality reduction when relevant.", tier: "free" },
          { num: "STEP 06", title: "Compare Classifiers and Ensembles", desc: "Try a simple classifier before trees and boosting. Choose metrics that reflect the costs of different mistakes.", tier: "free" },
          { num: "STEP 07", title: "Explore Clusters and Anomalies", desc: "Study grouping methods and their assumptions. An unusual point is a starting point for investigation, not proof of a problem.", tier: "free" },
          { num: "STEP 08", title: "Make Your Work Reproducible", desc: "Save preprocessing and model settings, rerun the notebook, and test on held-out data. A small demo is enough before exploring deployment.", tier: "free" },
          { num: "CAPSTONE PRACTICE", title: "Choose from 200 Capstone Project Outlines", desc: "Explore 20 projects in each of 10 disciplines. Pick a level that matches your background, complete a starting baseline, compare one improvement, and present evidence for your conclusions.", tier: "paid" }
        ]
      },
      dl: {
        id: "dl",
        title: "Deep Learning, Computer Vision & Transformers",
        short: "Deep Learning",
        icon: "🧠",
        color: "#8b5cf6",
        glow: "rgba(139, 92, 246, 0.4)",
        price: "$59",
        badge: "Build on Python and Machine Learning foundations",
        freeDesc: "Start with small neural networks and learn what happens during training. Once you can evaluate a baseline, explore images, sequences, and attention models. Some mathematical and programming preparation will help.",
        paidDesc: "Explore practical neural-network experiments with a manageable dataset. Compare against simpler models before considering larger architectures, optimisation, or device deployment. GPU availability and course resources should be checked in advance.",
        // playlistUrl: "https:www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
        // embedUrl: "https:www.youtube.com/embed/aircAruvnKk",
        playlistName: "Neural Networks & Deep Learning Foundations",
        freeFeatures: [
          "Understand neurons, activations, and small feed-forward networks",
          "Learn how gradients and loss guide model training",
          "Practise with tensors, automatic differentiation, and small batches",
          "Explore image classification before advanced object detection",
          "Compare sequence models with simple time-series baselines",
          "Build intuition for attention before larger transformer experiments"
        ],
        paidFeatures: [
          "Notebook-based experiments; check dataset and GPU availability",
          "Vision tasks with labelled test sets and error analysis",
          "Optional transformer experiments after the baseline works",
          "Measure speed and prediction changes when optimising models",
          "Device deployment as an optional supervised extension",
          "Check course completion and certificate details before enrolling"
        ],
        modules: [
          { num: "MODULE 01", title: "Your First Small Neural Network", desc: "Build intuition for layers, activations, and forward passes. Train on a small dataset and compare with a simpler model.", tier: "free" },
          { num: "MODULE 02", title: "Understand Training and Overfitting", desc: "Explore gradients, loss, optimisers, and regularisation. Use training and validation curves to guide your decisions.", tier: "free" },
          { num: "MODULE 03", title: "Learn from Images", desc: "Begin with transfer learning for classification. Explore detection or segmentation after you have suitable labels and an evaluation plan.", tier: "free" },
          { num: "MODULE 04", title: "Learn from Sequences", desc: "Study RNNs, LSTMs, and related models. Compare with simple forecasts and keep future data out of training.", tier: "free" },
          { num: "MODULE 05", title: "Explore Attention and Transformers", desc: "Learn attention and positional information using small examples. Try a pretrained model when your task and computing budget justify it.", tier: "paid" },
          { num: "MODULE 06", title: "Evaluate and Share Your Model", desc: "Record your environment and results. Treat export, quantisation, and edge deployment as optional advanced work, and measure their trade-offs.", tier: "paid" }
        ]
      }
    };

    // State Variables
    const heroDisciplineWhyAI = {
      eee: {
        deptKey: 'eee',
        name: 'Electrical & Electronic Engineering (EEE)',
        icon: '⚡',
        tagline: 'Explore energy data, then build towards electrical-system AI.',
        classicalLimit: "Electrical principles remain the foundation. AI adds ways to explore recorded signals, compare forecasts, and investigate simulated control problems.",
        aiBreakthrough: "Start with household appliance activity or thermal-image exploration; move to solar forecasting and motor signals; attempt grid stability or microgrid control after studying the prerequisites.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Exploring Power Grid Stability with Physics-Informed AI",
        blueprintIndex: 0
      },
      ece: {
        deptKey: 'ece',
        name: 'Electronics & Communication Engineering (ECE / Telecom)',
        icon: '📡',
        tagline: 'Start with signals you can see, hear, and measure.',
        classicalLimit: "Signal processing and communication theory help you understand what a model is learning. Use AI as an approach to compare with conventional estimators and detectors.",
        aiBreakthrough: "Start with voice activity detection; progress to radio-signal classification or PCB images; try MIMO estimation and learned equalisation once you know the signal theory.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Learning-Based Channel Estimation in a MIMO Simulation",
        blueprintIndex: 0
      },
      civil: {
        deptKey: 'civil',
        name: 'Civil & Structural Engineering',
        icon: '🏗️',
        tagline: 'Connect materials and infrastructure questions with data.',
        classicalLimit: "Engineering judgement, inspection, and physical models remain essential. Student AI projects can help explore recorded measurements and image patterns within a defined scope.",
        aiBreakthrough: "Start with concrete strength, road images, or visible safety equipment; progress to crack or flood analysis; explore graph learning and traffic control as advanced projects.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Detecting Concrete Cracks in Images",
        blueprintIndex: 0
      },
      agriculture: {
        deptKey: 'agriculture',
        name: 'Smart Agriculture & Biosystems',
        icon: '🌱',
        tagline: 'Use familiar farming questions to begin learning AI.',
        classicalLimit: "Crop knowledge and field observations give meaning to agricultural data. Models can help you explore patterns, but results from one dataset may not transfer to another farm or season.",
        aiBreakthrough: "Start with soil moisture, leaf images, or grain appearance; progress to yield and hive audio; try simulated robotics or greenhouse control when ready.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Exploring Crop Stress in Multispectral Images",
        blueprintIndex: 0
      },
      business: {
        deptKey: 'business',
        name: 'FinTech, Business & Quantitative Management',
        icon: '📈',
        tagline: 'Turn a business question into a measurable experiment.',
        classicalLimit: "Business knowledge helps define useful targets and costs. Spreadsheets, statistics, and simple rules remain useful baselines when you evaluate a more complex model.",
        aiBreakthrough: "Start with churn prediction or invoice fields; progress to demand, reviews, and recommendations; attempt document assistants or simulated pricing after learning evaluation methods.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Fraud Detection in Historical Transaction Data",
        blueprintIndex: 0
      },
      healthcare: {
        deptKey: 'healthcare',
        name: 'Healthcare, Biomedical & Life Sciences',
        icon: '🏥',
        tagline: 'Learn from health research data with care and curiosity.',
        classicalLimit: "Healthcare AI learning starts with approved data, well-defined labels, and careful evaluation. Models support research questions; a student result does not establish clinical usefulness.",
        aiBreakthrough: "Start with prepared ECG features; move to retinal images or recorded instrument detection; attempt segmentation, federated learning, or retrospective risk models with suitable supervision.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Retinal Image Classification for a Research Prototype",
        blueprintIndex: 0
      },
      cs: {
        deptKey: 'cs',
        name: 'Computer Science & Software Systems',
        icon: '💻',
        tagline: 'Build software that you can test, understand, and improve.',
        classicalLimit: "Reliable AI software still depends on sound programming, testing, and data handling. Start with a clear user task and add a model only where it helps.",
        aiBreakthrough: "Start with Python and a small tabular model; progress to APIs and retrieval; explore transformers and model monitoring after learning how to evaluate outputs.",
        careerWhy: "Use your project to show how you frame a problem, work with data, test an idea, and explain your decisions. These are useful skills to discuss in coursework, research, and interviews; a project does not guarantee a job or qualification.",
        blueprint: "Email Spam Classification",
        blueprintIndex: 0
      },
  "mechanical": {
    "deptKey": "mechanical",
    "name": "Mechanical Engineering",
    "icon": "\u2699\ufe0f",
    "tagline": "Connect machines, measurements, and models.",
    "classicalLimit": "Mechanical principles and measurement quality remain the foundation. These projects use recorded data or simulation to investigate a manageable engineering question.",
    "aiBreakthrough": "Start with temperature forecasts or part images; progress to tool wear and vibration; explore simulated control and design optimisation after building the prerequisites.",
    "careerWhy": "Use your work to demonstrate problem framing, data preparation, evaluation, and clear communication. A student project can support a portfolio, but does not guarantee employment or professional qualification.",
    "blueprint": "Machine Temperature Forecasting",
    "blueprintIndex": 0
  },
  "environment": {
    "deptKey": "environment",
    "name": "Environmental Science",
    "icon": "\ud83c\udf0d",
    "tagline": "Use data to explore environmental questions with care.",
    "classicalLimit": "Environmental models depend on measurement quality, time, and location. Explore the evidence and its uncertainty before drawing conclusions beyond the study area.",
    "aiBreakthrough": "Start with pollution records, waste images, or rainfall; progress to satellite mapping; explore spatial uncertainty and planning trade-offs with suitable background.",
    "careerWhy": "Use your work to demonstrate problem framing, data preparation, evaluation, and clear communication. A student project can support a portfolio, but does not guarantee employment or professional qualification.",
    "blueprint": "Air Pollution Concentration Forecasting",
    "blueprintIndex": 0
  },
  "education": {
    "deptKey": "education",
    "name": "Education & Learning Sciences",
    "icon": "\ud83d\udcda",
    "tagline": "Build learning tools that support teachers and students.",
    "classicalLimit": "Educational judgement stays with teachers and learners. Begin with synthetic or approved de-identified data and evaluate usefulness without making consequential decisions about students.",
    "aiBreakthrough": "Start with resource tagging, FAQ search, or synthetic quiz data; progress to feedback evaluation; explore adaptive learning in simulation with teacher review.",
    "careerWhy": "Use your work to demonstrate problem framing, data preparation, evaluation, and clear communication. A student project can support a portfolio, but does not guarantee employment or professional qualification.",
    "blueprint": "Study-Time and Quiz-Score Exploration",
    "blueprintIndex": 0
  }
    };

