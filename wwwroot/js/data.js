    const projectsData = {
  "eee": {
    "name": "Electrical & Electronics Engineering (EEE)",
    "short": "EEE",
    "icon": "\u26a1",
    "color": "#f59e0b",
    "tagline": "Powering smart grids, autonomous energy management, and intelligent power electronics.",
    "description": "Artificial Intelligence is transforming EEE from classical control systems to adaptive, self-healing electrical ecosystems. Applications span predictive equipment maintenance, microgrid energy dispatch, physics-informed load forecasting, and silicon-level fault diagnostics.",
    "projects": [
      {
        "id": "eee-01",
        "title": "Physics-Informed Neural Networks (PINNs) for Power Grid Transient Stability",
        "difficulty": "Advanced",
        "summary": "Hybrid deep learning model integrating swing equations with neural operators to predict multi-generator transient instability in sub-millisecond windows.",
        "problem": "Traditional transient stability analysis relies on solving high-dimensional non-linear differential-algebraic equations via numerical integration, which is too slow for real-time grid contingency management during severe line faults.",
        "ai_technique": "Physics-Informed Neural Networks (PINNs) & Fourier Neural Operators (FNO)",
        "tech_stack": [
          "PyTorch",
          "DeepXDE",
          "OpenDSS",
          "Pandapower",
          "Python 3.10"
        ],
        "data_pipeline": "IEEE 39-Bus New England & IEEE 118-Bus dynamic simulation logs generated under 10,000 N-1 line tripping scenarios.",
        "hardware_target": "NVIDIA RTX 4090 / Cloud GPU Cluster for training; Edge industrial controller for inference.",
        "impact": "Reduces transient stability assessment latency from 4.2 seconds to 8.5 milliseconds (99.8% speedup) while guaranteeing physical conservation laws.",
        "code_snippet": "class GridPINN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(nn.Linear(39, 128), nn.GELU(), nn.Linear(128, 64), nn.Linear(64, 39))\n    def loss_physics(self, rotor_angle, omega, P_mech, P_elec, D, H):\n        # Residual of swing equation: 2H d(omega)/dt = P_m - P_e - D*omega\n        d_omega = torch.autograd.grad(omega, self.time, create_graph=True)[0]\n        return torch.mean((2*H*d_omega - (P_mech - P_elec - D*omega))**2)"
      },
      {
        "id": "eee-02",
        "title": "Deep Reinforcement Learning for Renewable Microgrid Energy Management",
        "difficulty": "Advanced",
        "summary": "Multi-agent deep Q-learning and PPO agent coordinating battery storage (BESS), solar PV dispatch, and dynamic tariff arbitrage.",
        "problem": "High stochasticity of solar irradiance and local load profiles causes voltage fluctuations and expensive peak-demand grid penalties.",
        "ai_technique": "Multi-Agent Proximal Policy Optimization (MAPPO) & DRL",
        "tech_stack": [
          "Ray RLlib",
          "Gymnasium",
          "PyTorch",
          "Pandas",
          "FastAPI"
        ],
        "data_pipeline": "NREL Solar Radiation Database + 15-minute resolution smart meter time-series data.",
        "hardware_target": "Raspberry Pi 4 / Siemens IoT2050 Industrial Gateway connected via Modbus RTU.",
        "impact": "Cuts electricity operational costs by 23.4% and limits peak-load feeder stress by 38% under high solar penetration.",
        "code_snippet": "from ray.rllib.algorithms.ppo import PPOConfig\nconfig = (PPOConfig().environment(MicrogridGymEnv)\n    .framework('torch')\n    .rollouts(num_rollout_workers=4)\n    .training(lr=3e-4, gamma=0.99, clip_param=0.2))\nalgo = config.build()"
      },
      {
        "id": "eee-03",
        "title": "Acoustic and Vibration Edge-AI for Induction Motor Fault Detection",
        "difficulty": "Intermediate",
        "summary": "1D-CNN and Spectrogram ViT deployed on microcontrollers for early detection of bearing outer-race failure, stator winding faults, and rotor eccentricity.",
        "problem": "Sudden induction motor catastrophic failures cause millions in industrial downtime. Manual acoustic inspection is subjective and periodic.",
        "ai_technique": "1D Convolutional Neural Network + Wavelet Packet Transform (WPT)",
        "tech_stack": [
          "TensorFlow Lite for Microcontrollers",
          "Edge Impulse",
          "C++",
          "Python",
          "NumPy"
        ],
        "data_pipeline": "CWRU (Case Western Reserve University) Bearing Vibration Dataset + real factory 3-axis accelerometer logs.",
        "hardware_target": "STM32H7 / ESP32-S3 with MEMS accelerometer (ADXL345) and I2S microphone (INMP441).",
        "impact": "Achieves 98.7% classification accuracy across 4 fault categories with 14ms inference time running entirely on a 160MHz MCU.",
        "code_snippet": "# Edge feature extraction via FFT Spectrogram -> TinyCNN\ninput_data = apply_fft(mems_samples, window_size=256)\ninterpreter.set_tensor(input_index, input_data)\ninterpreter.invoke()\npredictions = interpreter.get_tensor(output_index) # [Healthy, InnerRace, OuterRace, BallFault]"
      },
      {
        "id": "eee-04",
        "title": "Transformer-Based Solar PV Farm Power Generation Forecasting",
        "difficulty": "Intermediate",
        "summary": "Spatial-temporal Temporal Fusion Transformer (TFT) modeling meteorological forecasts, cloud satellite imagery, and historical inverter telemetry.",
        "problem": "Cloud-induced ramping events cause severe generation drops within minutes, risking local distribution feeder destabilization.",
        "ai_technique": "Temporal Fusion Transformer (TFT) & Cross-Attention Mechanisms",
        "tech_stack": [
          "PyTorch Forecasting",
          "Scikit-Learn",
          "Dask",
          "Plotly",
          "PostgreSQL"
        ],
        "data_pipeline": "100 MW solar plant telemetry (pyranometers, ambient temp, panel surface temp, active power output).",
        "hardware_target": "On-premise edge server (Intel Xeon / NVIDIA T4).",
        "impact": "Reduces Normalized Root Mean Square Error (nRMSE) to 4.1% for 1-hour day-ahead forecasting, beating standard ARIMA by 42%.",
        "code_snippet": "from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet\ntraining_data = TimeSeriesDataSet(df, time_idx='time_step', target='power_mw', group_ids=['inverter_id'], max_encoder_length=96, max_prediction_length=24)"
      },
      {
        "id": "eee-05",
        "title": "Autonomous Lithium-Ion Battery State of Health (SOH) Estimation",
        "difficulty": "Intermediate",
        "summary": "Dual-stream Bi-LSTM with Bayesian optimization estimating cell capacity degradation and Remaining Useful Life (RUL) from partial charging curves.",
        "problem": "Battery degradation is non-linear and electrochemical impedance spectroscopy (EIS) is impossible during active electric vehicle driving cycles.",
        "ai_technique": "Bidirectional LSTM + Gaussian Process Regression (GPR) for uncertainty bounds",
        "tech_stack": [
          "TensorFlow/Keras",
          "Scipy",
          "Optuna",
          "Pandas",
          "Matplotlib"
        ],
        "data_pipeline": "NASA Battery Prognostics Dataset + Oxford Battery Degradation Suite.",
        "hardware_target": "Automotive BMS Microcontroller (Infineon AURIX TC3xx / NXP S32K).",
        "impact": "Predicts battery SOH with under 1.2% Mean Absolute Error even with incomplete 15-minute opportunistic charging sessions.",
        "code_snippet": "model = Sequential([\n    Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, features)),\n    Dropout(0.2),\n    Bidirectional(LSTM(32)),\n    Dense(16, activation='relu'),\n    Dense(1, activation='linear') # Remaining Capacity / SOH\n])"
      },
      {
        "id": "eee-06",
        "title": "Edge-AI Smart Inverter for Harmonic Distortion Mitigation",
        "difficulty": "Advanced",
        "summary": "Neural network-based Active Power Filter (APF) calculating instant compensation currents to eliminate grid harmonic pollution (THD).",
        "problem": "Proliferation of non-linear loads (rectifiers, EV chargers, variable frequency drives) degrades power quality below IEEE 519 standards.",
        "ai_technique": "Deep Artificial Neural Network (ANN) approximating Generalized Instantaneous Reactive Power Theory (p-q theory)",
        "tech_stack": [
          "MATLAB/Simulink",
          "Xilinx Vivado HLS",
          "C++",
          "PyTorch"
        ],
        "data_pipeline": "Synthesized 3-phase distorted voltage/current waveforms containing 3rd, 5th, 7th, and 11th harmonics.",
        "hardware_target": "AMD-Xilinx Zynq UltraScale+ MPSoC / TI C2000 DSP.",
        "impact": "Reduces Total Harmonic Distortion (THD) from 18.4% to 2.1% across dynamic non-linear load swings under 2 PWM switching periods.",
        "code_snippet": "# High-frequency neural inference pipeline for FPGA synthesis (HLS)\n# void infer_harmonic_current(float i_load[3], float i_comp[3]);"
      },
      {
        "id": "eee-07",
        "title": "Computer Vision-Based High-Voltage Transmission Line Defect Inspection",
        "difficulty": "Intermediate",
        "summary": "UAV-mounted edge computer running YOLOv8-OBB and segmentation models to spot rusted dampers, cracked porcelain insulators, and vegetation encroachment.",
        "problem": "Manual line inspection over thousands of kilometers of rugged terrain is hazardous, slow, and prone to human inspection fatigue.",
        "ai_technique": "YOLOv8 Oriented Bounding Box (OBB) & Mask R-CNN",
        "tech_stack": [
          "Ultralytics YOLO",
          "OpenCV",
          "NVIDIA DeepStream",
          "DroneKit",
          "PyTorch"
        ],
        "data_pipeline": "High-resolution 4K aerial drone imagery of electrical towers under diverse lighting and weather.",
        "hardware_target": "NVIDIA Jetson Orin Nano onboard inspection drone.",
        "impact": "Detects 12 classes of transmission line hardware defects at 45 FPS with 94.6% mAP50, inspecting 35km of line per battery charge.",
        "code_snippet": "from ultralytics import YOLO\nmodel = YOLO('yolov8n-obb-powerlines.pt')\nresults = model.predict(source='drone_feed_rtsp', conf=0.6, stream=True)"
      },
      {
        "id": "eee-08",
        "title": "AI-Driven Non-Intrusive Load Monitoring (NILM) for Smart Meters",
        "difficulty": "Beginner",
        "summary": "Sequence-to-point deep convolutional autoencoder disaggregating household total power consumption into individual appliance loads.",
        "problem": "Deploying separate power sensors for every home appliance is cost-prohibitive for residential energy efficiency audits.",
        "ai_technique": "Sequence-to-Point (Seq2Point) 1D CNN with Dilated Convolutions",
        "tech_stack": [
          "PyTorch",
          "NILMTK toolkit",
          "FastAPI",
          "SQLite",
          "Chart.js"
        ],
        "data_pipeline": "REDD (Reference Energy Disaggregation Data Set) and UK-DALE electrical load records.",
        "hardware_target": "Smart energy meter with embedded ARM Cortex-A7 or Home Assistant server.",
        "impact": "Identifies HVAC, refrigeration, EV charging, and water heaters with 89.2% F1-score from a single smart meter power feed.",
        "code_snippet": "class Seq2Point(nn.Module):\n    def __init__(self, window_size=99):\n        super().__init__()\n        self.conv = nn.Sequential(nn.Conv1d(1, 30, 10, dilation=1), nn.ReLU(), nn.Conv1d(30, 40, 8, dilation=2), nn.ReLU())\n        self.fc = nn.Sequential(nn.Linear(40 * 79, 1024), nn.ReLU(), nn.Linear(1024, 1))"
      },
      {
        "id": "eee-09",
        "title": "Autonomous Substation Thermal Anomaly Spotting via Infrared Vision",
        "difficulty": "Intermediate",
        "summary": "Thermal radiometric image segmentation combined with temporal heat trend analysis to spot overheated transformer bushings and loose busbar disconnects.",
        "problem": "Thermal runaway and contact resistance failures in high-voltage switchyards can lead to explosive transformer fires if unaddressed.",
        "ai_technique": "U-Net with radiometric temperature matrix thresholding and anomaly bounding",
        "tech_stack": [
          "OpenCV",
          "PyTorch",
          "FLIR Atlas SDK",
          "MQTT",
          "Grafana"
        ],
        "data_pipeline": "Long-wave infrared (LWIR) radiometric thermal video streams from substation pan-tilt-zoom cameras.",
        "hardware_target": "Industrial Fanless Edge PC with FLIR Thermal Camera.",
        "impact": "Flags dangerous thermal hot spots exceeding 85\u00b0C delta over ambient 14 days before catastrophic insulation breakdown.",
        "code_snippet": "# Process radiometric matrix: Celsius = (RAW - B) / R\ntemp_matrix = flir_sdk.extract_temperature(thermal_frame)\nanomaly_mask = temp_matrix > (ambient_temp + 35.0)\ncontours, _ = cv2.findContours(anomaly_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"
      },
      {
        "id": "eee-10",
        "title": "Reinforcement Learning for Electric Vehicle Charging Fleet Optimization",
        "difficulty": "Intermediate",
        "summary": "Deep deterministic policy gradient (DDPG) algorithm optimizing charging schedules for 100+ commercial fleet EVs balancing transformer thermal limits.",
        "problem": "Uncoordinated simultaneous fast charging of delivery fleets causes local distribution transformer overloads and heavy demand surcharges.",
        "ai_technique": "Deep Deterministic Policy Gradients (DDPG) & Convex Optimization Hybrid",
        "tech_stack": [
          "Stable-Baselines3",
          "Pyomo",
          "Pandas",
          "Streamlit",
          "Docker"
        ],
        "data_pipeline": "Commercial fleet telematics (arrival SoC, target departure time, battery capacity) + dynamic spot electricity prices.",
        "hardware_target": "Depot Central Energy Management Server.",
        "impact": "Eliminates transformer overload occurrences 100% while reducing total fleet charging electricity expenditure by 28.6%.",
        "code_snippet": "from stable_baselines3 import DDPG\nmodel = DDPG('MlpPolicy', EVDepotEnv, learning_rate=1e-3, buffer_size=50000, batch_size=64)\nmodel.learn(total_timesteps=100000)"
      }
    ]
  },
  "ece": {
    "name": "Electronics & Communication Engineering (ECE)",
    "short": "ECE",
    "icon": "\ud83d\udce1",
    "color": "#3b82f6",
    "tagline": "Architecting 5G/6G beamforming, neural signal processing, and intelligent wireless communication.",
    "description": "ECE merges signal theory, semiconductor physics, and communications with AI. Emerging applications include deep learning physical-layer transceivers, cognitive radio spectrum sensing, channel state estimation for Massive MIMO, and ultra-low-power neuromorphic hardware design.",
    "projects": [
      {
        "id": "ece-01",
        "title": "Deep Learning-Based Channel Estimation for 5G Massive MIMO Systems",
        "difficulty": "Advanced",
        "summary": "Convolutional Residual Network estimating high-dimensional Channel State Information (CSI) from minimal pilot signals in mmWave massive MIMO.",
        "problem": "In massive MIMO (64+ antenna arrays), pilot contamination and overhead scale linearly with antennas, drastically eating into usable data throughput.",
        "ai_technique": "CsiNet (Deep Complex-Valued Convolutional Autoencoder)",
        "tech_stack": [
          "PyTorch",
          "Sionna (NVIDIA 6G Library)",
          "NumPy",
          "Matplotlib"
        ],
        "data_pipeline": "3GPP TR 38.901 5G channel models synthesized via Sionna ray-tracing simulator.",
        "hardware_target": "NVIDIA Grace Hopper / FPGA DSP accelerator in Baseband Unit (BBU).",
        "impact": "Reduces CSI feedback overhead by 87.5% with less than 0.8 dB NMSE degradation at 28 GHz mmWave carrier frequencies.",
        "code_snippet": "import sionna\nfrom sionna.channel import RayTracing\n# Deep autoencoder compressing complex H channel tensor into compact latent codeword\nencoded = self.encoder(csi_real_imag)\nfeedback_bits = quantize(encoded, num_bits=16)\nrecovered_csi = self.decoder(dequantize(feedback_bits))"
      },
      {
        "id": "ece-02",
        "title": "Edge Cognitive Radio Spectrum Sensing via Deep Convolutional Networks",
        "difficulty": "Intermediate",
        "summary": "Deep CNN recognizing primary user transmissions in low SNR environments using In-Phase/Quadrature (I/Q) raw samples.",
        "problem": "Conventional energy detectors fail under low signal-to-noise ratio (< -10 dB) and struggle against non-stationary noise floors in crowded spectrum bands.",
        "ai_technique": "2D-CNN trained on Raw I/Q Constellation Diagrams & Cyclic Feature Maps",
        "tech_stack": [
          "GNU Radio",
          "TensorFlow",
          "UHD (USRP Hardware Driver)",
          "Python 3.9"
        ],
        "data_pipeline": "RadioML 2018.01A benchmark dataset + live SDR over-the-air capture.",
        "hardware_target": "RTL-SDR / HackRF One paired with Raspberry Pi 4 or Jetson Nano.",
        "impact": "Achieves 93.4% probability of detection at -12 dB SNR with a false alarm rate below 4% across 16 digital modulation schemes.",
        "code_snippet": "# GNU Radio to Python IQ buffer stream\niq_samples = sdr.read_samples(1024)\nfeatures = np.stack([np.real(iq_samples), np.imag(iq_samples)], axis=-1)\npred = model.predict(np.expand_dims(features, 0)) # [Primary User Active vs Idle]"
      },
      {
        "id": "ece-03",
        "title": "Automated PCB Component and Solder Defect Inspection with YOLOv8",
        "difficulty": "Intermediate",
        "summary": "High-throughput industrial vision system classifying missing SMD components, solder bridging, tombstoning, and polarity misalignment.",
        "problem": "Surface mount technology (SMT) inspection stations rely on rule-based optical inspection (AOI) which generates excessive false defect rejects.",
        "ai_technique": "YOLOv8x + Fine-grained Attention Heads for Micro-Defect Localization",
        "tech_stack": [
          "Ultralytics YOLO",
          "OpenCV",
          "Industrial GigE Camera API",
          "PyQt6"
        ],
        "data_pipeline": "PKU-Market-PCB dataset + 5,000 custom micro-macro SMT manufacturing board photos.",
        "hardware_target": "Industrial Vision PC with NVIDIA RTX A4000 GPU and ring LED lighting.",
        "impact": "Inspects complete motherboard PCBs in under 180ms with 99.1% precision, dropping false rejection rate by 65%.",
        "code_snippet": "results = pcb_detector.predict(board_image, imgsz=1280, conf=0.45)\nfor box in results[0].boxes:\n    defect_class = pcb_detector.names[int(box.cls)]\n    # e.g., 'solder_bridge', 'missing_capacitor', 'tombstone'"
      },
      {
        "id": "ece-04",
        "title": "Neuromorphic Spiking Neural Network (SNN) for Keyword Spotting",
        "difficulty": "Advanced",
        "summary": "Bio-inspired event-based Spiking Neural Network performing wake-word voice recognition with micro-watt energy consumption.",
        "problem": "Always-on voice assistants quickly drain battery on wearable hearables and smart earbuds when running traditional continuous deep CNNs.",
        "ai_technique": "Spiking Neural Network (SNN) with Leaky Integrate-and-Fire (LIF) Neurons",
        "tech_stack": [
          "snnTorch",
          "PyTorch",
          "Lava Neuromorphic Framework",
          "NumPy"
        ],
        "data_pipeline": "Google Speech Commands v2 transformed via Cochlear Filterbanks into spike trains.",
        "hardware_target": "SynSense Speck / Intel Loihi 2 / BrainChip Akida Neuromorphic Chip.",
        "impact": "Consumes less than 45 microwatts of power with 95.8% accuracy on a 12-word lexicon, enabling year-long watch battery operation.",
        "code_snippet": "import snntorch as snn\nlif1 = snn.Leaky(beta=0.9, spike_grad=snn.surrogate.fast_sigmoid())\nmem = lif1.init_leaky()\nfor step in range(num_steps):\n    spk, mem = lif1(input_spike_train[step], mem)"
      },
      {
        "id": "ece-05",
        "title": "AI-Powered Adaptive Beamforming for Satellite-to-Ground Communications",
        "difficulty": "Advanced",
        "summary": "Deep Reinforcement Learning dynamically adjusting phase shifter weights in phased array antennas to maintain high-gain links to LEO satellites.",
        "problem": "Fast angular velocities of Low Earth Orbit (LEO) constellations cause frequent handovers and phase distortion due to atmospheric scintillation.",
        "ai_technique": "Deep Deterministic Policy Gradient (DDPG) with Doppler Tracking",
        "tech_stack": [
          "PyTorch",
          "MATLAB Phased Array System Toolbox",
          "SciPy"
        ],
        "data_pipeline": "LEO orbit telemetry (Starlink/OneWeb TLE orbital files) + atmospheric attenuation matrices.",
        "hardware_target": "Software Defined Phased Array Controller / Zynq UltraScale+ FPGA.",
        "impact": "Maintains 99.4% link availability during high-speed passes, improving received Signal-to-Interference-plus-Noise Ratio (SINR) by 5.8 dB.",
        "code_snippet": "# Compute complex weights for N-element phased array\nphase_angles = actor_network(state_telemetry)\nw = torch.exp(1j * phase_angles)\narray_factor = torch.matmul(steering_vector, w)"
      },
      {
        "id": "ece-06",
        "title": "RF Fingerprinting for IoT Device Authentication and Anti-Spoofing",
        "difficulty": "Intermediate",
        "summary": "Transformer-based physical-layer identification extracting microscopic silicon hardware imperfections from Wi-Fi/Bluetooth preamble waveforms.",
        "problem": "MAC addresses and cryptographic keys can be cloned or stolen; zero-trust IoT networks require unclonable physical-layer identity.",
        "ai_technique": "1D Vision Transformer (ViT) on Raw Baseband Transients",
        "tech_stack": [
          "PyTorch",
          "Ettus USRP N210",
          "GNU Radio",
          "Scikit-Learn"
        ],
        "data_pipeline": "Transmitter RF fingerprint datasets containing 50 identical ESP32 and CC2650 chips transmitting IEEE 802.11 frames.",
        "hardware_target": "USRP SDR connected to security gateway.",
        "impact": "Correctly authenticates legitimate devices with 98.3% accuracy, rejecting MAC-cloned counterfeit transmitters in real time.",
        "code_snippet": "# Silicon transient extraction\ntransient_slice = extract_preamble_turn_on(rf_signal)\nembedding = rf_vit_model(transient_slice)\nsimilarity = cosine_distance(embedding, enrolled_device_embedding)"
      },
      {
        "id": "ece-07",
        "title": "End-to-End Deep Learning Optical Fiber Non-Linearity Compensation",
        "difficulty": "Advanced",
        "summary": "Bi-directional Gated Recurrent Units (Bi-GRU) modeling and reversing Kerr non-linear phase noise in long-haul optical fiber links.",
        "problem": "Kerr non-linearities severely constrain transmission reach and maximum launch power in multi-terabit coherent optical communications.",
        "ai_technique": "Deep Learned Digital Back-Propagation (Learned DBP via Bi-GRU)",
        "tech_stack": [
          "PyTorch",
          "OptiSystem",
          "NumPy",
          "CUDA C++"
        ],
        "data_pipeline": "Synthetic and laboratory 64-QAM 800 Gbps single-mode fiber (SMF) propagation records over 1,200 km.",
        "hardware_target": "Optical DSP ASIC / high-speed FPGA accelerator.",
        "impact": "Extends optical transmission reach by 450 km or increases achievable data rate by 1.8 bits/symbol compared to linear equalization.",
        "code_snippet": "# Learned DBP layer replacing split-step Fourier method\nclass LearnedDBPBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.dispersion = nn.Conv1d(2, 2, kernel_size=15, padding=7)\n        self.kerr_net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))"
      },
      {
        "id": "ece-08",
        "title": "Embedded Edge Voice Activity Detection (VAD) and Denoising",
        "difficulty": "Beginner",
        "summary": "Dual-path recurrent network isolating human voice from heavy background noise on ultra-low-power microcontrollers.",
        "problem": "Environmental noise (wind, traffic, factory machinery) destroys automated voice transcription and communications clarity.",
        "ai_technique": "Depthwise Separable 1D-ConvNet + GRU (RNNoise-inspired)",
        "tech_stack": [
          "TensorFlow Lite",
          "C",
          "Python",
          "Librosa",
          "PyAudio"
        ],
        "data_pipeline": "Mozilla Common Voice + DEMAND noise dataset mixed at SNRs between -5 dB and 15 dB.",
        "hardware_target": "ARM Cortex-M4 (STM32F4) / Nordic nRF5340 Bluetooth audio SoC.",
        "impact": "Delivers +14.2 dB PESQ voice quality improvement with less than 28ms algorithmic latency and 120KB memory footprint.",
        "code_snippet": "// C embedded inference loop running on ARM Cortex-M\ntflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);\ninterpreter.AllocateTensors();\n// Audio circular buffer DMA -> FFT -> Run Model -> IFFT Output"
      },
      {
        "id": "ece-09",
        "title": "Indoor Localization via Wi-Fi CSI Spatial-Temporal Attention Networks",
        "difficulty": "Intermediate",
        "summary": "Deep neural network exploiting multipath Channel State Information subcarriers from commodity Wi-Fi access points for sub-meter indoor tracking.",
        "problem": "GPS signals cannot penetrate indoors, while Bluetooth beacons require tedious battery maintenance and hardware deployment.",
        "ai_technique": "Spatial-Temporal Graph Convolutional Network (ST-GCN)",
        "tech_stack": [
          "PyTorch",
          "Linux 802.11n CSI Tool / ESP32 CSI",
          "Pandas",
          "Plotly"
        ],
        "data_pipeline": "Wi-Fi CSI packets collected across 56 subcarriers across 3 antennas in a 500 sq meter office building.",
        "hardware_target": "3 standard Wi-Fi 6 routers (ESP32-S3 / Intel 5300 NIC).",
        "impact": "Achieves 0.42-meter mean localization error without requiring users to wear any special tags or beacons.",
        "code_snippet": "# Process complex CSI matrix: 3 antennas x 56 subcarriers x T time stamps\ncsi_amplitude = np.abs(csi_matrix)\ncsi_phase = np.unwrap(np.angle(csi_matrix), axis=-1)\npredicted_xy = st_gcn_model(torch.tensor(csi_features))"
      },
      {
        "id": "ece-10",
        "title": "Machine Learning for Analog Circuit Sizing and Layout Automation",
        "difficulty": "Advanced",
        "summary": "Graph Neural Network and reinforcement learning agent automating transistor sizing (W/L) for operational amplifiers and PLLs.",
        "problem": "Manual analog and RF integrated circuit sizing requires weeks of iterative SPICE simulations by seasoned analog design engineers.",
        "ai_technique": "Graph Attention Network (GAT) + Bayesian Optimization",
        "tech_stack": [
          "Ngspice / Cadence Spectre",
          "PyTorch Geometric",
          "BoTorch",
          "Python"
        ],
        "data_pipeline": "Two-stage Miller OTA, Bandgap references, and Folded Cascode netlists in TSMC 65nm / SkyWater 130nm PDKs.",
        "hardware_target": "CAD Workstation / Multi-core Linux Server.",
        "impact": "Achieves 100% design specification compliance (DC Gain > 75dB, Phase Margin > 60 deg, UGBW > 50MHz) in 4 hours vs 2 weeks manual tuning.",
        "code_snippet": "from torch_geometric.nn import GATConv\n# Graph representation of SPICE circuit netlist (nodes: transistors, edges: nets)\ngraph = circuit_to_graph(netlist_file)\npredicted_performance = gnn_surrogate(graph.x, graph.edge_index)"
      }
    ]
  },
  "civil": {
    "name": "Civil & Structural Engineering",
    "short": "Civil",
    "icon": "\ud83c\udfd7\ufe0f",
    "color": "#10b981",
    "tagline": "Building resilient infrastructure, smart cities, and automated structural health monitoring.",
    "description": "Civil Engineering combines structural mechanics, geotechnics, and urban planning with AI. Modern projects leverage computer vision for crack detection, digital twins for bridge deflection, hydrological flood forecasting, and generative BIM structural optimization.",
    "projects": [
      {
        "id": "civ-01",
        "title": "Computer Vision Crack and Spalling Segmentation on Concrete Bridges",
        "difficulty": "Intermediate",
        "summary": "Pixel-level semantic segmentation isolating sub-millimeter cracks, concrete spalling, and rebar exposure from drone photography.",
        "problem": "Manual bridge inspection requires cherry-picker trucks, road lane closures, and subjective human visual assessments.",
        "ai_technique": "SegFormer / DeepLabV3+ with Crack-Width Measurement Metric",
        "tech_stack": [
          "PyTorch",
          "OpenCV",
          "Albumentations",
          "Docker",
          "QGIS"
        ],
        "data_pipeline": "Bridge inspection image database containing 8,500 annotated high-res photos under various sun and shadow angles.",
        "hardware_target": "Autonomous Inspection Drone + Cloud Analysis Server.",
        "impact": "Identifies cracks down to 0.15mm thickness with 92.4% Mean IoU, automatically compiling municipal safety rating compliance reports.",
        "code_snippet": "import segmentation_models_pytorch as smp\nmodel = smp.DeepLabV3Plus(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=3)\n# Classes: [Background, Crack, Spalling]"
      },
      {
        "id": "civ-02",
        "title": "Digital Twin Structural Health Monitoring (SHM) via GNNs",
        "difficulty": "Advanced",
        "summary": "Graph Neural Network modeling accelerometer, strain gauge, and inclinometer sensor nodes across suspension bridges to spot modal shifts.",
        "problem": "Vibration signals are heavily masked by ambient temperature changes and daily vehicular traffic, disguising internal structural micro-damage.",
        "ai_technique": "Spatio-Temporal Graph Neural Network (ST-GNN) & Physics-Guided Loss",
        "tech_stack": [
          "PyTorch Geometric",
          "OpenSees (Structural FEA)",
          "Pandas",
          "Grafana"
        ],
        "data_pipeline": "Continuous 2-year sensor telemetry from a 1.2km suspension bridge (64 accelerometers + 32 strain gauges).",
        "hardware_target": "On-bridge industrial edge server (Advantech) + cloud digital twin dashboard.",
        "impact": "Detects structural stiffness degradation and bolt loosening 6 months before visible external cracks appear, with zero false alerts.",
        "code_snippet": "# Structural sensor network represented as spatial graph based on physical bridge girder topology\nclass BridgeGNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = GCNConv(in_features=8, out_features=32)\n        self.gru = nn.GRU(32, 16, batch_first=True)\n        self.classifier = nn.Linear(16, 2) # [Healthy, Damaged]"
      },
      {
        "id": "civ-03",
        "title": "Urban Flood Inundation Prediction via Spatial-Temporal LSTM",
        "difficulty": "Intermediate",
        "summary": "Real-time rainfall-runoff inundation mapping fusing weather radar, storm drain telemetry, and digital elevation models (DEM).",
        "problem": "Hydrodynamic 2D flood simulators (e.g. SWMM, HEC-RAS) take hours to run, making real-time evacuation routing impossible during cloudburst storms.",
        "ai_technique": "ConvLSTM2D + Physics-Informed Continuity Constraint",
        "tech_stack": [
          "TensorFlow/Keras",
          "GDAL",
          "Rasterio",
          "GeoPandas",
          "PostGIS"
        ],
        "data_pipeline": "Doppler precipitation radar grids + 1m resolution LiDAR digital terrain models + drainage telemetry.",
        "hardware_target": "Municipal Disaster Management Center Cloud Cluster.",
        "impact": "Generates street-level flood depth inundation predictions for the next 3 hours in 1.4 seconds (compared to 4 hours in physical simulation).",
        "code_snippet": "model = Sequential([\n    ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(None, 128, 128, 4)),\n    BatchNormalization(),\n    Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same') # Output: Inundation Depth (m)\n])"
      },
      {
        "id": "civ-04",
        "title": "Generative AI for Optimal Concrete Structural Topology Optimization",
        "difficulty": "Advanced",
        "summary": "Conditional Diffusion Models and Generative Adversarial Networks designing lightweight, high-strength structural beams and trusses.",
        "problem": "Finite Element Method (FEM) topology optimization is computationally punishing and frequently generates shapes that are impossible to 3D print or cast.",
        "ai_technique": "Conditional Diffusion Probabilistic Models (CDPM) + FEA Validation",
        "tech_stack": [
          "PyTorch",
          "ANSYS / Abaqus Python API",
          "Trimesh",
          "Open3D"
        ],
        "data_pipeline": "100,000 SIMP (Solid Isotropic Material with Penalization) optimized 3D structural load-stress volumes.",
        "hardware_target": "NVIDIA RTX 4090 / A100 Workstation.",
        "impact": "Reduces structural material mass by 28% while sustaining equivalent design load ratings, speeding up design discovery 800x.",
        "code_snippet": "# Condition diffusion on boundary conditions: load vectors and fixed support nodes\npredicted_density_field = diffusion_model.sample(batch_size=1, condition=stress_tensor)"
      },
      {
        "id": "civ-05",
        "title": "AI Pavement Distress and Pothole Assessment via In-Vehicle Dashcam",
        "difficulty": "Beginner",
        "summary": "Lightweight edge-AI computer vision model mounted in municipal garbage trucks and city buses mapping road quality across thousands of miles.",
        "problem": "City road surveys rely on specialized laser profilometer vans costing over $300,000 each, limiting surveys to once every 3-5 years.",
        "ai_technique": "YOLOv8-Nano + GPS Geotagging + Pavement Condition Index (PCI) Score",
        "tech_stack": [
          "Ultralytics YOLO",
          "OpenCV",
          "SQLite",
          "Leaflet.js",
          "Python"
        ],
        "data_pipeline": "Road Damage Dataset (RDD2022) containing pothole, alligator cracking, and longitudinal rutting samples.",
        "hardware_target": "Raspberry Pi 5 with AI Kit / Android Smartphone.",
        "impact": "Surveys city-wide road networks weekly at 1/100th the cost of dedicated survey vehicles, with 91.5% pothole detection accuracy.",
        "code_snippet": "# Live road damage parsing and GIS mapping\nresults = model(frame)\nfor r in results:\n    if r.boxes:\n        pothole_count = sum(1 for c in r.boxes.cls if model.names[int(c)] == 'pothole')\n        log_road_condition(lat, lon, pothole_count)"
      },
      {
        "id": "civ-06",
        "title": "Geotechnical Tunnel Face Stability Classification using Rock Mass Vision",
        "difficulty": "Intermediate",
        "summary": "Vision Transformer classifying Geological Strength Index (GSI) and joint roughness from tunnel boring machine (TBM) forward face cameras.",
        "problem": "Tunneling through unexpected geological fault zones risks catastrophic tunnel cave-ins and multimillion-dollar TBM entrapment.",
        "ai_technique": "Vision Transformer (ViT-Base) fine-tuned on underground rock faces",
        "tech_stack": [
          "PyTorch",
          "Hugging Face Transformers",
          "OpenCV",
          "FastAPI"
        ],
        "data_pipeline": "High-intensity strobe camera images of tunnel excavation faces mapped against geologists' ground-truth RQD and Q-system ratings.",
        "hardware_target": "Ruggedized IP67 In-Cab Industrial PC inside TBM operator cabin.",
        "impact": "Delivers continuous real-time rock classification with 94.1% agreement with senior geotechnical engineers, preventing collapse events.",
        "code_snippet": "from transformers import ViTForImageClassification, ViTImageProcessor\nprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')\nmodel = ViTForImageClassification.from_pretrained('custom-tbm-rock-classifier')"
      },
      {
        "id": "civ-07",
        "title": "Smart Traffic Signal Optimization via Multi-Agent Reinforcement Learning",
        "difficulty": "Advanced",
        "summary": "Multi-agent reinforcement learning (MARL) synchronizing traffic light green-time phases across 20+ interconnected urban intersections.",
        "problem": "Fixed-time or simple inductive-loop traffic signals cause massive vehicular idling, smog emissions, and gridlocked commute corridors.",
        "ai_technique": "Multi-Agent Deep Deterministic Policy Gradient (MADDPG) / PressLight",
        "tech_stack": [
          "SUMO (Simulation of Urban MObility)",
          "Ray RLlib",
          "PyTorch",
          "Python"
        ],
        "data_pipeline": "Inductive loop vehicle counts, intersection camera flow rates, and Uber Movement travel times.",
        "hardware_target": "Traffic Management Center Server running SUMO simulation bridge.",
        "impact": "Reduces average commuter intersection wait times by 31.8% and vehicle carbon emissions by 19.4% during peak rush hours.",
        "code_snippet": "import traci\n# Connect to SUMO simulator and execute RL phase control\ntraci.start(['sumo', '-c', 'city_network.sumocfg'])\nstate = get_intersection_queue_lengths(traci)\naction = agent.act(state)\ntraci.trafficlight.setPhase('Junction_4', action)"
      },
      {
        "id": "civ-08",
        "title": "Satellite InSAR Land Subsidence and Slope Instability Forecasting",
        "difficulty": "Intermediate",
        "summary": "Recurrent neural network analyzing Interferometric Synthetic Aperture Radar (InSAR) time-series to spot millimeter-scale ground sinkholes.",
        "problem": "Groundwater over-extraction causes gradual land subsidence that cracks building foundations and underground utility pipelines without notice.",
        "ai_technique": "Transformer-based Sequence Forecaster + Spatial Kriging",
        "tech_stack": [
          "ISCE2 / SNAP",
          "PyTorch",
          "Rasterio",
          "NumPy",
          "Xarray"
        ],
        "data_pipeline": "Sentinel-1 C-band SAR radar imagery spanning 5 years of ascending and descending passes.",
        "hardware_target": "High-Performance Cloud Computing Cluster.",
        "impact": "Forecasts land subsidence rate with 1.8mm/year accuracy, providing 3 months advance warning before structural foundation failure.",
        "code_snippet": "# Process unwrapped interferogram phase deformation\ndef compute_deformation_mm(unwrapped_phase, wavelength=55.46):\n    return (unwrapped_phase * wavelength) / (-4 * np.pi)"
      },
      {
        "id": "civ-09",
        "title": "Construction Site Safety Compliance Monitoring via Edge Computer Vision",
        "difficulty": "Beginner",
        "summary": "Real-time CCTV video analytics verifying personal protective equipment (PPE: helmets, high-vis vests, harnesses) and heavy machinery exclusion zones.",
        "problem": "Falls and struck-by accidents remain the leading causes of worker fatalities on commercial construction sites worldwide.",
        "ai_technique": "YOLOv8 Person & PPE Detection + Geofencing Polygon Collision Math",
        "tech_stack": [
          "Ultralytics YOLO",
          "OpenCV",
          "Shapely",
          "Twilio API (SMS Alerts)"
        ],
        "data_pipeline": "Construction Safety PPE Dataset (hard hats, safety vests, gloves, worker bounding boxes).",
        "hardware_target": "NVIDIA Jetson AGX Orin connected to site RTSP cameras.",
        "impact": "Identifies PPE safety violations and dangerous machinery proximity in under 40ms, dropping site injury incidents by 74%.",
        "code_snippet": "from shapely.geometry import Point, Polygon\nzone_polygon = Polygon([(100, 100), (400, 100), (450, 400), (80, 400)])\nfor person_box in detected_workers:\n    center = Point(person_box.cx, person_box.cy)\n    if zone_polygon.contains(center) and excavator_active:\n        trigger_loudspeaker_alarm('Danger: Personnel in Heavy Equipment Swing Radius!')"
      },
      {
        "id": "civ-10",
        "title": "Machine Learning Prediction of High-Performance Concrete Compressive Strength",
        "difficulty": "Beginner",
        "summary": "XGBoost and CatBoost ensemble predicting 28-day concrete compressive strength based on chemical mix proportions and curing conditions.",
        "problem": "Standard cylinder break tests require waiting 28 days of curing time, causing expensive delays or rework if concrete mix batches fail spec.",
        "ai_technique": "XGBoost Regressor + SHAP (SHapley Additive exPlanations) for Interpretability",
        "tech_stack": [
          "XGBoost",
          "CatBoost",
          "Scikit-Learn",
          "SHAP",
          "Streamlit"
        ],
        "data_pipeline": "UCI Concrete Compressive Strength Dataset (cement, blast furnace slag, fly ash, water, superplasticizer, coarse/fine aggregate, age).",
        "hardware_target": "Tablet / Mobile Web App for concrete batching plant operators.",
        "impact": "Predicts 28-day strength instantly with an R\u00b2 score of 0.94 and RMSE under 3.5 MPa, saving concrete plants 12% in cement over-design.",
        "code_snippet": "import xgboost as xgb\nmodel = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)\nmodel.fit(X_train, y_train)\n# Explaining influence of water-cement ratio via SHAP\nexplainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_test)"
      }
    ]
  },
  "agriculture": {
    "name": "Agriculture & Smart Farming",
    "short": "Agriculture",
    "icon": "\ud83c\udf31",
    "color": "#84cc16",
    "tagline": "Empowering precision agriculture, autonomous robotics, crop disease diagnostics, and yield prediction.",
    "description": "Agriculture meets AI to tackle global food security, climate volatility, and sustainable farming. Key applications include multispectral drone crop scouting, AI-driven variable rate irrigation, robotic weed laser targeting, and livestock health monitoring.",
    "projects": [
      {
        "id": "agr-01",
        "title": "Hyperspectral Drone Crop Stress and Disease Early Warning System",
        "difficulty": "Intermediate",
        "summary": "3D-CNN and Vision Transformer identifying fungal rust, blight, and nitrogen deficiency in crops days before chlorosis becomes visible to the naked eye.",
        "problem": "By the time crop diseases manifest as visible yellowing leaves, systemic vascular damage has already occurred, causing major harvest yield loss.",
        "ai_technique": "Spectral-Spatial 3D-CNN + NDVI / NDRE Vegetation Indices",
        "tech_stack": [
          "PyTorch",
          "SpectralPython (SPy)",
          "Rasterio",
          "OpenCV",
          "Flask"
        ],
        "data_pipeline": "Multispectral & hyperspectral drone imagery (RedEdge, NIR, Red, Green bands) over wheat and maize fields.",
        "hardware_target": "DJI Matrice drone with MicaSense RedEdge camera + Cloud processing pipeline.",
        "impact": "Spots fungal infection 7 to 10 days before human visual observation, enabling targeted fungicide spraying that cuts chemical usage by 60%.",
        "code_snippet": "# Compute Normalized Difference Red Edge index for early chlorophyll stress\nndre = (nir_band - red_edge_band) / (nir_band + red_edge_band + 1e-6)\n# Feed 3D spectral datacube into spatial-spectral CNN\nstress_map = model(torch.tensor(spectral_cube))"
      },
      {
        "id": "agr-02",
        "title": "Autonomous Agricultural Robot for Precision Laser Weeding",
        "difficulty": "Advanced",
        "summary": "Real-time edge computer vision model distinguishing crop seedlings from weed varieties, guiding robotic delta arms or CO2 lasers to vaporize weeds.",
        "problem": "Broadband chemical herbicide spraying contaminates groundwater, accelerates herbicide resistance, and costs farmers thousands per acre.",
        "ai_technique": "YOLOv8-Seg Instance Segmentation with Sub-Millimeter Centroid Tracking",
        "tech_stack": [
          "ROS 2 (Robot Operating System)",
          "Ultralytics",
          "TensorRT",
          "C++",
          "Python"
        ],
        "data_pipeline": "Weed25 and CropWeed6K datasets containing annotated crops and 25 invasive weed species at cotyledon growth stage.",
        "hardware_target": "NVIDIA Jetson AGX Orin driving Galvo mirror laser controllers.",
        "impact": "Eliminates up to 98% of weeds at 5km/h field travel speed, reducing herbicide chemical dependency to zero.",
        "code_snippet": "# Real-time ROS 2 weed tracker node\ndef image_callback(self, msg):\n    frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')\n    results = self.model(frame, conf=0.7)\n    for weed in results.weeds:\n        self.publish_laser_target(weed.centroid_x, weed.centroid_y)"
      },
      {
        "id": "agr-03",
        "title": "Smart IoT & AI Soil Moisture Forecasting and Precision Irrigation",
        "difficulty": "Beginner",
        "summary": "LSTM network assimilating LoRaWAN soil moisture probes, evapo-transpiration metrics, and satellite weather data to automate drip valves.",
        "problem": "Irrigation based on fixed timers wastes up to 50% of fresh water through deep percolation and causes root rot diseases.",
        "ai_technique": "Bi-directional Long Short-Term Memory (Bi-LSTM) Regressor",
        "tech_stack": [
          "TensorFlow Lite",
          "LoRaWAN",
          "ESP32",
          "InfluxDB",
          "Grafana"
        ],
        "data_pipeline": "Soil moisture at 10cm, 30cm, 60cm depths, ambient temp, solar radiation, and FAO-56 Penman-Monteith ET0.",
        "hardware_target": "Solar-powered ESP32 gateway with LoRaWAN radio triggering 12V solenoid valves.",
        "impact": "Conserves 38.5% of freshwater while boosting crop harvest yield by 14% through sustained root-zone osmotic equilibrium.",
        "code_snippet": "# ESP32 Python/MicroPython control loop\npredicted_moisture_24h = lstm_predict(sensor_history)\nif predicted_moisture_24h < wilting_point_threshold:\n    trigger_solenoid_valve(duration_minutes=calculate_water_volume(field_capacity))"
      },
      {
        "id": "agr-04",
        "title": "End-to-End Crop Yield Prediction via Satellite Multimodal Fusion",
        "difficulty": "Intermediate",
        "summary": "Deep cross-modal attention network combining Sentinel-2 optical imagery, Sentinel-1 radar, soil topography, and regional meteorological series.",
        "problem": "National food security planners and commodity traders struggle to forecast harvest totals until late in the season, causing market shocks.",
        "ai_technique": "Cross-Modal Spatial-Temporal Attention Transformer",
        "tech_stack": [
          "PyTorch",
          "Google Earth Engine (GEE) Python API",
          "Xarray",
          "Pandas"
        ],
        "data_pipeline": "10-year USDA crop progress & NASS yield county records + Harmonized Landsat-Sentinel (HLS) surface reflectance.",
        "hardware_target": "Cloud Compute (Google Cloud Vertex AI / AWS SageMaker).",
        "impact": "Forecasts regional corn and soybean yield 2 months before harvest with an error margin below 4.8% at county level.",
        "code_snippet": "import ee\nee.Initialize()\nsentinel_collection = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2026-05-01', '2026-09-01').map(mask_clouds)\n# Feature extraction and multi-modal tensor assembly"
      },
      {
        "id": "agr-05",
        "title": "Edge-AI Audio Classification for Honeybee Colony Health and Queen Status",
        "difficulty": "Intermediate",
        "summary": "Continuous acoustic monitoring inside beehives analyzing buzzing frequency acoustics to detect queenlessness, swarming, and Varroa destructor mite stress.",
        "problem": "Colony Collapse Disorder (CCD) and unnoticed queen loss destroy commercial bee colonies. Opening hives during winter or bad weather stresses the bees.",
        "ai_technique": "Mel-Frequency Cepstral Coefficients (MFCC) + MobileNetV3 Audio Classifier",
        "tech_stack": [
          "TensorFlow Lite",
          "Librosa",
          "Raspberry Pi Zero 2W",
          "MQTT"
        ],
        "data_pipeline": "BeeHume acoustics database + 2,000 hours of beehive in-hive audio across healthy, queenless, and swarming states.",
        "hardware_target": "In-hive solar-powered Raspberry Pi Zero 2W with MEMS microphone.",
        "impact": "Alerts beekeepers to impending swarming or queen death 48 hours in advance with 96.2% precision without opening the hive.",
        "code_snippet": "# Audio feature transformation\nmfcc = librosa.feature.mfcc(y=audio_signal, sr=16000, n_mfcc=40)\nstatus = tflite_model.predict(mfcc) # [Healthy Queen, Queenless, Pre-Swarm Buzz]"
      },
      {
        "id": "agr-06",
        "title": "Automated Cattle Face Recognition and Lameness Detection",
        "difficulty": "Intermediate",
        "summary": "Non-invasive computer vision tracking cattle muzzle and face biometrics for individual identification and gait analysis to catch early lameness.",
        "problem": "Ear tags frequently tear out or are swapped, and manual lameness inspection across 1,000+ head dairy herds is labor-intensive and delayed.",
        "ai_technique": "ArcFace Metric Learning for Muzzle Recognition + YOLOv8-Pose for Gait Locomotion Scoring",
        "tech_stack": [
          "PyTorch",
          "OpenCV",
          "MediaPipe / MMPose",
          "FastAPI"
        ],
        "data_pipeline": "Video streams of dairy cows walking through the milking parlor corridor.",
        "hardware_target": "Overhead parlor camera connected to local Edge AI workstation.",
        "impact": "Replaces plastic ear tags with 99.1% biometric identity accuracy and detects hoof lesions (score 3+ lameness) 14 days earlier than dairy staff.",
        "code_snippet": "# Extract cow posture keypoints: spine curvature, head bobbing, step asymmetry\nkeypoints = pose_model(cow_gait_frame)\nspine_curvature_angle = calculate_angle(keypoints['shoulder'], keypoints['mid_back'], keypoints['pelvis'])\nif spine_curvature_angle > 15.0:\n    flag_early_lameness_warning(cow_id)"
      },
      {
        "id": "agr-07",
        "title": "Smartphone-Based Plant Leaf Disease Diagnostic Mobile App",
        "difficulty": "Beginner",
        "summary": "Offline-capable lightweight vision classifier enabling smallholder rural farmers to snap leaf photos and receive instant diagnosis and treatment advice.",
        "problem": "Smallholder farmers in remote regions have zero access to agricultural extension officers and often misidentify fungal vs bacterial pathogens.",
        "ai_technique": "MobileNetV4 / EfficientNet-Lite with Quantization-Aware Training (INT8)",
        "tech_stack": [
          "TensorFlow Lite / Flutter",
          "PyTorch",
          "ONNX",
          "Python"
        ],
        "data_pipeline": "PlantVillage dataset (54,306 images of 14 crop species and 38 diseases) + field validation imagery.",
        "hardware_target": "Entry-level Android smartphone (works completely offline without cellular internet).",
        "impact": "Diagnoses 38 distinct crop diseases in under 80ms on a $60 smartphone with 97.4% top-1 accuracy in localized native languages.",
        "code_snippet": "// Flutter / TFLite Flutter implementation\nvar recognitions = await Tflite.runModelOnImage(\n  path: imageFile.path,\n  numResults: 3,\n  threshold: 0.5,\n  imageMean: 127.5,\n  imageStd: 127.5\n);"
      },
      {
        "id": "agr-08",
        "title": "Robotic Fruit Harvesting Soft Gripper Vision and Ripeness Estimation",
        "difficulty": "Advanced",
        "summary": "RGB-D camera vision system estimating 3D fruit pose, branch occlusions, and brix sweetness/ripeness for robotic strawberry and tomato harvesting.",
        "problem": "Labor shortages during picking seasons cause crops to rot on the vine, while rigid mechanical harvesters bruise soft fruits.",
        "ai_technique": "Mask R-CNN + PointNet++ for 6D Grasping Pose Estimation",
        "tech_stack": [
          "ROS 2",
          "PyTorch",
          "Intel RealSense SDK",
          "MoveIt 2",
          "Open3D"
        ],
        "data_pipeline": "In-orchard RGB-D point clouds of fruit clusters under direct sunlight, shadows, and foliage occlusions.",
        "hardware_target": "Harvesting rover with UR5 robotic arm, Intel RealSense D435i camera, and soft pneumatic gripper.",
        "impact": "Picks ripe fruit with 94.2% grasp success and zero skin bruising at an average picking cycle time of 3.8 seconds per fruit.",
        "code_snippet": "# Estimate 3D grasp center and surface normal from depth point cloud\npoint_cloud = depth_to_pointcloud(rgbd_frame)\nfruits = segment_ripe_fruits(rgb_frame)\ngrasp_poses = pointnet_grasp_evaluator(point_cloud, fruits)"
      },
      {
        "id": "agr-09",
        "title": "Autonomous Greenhouse Climate and CO2 Control via Reinforcement Learning",
        "difficulty": "Intermediate",
        "summary": "Deep RL controller adjusting ventilation louvers, LED grow lighting spectrum, misting, and CO2 injection inside high-tech glass greenhouses.",
        "problem": "Static climate rules in commercial greenhouses fail to adapt to fluctuating outdoor weather, burning unnecessary gas and electricity.",
        "ai_technique": "Deep Q-Networks (DQN) with Thermal Equilibrium Physics Constraints",
        "tech_stack": [
          "Ray RLlib",
          "EnergyPlus / GreenLight",
          "Python",
          "MQTT"
        ],
        "data_pipeline": "Autonomous Greenhouse Challenge (Wageningen University) telemetry datasets.",
        "hardware_target": "Greenhouse Climate PLC (Priva / Hoogendoorn) integrated with IoT Edge server.",
        "impact": "Boosts marketable crop biomass yield by 18.2% while trimming heating energy and supplemental CO2 consumption by 24.7%.",
        "code_snippet": "class GreenhouseEnv(gym.Env):\n    def step(self, action):\n        # Action: [ventilation_pct, led_intensity, misting_active, co2_ppm]\n        next_state, reward, done, info = self.simulate_physics(action)\n        return next_state, reward, done, info"
      },
      {
        "id": "agr-10",
        "title": "Machine Learning for Grain Quality Sorting and Mycotoxin Detection",
        "difficulty": "Intermediate",
        "summary": "Short-wave infrared (SWIR) hyperspectral sorting line classifying aflatoxin and mold-infected cereal grains at 40 kernels per second.",
        "problem": "Carcinogenic aflatoxin-contaminated grains contaminate entire storage silos, posing lethal risks to human and animal food chains.",
        "ai_technique": "Support Vector Machines (SVM) & 1D-CNN Spectral Signature Matching",
        "tech_stack": [
          "Scikit-Learn",
          "PyTorch",
          "C++",
          "Opto-Mechanical Ejector Driver"
        ],
        "data_pipeline": "SWIR reflectance spectra (900nm - 1700nm) of corn, wheat, and peanut kernels.",
        "hardware_target": "High-speed industrial vibratory chute with pneumatic air ejection nozzles.",
        "impact": "Identifies contaminated kernels with 96.8% accuracy, rejecting toxic grain with 99.9% containment before food supply processing.",
        "code_snippet": "# Kernel classification on industrial conveyor belt\nspectrum = swir_sensor.read_spectrum(kernel_trigger)\nprob_aflatoxin = mycotoxin_classifier.predict_proba([spectrum])[0][1]\nif prob_aflatoxin > 0.05:\n    fire_pneumatic_nozzle(delay_ms=12)"
      }
    ]
  },
  "business": {
    "name": "Business, Finance & Management",
    "short": "Business",
    "icon": "\ud83d\udcc8",
    "color": "#8b5cf6",
    "tagline": "Revolutionizing algorithmic finance, supply chain intelligence, fraud prevention, and customer lifetime optimization.",
    "description": "AI in modern business drives hyper-personalized customer journeys, real-time algorithmic risk modeling, corporate governance automation, and supply chain resiliency. Solutions bridge quantitative finance, large language model document synthesis, and automated forecasting.",
    "projects": [
      {
        "id": "biz-01",
        "title": "Real-Time Financial Credit Card Fraud Detection with Graph Neural Networks",
        "difficulty": "Advanced",
        "summary": "Heterogeneous Graph Neural Network identifying coordinated credit card fraud rings and identity syndicates in sub-50ms transaction windows.",
        "problem": "Traditional tabular machine learning evaluates transactions in isolation, failing to spot distributed fraud rings that disguise identities across multiple accounts.",
        "ai_technique": "Heterogeneous Graph Transformer (HGT) & Relational GCN",
        "tech_stack": [
          "PyTorch Geometric",
          "DGL",
          "Apache Kafka",
          "Redis",
          "FastAPI"
        ],
        "data_pipeline": "Synthetic and anonymized banking graph datasets (nodes: User, Card, Device, Merchant, IP; edges: TRANSACTED, USED_DEVICE).",
        "hardware_target": "Distributed high-throughput banking gateway cluster.",
        "impact": "Catches 43% more organized fraud ring transactions than standalone tree models, while maintaining sub-0.01% false decline rates.",
        "code_snippet": "import torch_geometric.transforms as T\nfrom torch_geometric.nn import HGTConv\nclass FraudHGT(nn.Module):\n    def __init__(self, metadata):\n        super().__init__()\n        self.conv1 = HGTConv(in_channels=64, out_channels=32, metadata=metadata, heads=4)\n        self.classifier = nn.Linear(32, 2)"
      },
      {
        "id": "biz-02",
        "title": "LLM-Powered Multi-Agent Financial Report and 10-K SEC Due Diligence",
        "difficulty": "Intermediate",
        "summary": "Specialized RAG system with autonomous analytical agents parsing balance sheets, MD&A disclosures, and cash flow statements for hedge funds.",
        "problem": "Financial analysts spend hundreds of hours manually combing through 200+ page SEC 10-K filings to extract subtle accounting shifts and litigation risks.",
        "ai_technique": "Retrieval-Augmented Generation (RAG) + Hybrid Dense-Sparse Search + ReAct Multi-Agent",
        "tech_stack": [
          "LangChain / LlamaIndex",
          "ChromaDB / Qdrant",
          "Hugging Face / OpenAI API",
          "FastAPI",
          "Streamlit"
        ],
        "data_pipeline": "SEC EDGAR 10-K and 10-Q filing archives spanning S&P 500 corporations.",
        "hardware_target": "Cloud Enterprise Microservice.",
        "impact": "Reduces financial due diligence reporting time from 3 days to 45 minutes, with zero hallucinations via strict grounded attribution citations.",
        "code_snippet": "from llama_index.core import VectorStoreIndex, SimpleDirectoryReader\nfrom llama_index.core.tools import QueryEngineTool\n# Build hierarchical query engine comparing year-over-year balance sheet line items\nengine = index.as_query_engine(similarity_top_k=5, response_mode='tree_summarize')"
      },
      {
        "id": "biz-03",
        "title": "Algorithmic Market Making and High-Frequency Trading via Deep RL",
        "difficulty": "Advanced",
        "summary": "Deep Q-Learning and Actor-Critic agents managing bid-ask spread placement and inventory risk across limit order books (LOB).",
        "problem": "Market makers face adverse selection risk when trading against informed market participants, suffering heavy inventory losses during market crashes.",
        "ai_technique": "Recurrent DDPG / PPO with Limit Order Book Level-2 Feature Embeddings",
        "tech_stack": [
          "PyTorch",
          "Gymnasium",
          "C++ Trading Engine",
          "ZeroMQ",
          "Pandas"
        ],
        "data_pipeline": "Millisecond-level Limit Order Book (LOB) tick data from NASDAQ and CME futures.",
        "hardware_target": "Colocated Linux bare-metal trading server.",
        "impact": "Improves Sharpe ratio by 34% and reduces maximum inventory drawdown during flash volatility events compared to Avellaneda-Stoikov model.",
        "code_snippet": "# Process LOB Level 2 depth: 10 bids, 10 asks (price and volume)\nlob_state = extract_order_book_features(market_feed)\naction = rl_agent.get_action(lob_state) # [bid_offset, ask_offset, cancel_threshold]"
      },
      {
        "id": "biz-04",
        "title": "Supply Chain Demand Forecasting with Hierarchical Spatio-Temporal Transformers",
        "difficulty": "Intermediate",
        "summary": "Deep neural temporal architecture predicting SKU-level retail demand across 5,000 stores, accounting for promotions, local weather, and macro indicators.",
        "problem": "Bullwhip effect and stockouts cause billions in lost sales, while overstocking leads to heavy inventory depreciation and warehouse storage overhead.",
        "ai_technique": "Temporal Fusion Transformer (TFT) with Hierarchical Coherent Reconciliation",
        "tech_stack": [
          "PyTorch Forecasting",
          "Ray Tune",
          "PostgreSQL",
          "Snowflake",
          "Docker"
        ],
        "data_pipeline": "Retail sales history (Kaggle Corporaci\u00f3n Favorita / Walmart M5 benchmark) + regional economic indicators.",
        "hardware_target": "Cloud Enterprise Data Platform.",
        "impact": "Reduces SKU-level forecast Root Mean Squared Scaled Error (RMSSE) by 21.3%, eliminating over $4M in yearly perishable waste for retail pilots.",
        "code_snippet": "from pytorch_forecasting.models import TemporalFusionTransformer\ntft = TemporalFusionTransformer.from_dataset(\n    training_dataset, learning_rate=0.03, hidden_size=64, attention_head_size=4, dropout=0.2, loss=QuantileLoss()\n)"
      },
      {
        "id": "biz-05",
        "title": "Automated Customer Churn Prediction and Dynamic Retention Incentive Engine",
        "difficulty": "Beginner",
        "summary": "Survival analysis and gradient boosting models calculating real-time hazard rates and recommending optimal personalized discount offers.",
        "problem": "Acquiring a new customer costs 5 to 7 times more than retaining an existing one. Generic blanket discounts erode corporate profit margins.",
        "ai_technique": "Cox Proportional Hazards + LightGBM + Uplift Modeling (Causal ML)",
        "tech_stack": [
          "LightGBM",
          "CausalML",
          "Lifelines",
          "Scikit-Learn",
          "FastAPI"
        ],
        "data_pipeline": "Telco and SaaS subscription logs: daily usage time, customer support ticket frequency, billing history, NPS survey ratings.",
        "hardware_target": "CRM Cloud Server (Salesforce / HubSpot API integration).",
        "impact": "Increases net subscriber retention by 18.6% while cutting incentive promotional budget waste by 41% via targeted uplift modeling.",
        "code_snippet": "from causalml.inference.meta import BaseXLearner\n# Estimate Individual Treatment Effect (ITE) of a 20% discount offer\nxlearner = BaseXLearner(learner=LGBMRegressor())\nite = xlearner.fit_predict(X=features, Treatment=coupon_sent, y=retained_next_month)"
      },
      {
        "id": "biz-06",
        "title": "Multilingual E-Commerce Review Aspect-Based Sentiment Analysis (ABSA)",
        "difficulty": "Intermediate",
        "summary": "Fine-tuned DeBERTa model parsing millions of marketplace reviews to extract granular sentiment scores across specific product features.",
        "problem": "Star ratings are too coarse. Product managers cannot tell if a 3-star rating is due to shipping delays, battery life, packaging, or customer service.",
        "ai_technique": "Aspect-Based Sentiment Analysis (ABSA) via DeBERTa-v3 & Dependency Parsing",
        "tech_stack": [
          "Hugging Face Transformers",
          "spaCy",
          "PyTorch",
          "Elasticsearch",
          "Streamlit"
        ],
        "data_pipeline": "Amazon Customer Reviews & Trustpilot multi-category customer reviews in English, Spanish, and German.",
        "hardware_target": "Cloud GPU Inference Worker (NVIDIA L4).",
        "impact": "Categorizes feedback across 18 distinct product attributes with 91.8% F1-score, cutting product defect discovery cycles from months to days.",
        "code_snippet": "from transformers import pipeline\nabsa_classifier = pipeline('sentiment-analysis', model='yangheng/deberta-v3-base-absa-v1.1')\n# 'The battery life is stellar but the screen scratches easily'\n# -> {'Aspect': 'battery life', 'Sentiment': 'Positive'}, {'Aspect': 'screen', 'Sentiment': 'Negative'}"
      },
      {
        "id": "biz-07",
        "title": "Enterprise B2B Dynamic Pricing Optimization via Contextual Bandits",
        "difficulty": "Advanced",
        "summary": "Reinforcement learning contextual bandit algorithm balancing win rate vs gross profit margin in real-time quote generation.",
        "problem": "Sales teams rely on static spreadsheet cost-plus models or gut instinct, underpricing high-willingness-to-pay deals and losing borderline bids.",
        "ai_technique": "Contextual Multi-Armed Bandits (LinUCB / Thompson Sampling)",
        "tech_stack": [
          "Vowpal Wabbit",
          "PyTorch",
          "FastAPI",
          "PostgreSQL",
          "React.js"
        ],
        "data_pipeline": "Historical enterprise RFP responses, deal sizes, competitor presence, client industry, and macroeconomic factors.",
        "hardware_target": "Enterprise ERP / CPQ (Configure, Price, Quote) Cloud Server.",
        "impact": "Generates a 3.4% lift in overall gross profit margins without hurting aggregate deal conversion win rates.",
        "code_snippet": "# Linear Upper Confidence Bound (LinUCB) for price tier recommendation\narm_scores = [np.dot(theta[arm], context_vector) + alpha * np.sqrt(np.dot(context_vector, A_inv[arm]).dot(context_vector)) for arm in price_tiers]\nchosen_price_tier = np.argmax(arm_scores)"
      },
      {
        "id": "biz-08",
        "title": "Automated Invoice and Purchase Order Document AI via LayoutLMv3",
        "difficulty": "Beginner",
        "summary": "Multimodal Document AI combining text OCR, spatial 2D coordinates, and visual layout features to extract tabular invoice line items.",
        "problem": "Accounts payable departments manually type vendor invoice line items, suffering human data entry error rates and delayed supplier payment terms.",
        "ai_technique": "LayoutLMv3 Multimodal Transformer (Text + Layout + Image)",
        "tech_stack": [
          "Hugging Face Transformers",
          "Tesseract / PaddleOCR",
          "PyMuPDF",
          "FastAPI"
        ],
        "data_pipeline": "CORD (Consolidated Receipt Dataset) and FUNSD invoice/receipt benchmark documents.",
        "hardware_target": "On-premise / Private Cloud Document Processing Server.",
        "impact": "Achieves 97.2% key-value extraction accuracy across unseen international invoice templates, processing invoices in 650ms.",
        "code_snippet": "from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor\nprocessor = LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)\nencoding = processor(image, words, boxes=boxes, word_labels=labels, return_tensors='pt')"
      },
      {
        "id": "biz-09",
        "title": "Corporate ESG (Environmental, Social, Governance) Greenwashing Detector",
        "difficulty": "Intermediate",
        "summary": "Zero-shot NLP classifier and factual consistency auditor auditing corporate sustainability claims against public satellite emissions and regulatory fines.",
        "problem": "Investors are misled by vague 'carbon neutral' marketing pledges that lack verifiable emission reduction initiatives (greenwashing).",
        "ai_technique": "Natural Language Inference (NLI) & Multi-Source Cross-Verification Transformer",
        "tech_stack": [
          "PyTorch",
          "Hugging Face",
          "BeautifulSoup4",
          "Pandas",
          "Plotly"
        ],
        "data_pipeline": "Corporate annual sustainability reports cross-referenced against EPA TRI (Toxics Release Inventory) and Climate TRACE satellite emissions.",
        "hardware_target": "ESG Investment Analytics Web Platform.",
        "impact": "Scores corporate ESG statements with an 88.5% validation correlation against factual regulatory enforcement actions, flagging deceptive metrics.",
        "code_snippet": "nli_model = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')\nclaim = 'We reduced scope 1 greenhouse gas emissions by 40% in our manufacturing sites.'\nlabels = ['verifiable empirical claim', 'vague marketing fluff', 'contradicted by regulatory data']"
      },
      {
        "id": "biz-10",
        "title": "Hyper-Personalized Recommendation Engine with Deep Autoencoders & Two-Tower DNN",
        "difficulty": "Intermediate",
        "summary": "Industrial two-tower deep neural network serving millisecond-latency personalized product rankings across 10 million active users.",
        "problem": "Collaborative filtering fails on sparse long-tail items, and naive matrix factorization cannot incorporate rich user real-time session context.",
        "ai_technique": "Two-Tower Deep Neural Network (User Tower + Item Tower) & Approximate Nearest Neighbor (ANN) Search",
        "tech_stack": [
          "TensorFlow Recommenders (TFRS)",
          "ScaNN / Faiss",
          "Redis",
          "Triton Inference Server"
        ],
        "data_pipeline": "E-commerce user clickstreams, search queries, cart additions, purchase conversions, and product metadata.",
        "hardware_target": "NVIDIA Triton Inference Server with GPU acceleration.",
        "impact": "Drives a 26.8% increase in Click-Through Rate (CTR) and an 18.2% uplift in checkout cart values compared to popularity baselines.",
        "code_snippet": "import tensorflow_recommenders as tfrs\nclass TwoTowerModel(tfrs.Model):\n    def __init__(self, user_model, item_model):\n        super().__init__()\n        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=items.batch(128)))\n        self.user_model = user_model\n        self.item_model = item_model"
      }
    ]
  },
  "healthcare": {
    "name": "Healthcare & Biomedical Sciences",
    "short": "Healthcare",
    "icon": "\ud83c\udfe5",
    "color": "#ef4444",
    "tagline": "Pioneering clinical diagnostics, biomedical AI, drug discovery, and medical digital twins.",
    "description": "Healthcare AI merges biomedical imaging, genomic sequencing, physiological sensor monitoring, and clinical decision support. Rigorous requirements demand high interpretability, privacy-preserving federated architectures, and strict HIPAA compliance.",
    "projects": [
      {
        "id": "hlth-01",
        "title": "Deep Learning Detection of Diabetic Retinopathy from Fundus Photography",
        "difficulty": "Intermediate",
        "summary": "Ensemble of EfficientNet and Vision Transformers grading diabetic retinopathy severity into 5 clinical stages with Grad-CAM lesion heatmaps.",
        "problem": "Diabetic retinopathy is the leading cause of preventable blindness in adults. Millions lack access to expert ophthalmologists for early screening.",
        "ai_technique": "EfficientNet-B4 + Grad-CAM Visual Explainability",
        "tech_stack": [
          "PyTorch",
          "Torchvision",
          "OpenCV",
          "FastAPI",
          "React"
        ],
        "data_pipeline": "EyePACS and Messidor-2 fundus photograph datasets (35,000+ annotated retinal images).",
        "hardware_target": "Portable handheld fundus camera with onboard edge processor.",
        "impact": "Exceeds 95.3% sensitivity and 96.1% specificity, matching board-certified retinal specialists in clinical diagnostic benchmarks.",
        "code_snippet": "# Generate Grad-CAM attribution heatmap for ophthalmologist review\nheatmap = grad_cam(model, retinal_image, target_layer='features.8')\noverlay = cv2.addWeighted(retinal_image, 0.6, heatmap, 0.4, 0)\n# Highlights microaneurysms, hemorrhages, and hard exudates"
      },
      {
        "id": "hlth-02",
        "title": "3D Brain Tumor Segmentation from Multimodal MRI Scans",
        "difficulty": "Advanced",
        "summary": "3D U-Net (nnU-Net) segmenting enhancing tumor core, whole tumor, and peri-tumoral edema across T1, T1Gd, T2, and FLAIR MRI volumes.",
        "problem": "Manual slice-by-slice 3D contouring of glioblastoma multiforme takes radiologists over an hour per patient, delaying radiation therapy planning.",
        "ai_technique": "3D nnU-Net (no-new-Net) with Self-Configuring Preprocessing",
        "tech_stack": [
          "PyTorch",
          "nnU-Net framework",
          "SimpleITK",
          "MONAI",
          "Docker"
        ],
        "data_pipeline": "BraTS (Brain Tumor Segmentation Challenge) multi-institution 3D MRI volumes.",
        "hardware_target": "Hospital PACS Workstation with NVIDIA RTX 6000 Ada GPU.",
        "impact": "Achieves a Dice similarity coefficient of 0.91 on whole tumor segmentation, reducing neurosurgical radiation planning time by 85%.",
        "code_snippet": "from monai.networks.nets import UNet\nmodel = UNet(spatial_dims=3, in_channels=4, out_channels=3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))\n# Input: 4-channel 3D volume (T1, T1c, T2, FLAIR) -> Output: [Edema, Non-Enhancing Core, Enhancing Tumor]"
      },
      {
        "id": "hlth-03",
        "title": "Transformer-Based Clinical Note Summarization and ICD-10 Code Extraction",
        "difficulty": "Intermediate",
        "summary": "Fine-tuned Clinical-Longformer parsing unstructured electronic health records (EHR) to auto-extract diagnostic ICD-10-CM codes and discharge summaries.",
        "problem": "Physicians spend more than 2 hours in EHR documentation for every 1 hour of patient bedside care, driving massive clinical burnout.",
        "ai_technique": "Clinical-Longformer / BioLinkBERT with Multi-Label Hierarchy Head",
        "tech_stack": [
          "Hugging Face Transformers",
          "PyTorch",
          "FHIR API",
          "PostgreSQL",
          "spaCy"
        ],
        "data_pipeline": "MIMIC-IV (Medical Information Mart for Intensive Care) de-identified clinical notes.",
        "hardware_target": "HIPAA-compliant hospital internal secure server.",
        "impact": "Automates 78% of medical billing code drafting with 89.4% Micro-F1 score, returning 90 minutes of daily doctor time to patient care.",
        "code_snippet": "from transformers import AutoTokenizer, AutoModelForSequenceClassification\ntokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')\n# Extracts ICD-10 diagnosis codes from 4,096-token patient doctor progress notes"
      },
      {
        "id": "hlth-04",
        "title": "Real-Time Edge-AI Arrhythmia Detection from Wearable Single-Lead ECG",
        "difficulty": "Beginner",
        "summary": "Lightweight 1D ResNet model classifying atrial fibrillation, ventricular ectopic beats, and sinus bradycardia directly on smartwatch firmware.",
        "problem": "Paroxysmal atrial fibrillation occurs unpredictably; traditional 24-hour Holter monitors miss sporadic episodes, leaving patients at stroke risk.",
        "ai_technique": "Residual 1D Convolutional Neural Network with Squeeze-and-Excitation Blocks",
        "tech_stack": [
          "TensorFlow Lite for Microcontrollers",
          "C++",
          "Python",
          "WFDB toolkit"
        ],
        "data_pipeline": "MIT-BIH Arrhythmia Database & PhysioNet Challenge records.",
        "hardware_target": "Nordic nRF52840 BLE SoC / Silicon Labs EFR32 wearable smartwatch.",
        "impact": "Achieves 98.2% Afib detection sensitivity consuming only 18 KB RAM and 0.4 mJ energy per 10-second ECG classification strip.",
        "code_snippet": "// Lightweight 1D convolution kernel on wearable embedded microcontroller\nint8_t ecg_quantized[250]; // 10 seconds sampled at 25 Hz\ntf_interpreter->Invoke();\nint8_t afib_prob = tf_interpreter->output(0)->data.int8[AFIB_INDEX];"
      },
      {
        "id": "hlth-05",
        "title": "Graph Neural Networks for Small Molecule Drug-Target Binding Affinity",
        "difficulty": "Advanced",
        "summary": "Equivariant Graph Neural Network (EGNN) predicting 3D binding affinity (pKd / pKi) between candidate drug molecules and viral target proteins.",
        "problem": "Wet-lab high-throughput drug screening costs millions of dollars per compound and takes years to identify viable lead candidates.",
        "ai_technique": "E(n)-Equivariant Graph Neural Network (EGNN) on 3D Molecular Coordinates",
        "tech_stack": [
          "PyTorch Geometric",
          "RDKit",
          "BioPython",
          "OpenMM",
          "PyMOL"
        ],
        "data_pipeline": "PDBbind v2020 dataset (curated 3D protein-ligand structural complexes with experimental binding constants).",
        "hardware_target": "High-Performance GPU Cluster (NVIDIA A100 / H100).",
        "impact": "Screens 100,000 candidate molecules in 12 hours with a Pearson correlation of 0.84 against experimental binding affinities.",
        "code_snippet": "from rdkit import Chem\nfrom torch_geometric.data import Data\n# Transform SMILES to 3D atomic graph with electrostatic and covalent edge features\nmol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')\npredicted_affinity = egnn_model(protein_graph, ligand_graph)"
      },
      {
        "id": "hlth-06",
        "title": "Histopathology Whole Slide Cancer Detection via Multiple Instance Learning",
        "difficulty": "Advanced",
        "summary": "Attention-based Deep Multiple Instance Learning (CLAM) detecting metastatic breast cancer in gigapixel lymph node biopsy digital slides.",
        "problem": "A single digital pathology Whole Slide Image (WSI) contains over 100,000 x 100,000 pixels. Scanning slides manually under microscopes is exhausting.",
        "ai_technique": "Clustering-constrained Attention Multiple Instance Learning (CLAM)",
        "tech_stack": [
          "OpenSlide",
          "PyTorch",
          "Torchvision",
          "FastAPI",
          "React OpenSeadragon"
        ],
        "data_pipeline": "CAMELYON16 and CAMELYON17 lymph node histopathology challenge datasets.",
        "hardware_target": "Hospital Pathology Department Workstation with high-speed NVMe storage.",
        "impact": "Achieves an AUC of 0.985 in identifying metastatic tissue patches, highlighting suspect biopsy micro-regions in seconds.",
        "code_snippet": "# Extract 256x256 tiles, embed via ResNet50, then aggregate via attention gating\nfeatures = resnet50_feature_extractor(wsi_patches)\nattention_weights, slide_prediction = clam_model(features)\n# Generates gigapixel attention heatmap overlay"
      },
      {
        "id": "hlth-07",
        "title": "Non-Invasive Continuous Blood Pressure Estimation from Photoplethysmogram (PPG)",
        "difficulty": "Intermediate",
        "summary": "Dual-channel CNN-LSTM estimating systolic and diastolic arterial blood pressure continuously from optical fingertip PPG sensors.",
        "problem": "Cuff-based blood pressure measurement is intermittent and disrupts sleep, while arterial line catheterization is invasive and carries infection risk.",
        "ai_technique": "1D-CNN + Bidirectional GRU + Attention Layer",
        "tech_stack": [
          "PyTorch",
          "SciPy",
          "Pandas",
          "Matplotlib",
          "Streamlit"
        ],
        "data_pipeline": "MIMIC-II clinical database matching continuous fingertip PPG waveforms with invasive arterial blood pressure lines.",
        "hardware_target": "Wearable smart ring or pulse oximeter with optical green/infrared LEDs.",
        "impact": "Satisfies the British Hypertension Society (BHS) Grade A standard with Mean Absolute Error below 4.5 mmHg for both SBP and DBP.",
        "code_snippet": "# Calculate pulse arrival time (PAT) and pulse transit time (PTT) from PPG derivatives\nfirst_derivative = np.diff(ppg_signal)\nsecond_derivative = np.diff(first_derivative)\nfeatures = np.stack([ppg_signal[2:], first_derivative[1:], second_derivative], axis=-1)\nsbp, dbp = model(torch.tensor(features))"
      },
      {
        "id": "hlth-08",
        "title": "Federated Learning for Privacy-Preserving Collaborative Healthcare AI",
        "difficulty": "Advanced",
        "summary": "Federated averaging framework training multi-hospital COVID-19 and pneumonia chest X-ray classifiers without sharing patient raw records.",
        "problem": "Strict medical privacy laws (HIPAA, GDPR) forbid hospitals from aggregating sensitive patient data to a central cloud server.",
        "ai_technique": "Federated Averaging (FedAvg) + Differential Privacy (DP-SGD)",
        "tech_stack": [
          "Flower (flwr)",
          "PyTorch",
          "Opacus (Differential Privacy)",
          "Docker"
        ],
        "data_pipeline": "Distributed partition of NIH Chest X-ray14 dataset across 5 simulated hospital client nodes.",
        "hardware_target": "Multi-node hospital on-premise compute servers connected over secure TLS.",
        "impact": "Reaches 94.7% diagnostic AUC across 14 thoracic pathologies while keeping raw patient records 100% within hospital firewalls.",
        "code_snippet": "import flwr as fl\nclass HospitalClient(fl.client.NumPyClient):\n    def fit(self, parameters, config):\n        self.set_parameters(parameters)\n        train_with_differential_privacy(self.model, self.train_loader, epochs=1)\n        return self.get_parameters(), len(self.train_loader), {}"
      },
      {
        "id": "hlth-09",
        "title": "AI Sepsis Early Warning System in Intensive Care Units (ICU)",
        "difficulty": "Intermediate",
        "summary": "Temporal Gradient Boosting and Weibull survival model providing continuous hourly sepsis risk scoring 6 hours before clinical septic shock onset.",
        "problem": "Sepsis mortality increases by 8% for every single hour of delayed antibiotic treatment. Clinical scoring tools (SOFA, SIRS) are reactive.",
        "ai_technique": "XGBoost + Temporal Windowing + Shapley Explanation Alerts",
        "tech_stack": [
          "XGBoost",
          "Python",
          "SQLAlchemy",
          "FastAPI",
          "FHIR / HL7"
        ],
        "data_pipeline": "PhysioNet Computing in Cardiology Challenge 2019 dataset (40,000 ICU patient hourly vital signs and lab results).",
        "hardware_target": "Hospital ICU Central Telemetry Server.",
        "impact": "Predicts sepsis onset 6 hours prior to clinical diagnosis with an AUC of 0.87, giving nurses crucial time to initiate resuscitation protocols.",
        "code_snippet": "# Real-time bedside vital feature windowing (HR, MAP, Temp, SpO2, WBC, Lactate)\npatient_vitals_6h = fetch_icu_telemetry(bed_id, window_hours=6)\nrisk_score = sepsis_model.predict_proba(patient_vitals_6h)[0][1]\nif risk_score > 0.65:\n    dispatch_icu_pager_alert(bed_id, 'High Sepsis Risk: Check Lactate & Blood Cultures')"
      },
      {
        "id": "hlth-10",
        "title": "Computer Vision Surgical Instrument Tracking and Phase Recognition",
        "difficulty": "Intermediate",
        "summary": "Spatio-temporal neural network recognizing surgical steps and instrument movement during laparoscopic cholecystectomy procedures.",
        "problem": "Operating room efficiency and resident surgical training lack objective, quantifiable metrics for procedural competency assessment.",
        "ai_technique": "ResNet-50 + Temporal Convolutional Network (TCN)",
        "tech_stack": [
          "PyTorch",
          "OpenCV",
          "FFmpeg",
          "FastAPI"
        ],
        "data_pipeline": "Cholec80 surgical video benchmark dataset (80 laparoscopic cholecystectomy procedures annotated with 7 phases and tool presence).",
        "hardware_target": "Endoscopy Tower Video Output + Medical Grade Edge PC.",
        "impact": "Recognizes surgical phases with 92.6% accuracy, automatically logging surgical duration metrics and alerting staff to phase transitions.",
        "code_snippet": "# Process laparoscopic endoscopic video feed\nframe_features = spatial_cnn(video_frame)\nphase_logits = temporal_tcn(frame_features)\ncurrent_surgical_phase = phases[torch.argmax(phase_logits)]\n# e.g., 'CalotTriangleDissection', 'ClippingAndCutting', 'GallbladderPackaging'"
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
        synergy: "Smart Grid Load Forecasting, Drone Substation Thermography, Edge Microgrid Optimizers."
      },
      ece: {
        name: "Electronics & Communication Engineering (ECE)",
        shortName: "Telecom & ECE",
        icon: "📡",
        color: "#0284c7",
        bg: "rgba(2, 132, 199, 0.2)",
        glow: "rgba(2, 132, 199, 0.45)",
        synergy: "Beamforming Transformers, Neuromorphic RF Signal Demodulation, 6G Latency Predictors."
      },
      civil: {
        name: "Civil & Structural Engineering",
        shortName: "Civil Engineering",
        icon: "🏗️",
        color: "#0ea5e9",
        bg: "rgba(14, 165, 233, 0.2)",
        glow: "rgba(14, 165, 233, 0.45)",
        synergy: "Structural Defect Computer Vision, Digital Twin Physics-Informed NNs, Smart Traffic Flow."
      },
      agriculture: {
        name: "Agriculture & Smart Farming",
        shortName: "Smart Agriculture",
        icon: "🌱",
        color: "#2563eb",
        bg: "rgba(37, 99, 235, 0.2)",
        glow: "rgba(37, 99, 235, 0.45)",
        synergy: "Drone Multispectral Weed Segmentation, Soil Nitrogen Transformers, Yield AI."
      },
      business: {
        name: "Business, Finance & Management",
        shortName: "FinTech & Business",
        icon: "📈",
        color: "#6366f1",
        bg: "rgba(99, 102, 241, 0.2)",
        glow: "rgba(99, 102, 241, 0.45)",
        synergy: "High-Frequency Algorithmic Execution, Graph Neural Fraud Networks, Automated Credit Risk."
      },
      healthcare: {
        name: "Healthcare & Biomedical Sciences",
        shortName: "Healthcare & Bio",
        icon: "🏥",
        color: "#38bdf8",
        bg: "rgba(56, 189, 248, 0.2)",
        glow: "rgba(56, 189, 248, 0.45)",
        synergy: "Multi-Modal Oncology Diagnosis, Protein Sequence Transformers, ICU Telemetry Early Warning."
      }
    };

    // WHY COURSES ARE ESSENTIAL FOR EACH DISCIPLINE
    const essentialDisciplineData = {
      eee: {
        name: "Electrical & Electronics Engineering (EEE)",
        icon: "⚡",
        deptKey: "eee",
        shiftSummary: "Modern power systems are shifting from steady-state rotating generators to dynamic, non-linear inverter-based renewable microgrids. Classical deterministic differential equations can no longer solve real-time grid stabilization and battery degradation at sub-millisecond speeds.",
        python: {
          title: "Python for Electrical Systems",
          desc: "Automates numerical power flow solvers, interfaces directly with SCADA and oscilloscope hardware (PySerial/VISA), parses high-frequency phasor telemetry (PMUs), and powers signal processing pipelines (SciPy, PyWavelets).",
          keySkills: "SPICE simulation scripting, hardware telemetry parsing, real-time power math."
        },
        ml: {
          title: "Machine Learning for Grid & Energy",
          desc: "Predicts non-linear solar irradiance and wind farm output from weather feeds, models battery State-of-Health (SoH) degradation over charge cycles, and classifies substation equipment anomalies using tree ensembles and regression.",
          keySkills: "Renewable energy forecasting, battery cycle regression, transformer predictive maintenance."
        },
        dl: {
          title: "Deep Learning for High-Speed Waveforms",
          desc: "Processes microsecond transient electrical waveforms with 1D-CNNs for fault localization, deploys Physics-Informed Neural Networks (PINNs) for non-linear power flow, and performs drone thermal inspection of high-voltage transmission lines.",
          keySkills: "Transient stability prediction, power inverter harmonics filtering, aerial thermography vision."
        },
        roiMetric: "35% reduction in unplanned grid transformer outages; 18% improvement in renewable power dispatch efficiency."
      },
      ece: {
        name: "Electronics & Communication Engineering (ECE / Telecom)",
        icon: "📡",
        deptKey: "ece",
        shiftSummary: "5G/6G communication systems operate at millimeter-wave frequencies with massive antenna arrays. Calculating optimal channel coefficients and beam angles using classical matrix inversion takes more time than the channel coherence window itself.",
        python: {
          title: "Python for RF & Embedded Systems",
          desc: "The global foundation for Software-Defined Radio (SDR) with GNU Radio, wireless packet capture analysis, RF link budget calculations, and scripting automated embedded firmware validation suites.",
          keySkills: "RF signal vector processing, SDR scripting, hardware-in-the-loop automation."
        },
        ml: {
          title: "Machine Learning for Spectrum & Channel",
          desc: "Enables dynamic cognitive radio spectrum access, channel state information (CSI) tracking across fading channels, network traffic burst prediction, and adaptive modulation and coding scheme (MCS) selection.",
          keySkills: "Dynamic spectrum sharing, QoS anomaly detection, channel coefficient regression."
        },
        dl: {
          title: "Deep Learning for 5G/6G & Massive MIMO",
          desc: "Optimizes hybrid analog-digital beamforming in millisecond deadlines using deep reinforcement learning, decodes distorted non-linear signals with neural autoencoders, and fingerprints wireless transmitters for zero-trust physical-layer security.",
          keySkills: "Massive MIMO beamforming, deep channel estimation, neuromorphic RF fingerprinting."
        },
        roiMetric: "4x increase in spectral efficiency; <5ms latency in adaptive multi-user beam steering."
      },
      civil: {
        name: "Civil & Structural Engineering",
        icon: "🏗️",
        deptKey: "civil",
        shiftSummary: "Aging municipal infrastructure cannot be safeguarded by periodic manual visual inspections. Continuous structural health monitoring demands computer vision drones and real-time vibration telemetry analytics.",
        python: {
          title: "Python for Structural Data & BIM",
          desc: "Automates OpenSees and ANSYS finite element analyses, queries Building Information Modeling (BIM) geometry databases, processes geotechnical borehole logs, and cleans urban sensor telemetry.",
          keySkills: "FEA automation, geotechnical data ingestion, CAD/BIM geometry scripting."
        },
        ml: {
          title: "Machine Learning for Materials & Geotech",
          desc: "Predicts 28-day concrete compressive strength based on chemical admixtures and curing humidity, models soil liquefaction susceptibility from CPT logs, and forecasts urban traffic congestion patterns.",
          keySkills: "Concrete strength regression, slope stability safety estimation, project delay risk modeling."
        },
        dl: {
          title: "Deep Learning for Drone Vision & Seismic AI",
          desc: "Executes automated millimeter-level concrete crack detection and spalling segmentation on bridge piers using drone vision (YOLO/U-Net), and forecasts structural seismic vibration response with LSTM neural networks.",
          keySkills: "Structural crack segmentation, seismic vibration time-series, satellite InSAR subsidence."
        },
        roiMetric: "90% reduction in bridge inspection inspection time; sub-millimeter crack detection precision."
      },
      agriculture: {
        name: "Agriculture & Smart Farming",
        icon: "🌱",
        deptKey: "agriculture",
        shiftSummary: "Blanket pesticide spraying and uniform water flooding deplete aquifers and contaminate soil. Modern robotic precision agriculture inspects each individual plant leaf to deliver micro-dosages.",
        python: {
          title: "Python for Agri-IoT & GIS Data",
          desc: "Aggregates agricultural micro-climate weather stations, parses multispectral drone flight geotags, processes raster satellite bands with Rasterio/GDAL, and scripts automated drip-irrigation pumps.",
          keySkills: "IoT sensor telemetry, GeoPandas spatial processing, drone geotag parsing."
        },
        ml: {
          title: "Machine Learning for Soil & Yield Prediction",
          desc: "Forecasts end-of-season crop yields by fusing meteorological variables with satellite NDVI features, optimizes fertilizer dosage (NPK) to avoid nitrogen runoff, and forecasts commodity market prices.",
          keySkills: "Multi-factor yield regression, soil nutrient optimization, agricultural price forecasting."
        },
        dl: {
          title: "Deep Learning for Robotic Harvesting & Pathology",
          desc: "Enables autonomous weeding robots to classify invasive species versus cash crops in real-time edge cameras, classifies foliar disease pathogens from smartphone leaf photos, and segments tree canopies.",
          keySkills: "Real-time weed vs crop vision, plant pathology leaf classification, robotic fruit grasping vision."
        },
        roiMetric: "45% reduction in herbicide chemical volume; 22% crop yield increase per acre."
      },
      business: {
        name: "Business, Finance & Quantitative Management",
        icon: "📈",
        deptKey: "business",
        shiftSummary: "Linear regression and Gaussian distribution assumptions fail catastrophically during market regime shifts and coordinated syndicate fraud attacks. Machine learning captures high-dimensional non-linear interactions.",
        python: {
          title: "Python for Quantitative Finance & ETL",
          desc: "The universal language on Wall Street and global FinTech. Cleans tick-by-tick market data, interfaces with Bloomberg/Refinitiv APIs, executes algorithmic backtesting, and generates automated regulatory compliance filings.",
          keySkills: "Time-series dataframes, algorithmic backtesting, financial REST APIs, ledger audits."
        },
        ml: {
          title: "Machine Learning for Risk & Fraud Scoring",
          desc: "Classifies high-velocity credit card transactions to prevent fraud, predicts customer loan default probabilities with explainable XGBoost, models customer churn, and segments corporate clients.",
          keySkills: "Real-time fraud classification, credit default scoring (PD/LGD), customer churn modeling."
        },
        dl: {
          title: "Deep Learning for High-Frequency Markets & NLP",
          desc: "Models non-linear Limit Order Book (LOB) microstructure dynamics using temporal convolutional networks, extracts market sentiment from earnings calls using financial LLMs, and maps money laundering syndicates with Graph Neural Networks.",
          keySkills: "Order book alpha modeling, financial sentiment NLP, anti-money laundering (AML) graph NNs."
        },
        roiMetric: "99.4% precision in zero-day financial fraud interception; $12M+ annual fraud loss mitigation."
      },
      healthcare: {
        name: "Healthcare, Biomedical & Life Sciences",
        icon: "🏥",
        deptKey: "healthcare",
        shiftSummary: "Clinical diagnosis generates petabytes of complex high-resolution imaging and continuous ICU telemetry. Clinicians require real-time AI co-pilots to flag early signs of deterioration hours before clinical crises occur.",
        python: {
          title: "Python for Bio-Signals & Clinical Data",
          desc: "Parses clinical DICOM radiology files, extracts multi-channel EEG/ECG biosignals (MNE-Python), processes genomic FASTA/BAM sequences (BioPython), and sanitizes electronic health records (EHR).",
          keySkills: "DICOM radiology parsing, bio-signal waveform filtering, clinical trial data pipelines."
        },
        ml: {
          title: "Machine Learning for Clinical Risk Stratification",
          desc: "Forecasts ICU patient septic shock onset 6 hours prior to clinical symptoms, calculates patient hospital readmission probabilities, and identifies biomarker signatures for cancer survival analysis.",
          keySkills: "Sepsis early warning, survival analysis (Cox/Random Survival Forests), patient risk scoring."
        },
        dl: {
          title: "Deep Learning for Medical Vision & Genomics",
          desc: "Segments 3D brain tumors and pulmonary nodules in MRI/CT scans using 3D U-Net, detects 12-lead ECG arrhythmias with CNN-LSTMs, and predicts macromolecular 3D protein structures using attention transformers.",
          keySkills: "Radiological 3D tumor segmentation, 12-lead ECG classification, protein structure modeling."
        },
        roiMetric: "6-hour advance warning for ICU sepsis; 97.8% diagnostic concordance in radiographic screening."
      },
      cs: {
        name: "Computer Science & Software Systems",
        icon: "💻",
        deptKey: "eee",
        shiftSummary: "Software engineering is evolving from manually hardcoded if-else business logic into adaptive, probabilistic foundation model architectures and autonomous self-debugging software agents.",
        python: {
          title: "Python for AI Systems & Microservices",
          desc: "The foundational language of the entire AI ecosystem. Powers FastAPI asynchronous microservices, distributed model training orchestration, automated test frameworks, and modern LLM agent tool calling.",
          keySkills: "Asynchronous microservices, clean OOP architecture, PyTest suites, Dockerized packaging."
        },
        ml: {
          title: "Machine Learning for Intelligent Software",
          desc: "Transforms software development from static if-else logic into data-driven decision engines: adaptive caching algorithms, search ranking systems, automated bug prediction, and user personalization engines.",
          keySkills: "Recommendation algorithms, predictive caching, anomaly detection in server logs."
        },
        dl: {
          title: "Deep Learning for Foundation Models & Agents",
          desc: "The architectural foundation of generative AI, large language models (LLMs), multimodal vision-language models, neural search vector embeddings, and autonomous software engineering agents.",
          keySkills: "Transformer self-attention mechanisms, GPU TensorRT acceleration, vector retrieval architectures."
        },
        roiMetric: "10x throughput enhancement on high-concurrency microservices; 60% faster test cycle completion."
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
        badge: "Foundation to Production Architecture",
        freeDesc: "Complete conceptual curriculum covering modern Python 3.12+, memory reference models, OOP design patterns, NumPy tensor vectorization, Pandas dataframes, and clean code architecture.",
        paidDesc: "15 Industrial Production Projects, Automated PyTest suites, Asynchronous Web Scraping, Custom PyPI package building, and 1-on-1 code reviews.",
        playlistUrl: "https://www.youtube.com/watch?v=kqtD5dpn9C8&list=PLKnIA16_Rmvb1m_BqV_eE9dJ3pW3_j_z1",
        embedUrl: "https://www.youtube.com/embed/kqtD5dpn9C8",
        playlistName: "Python for Data Science & AI Foundations",
        freeFeatures: [
          "Python 3.12+ Modern Syntax, Memory Model, Closures & Decorators",
          "Object-Oriented Design (OOP), Dunder Methods & Clean Architecture",
          "NumPy: Tensors, Broadcasting Rules, Vectorization & Matrix Math",
          "Pandas & Polars: High-Speed Wrangling, Cleaning & Timeseries",
          "Data Visualization: Matplotlib, Seaborn & Interactive Plotly",
          "Algorithmic Problem Solving & Big-O Computational Complexity"
        ],
        paidFeatures: [
          "15 Industrial Python Projects with automated PyTest test suites",
          "Asynchronous Web Scraping & Production ETL Data Ingestion",
          "High-Performance Python: Profiling, C-Extensions & Numba JIT acceleration",
          "Production Code Review: Linting (Ruff), Type Hints (Mypy), and Packaging",
          "Direct Discord channel access to the Instructor for 1-on-1 debugging",
          "Verified AI Connectra Python Developer Credential"
        ],
        modules: [
          { num: "MODULE 01", title: "Python Core: Memory, Functions, Lambdas & Generators", desc: "Data types, memory references, generator functions, closures, and functional programming constructs.", tier: "free" },
          { num: "MODULE 02", title: "Object-Oriented Programming & Clean Architecture", desc: "Classes, inheritance, polymorphism, dunder methods, abstract base classes, and design patterns in AI applications.", tier: "free" },
          { num: "MODULE 03", title: "NumPy: Tensor Operations & Vectorized Matrix Math", desc: "Multi-dimensional arrays, broadcasting rules, matrix dot products, and vectorized linear algebra routines.", tier: "free" },
          { num: "MODULE 04", title: "Pandas: Production Data Cleaning & Timeseries", desc: "DataFrames, indexing, missing data imputation, groupby aggregation, and temporal resample operations.", tier: "free" },
          { num: "MODULE 05", title: "Industrial Data Pipelines & Automated Testing", desc: "15 hands-on projects with PyTest, PyLint, Mypy type-checking, and CI/CD automated test pipelines.", tier: "paid" },
          { num: "MODULE 06", title: "High-Performance Concurrency & Package Publishing", desc: "AsyncIO, multiprocessing, packaging Python libraries on PyPI, and Dockerization for production microservices.", tier: "paid" }
        ]
      },
      ml: {
        id: "ml",
        title: "Machine Learning Masterclass (100 Days Series)",
        short: "Machine Learning",
        icon: "🤖",
        color: "#2563eb",
        glow: "rgba(37, 99, 235, 0.4)",
        price: "$49",
        badge: "Featuring CampusX 100-Day Masterclass Curriculum",
        freeDesc: "Complete 100-Day video lecture curriculum covering loss functions, mathematical derivations, convex optimization, gradient descent, tree algorithms, ensembles, and clustering.",
        paidDesc: "200 real-world capstone implementations across 10 disciplines with datasets, Dockerized FastAPI code, and capstone certification.",
        playlistUrl: "https://www.youtube.com/watch?v=ZftI2fEz0Fw&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH",
        embedUrl: "https://www.youtube.com/embed/ZftI2fEz0Fw?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH",
        playlistName: "100 Days of Machine Learning by CampusX",
        freeFeatures: [
          "100-Day Structured Video Series by CampusX (Full Free Open Access)",
          "Mathematical Optimization: Loss functions, convexity, and Gradient Descent derivations",
          "Feature Preprocessing: ColumnTransformer, FunctionTransformer & Scikit-Learn Pipelines",
          "Supervised Algorithms: Linear/Logistic, Decision Trees, Random Forests, XGBoost, SVM",
          "Unsupervised Algorithms: K-Means, DBSCAN, Hierarchical Clustering, and PCA",
          "Model Evaluation Metrics: Bias-Variance tradeoff, ROC-AUC, F1-Score, Cross-Validation"
        ],
        paidFeatures: [
          "200 Complete Capstone Projects across 10 academic disciplines",
          "Curated Domain Datasets (smart meters, RF IQ samples, satellite imagery, clinical records)",
          "FastAPI Microservices, Triton Inference Server & Docker containerization",
          "Step-by-step Jupyter and Google Colab Guided Notebooks for every blueprint",
          "Capstone Code Review, Portfolio Preparation, and Verified Industry Certification",
          "Exclusive Community Forum with direct instructor engineering support"
        ],
        modules: [
          { num: "DAYS 01-14", title: "ML Fundamentals, Problem Formulation & Lifecycle", desc: "What is Machine Learning? Supervised vs Unsupervised vs Reinforcement. ML project lifecycle, framing real-world business challenges, environment setup & Git.", tier: "free" },
          { num: "DAYS 15-23", title: "Data Ingestion, Web Scraping & Exploratory Data Analysis (EDA)", desc: "Reading CSV/JSON/SQL, web scraping with BeautifulSoup, descriptive statistics, univariate, bivariate & multivariate analysis, and automated Pandas Profiling.", tier: "free" },
          { num: "DAYS 24-34", title: "Feature Engineering, Encoders & Scikit-Learn Pipelines", desc: "Standardization, Normalization, Ordinal & One-Hot Encoding, ColumnTransformer, FunctionTransformer, PowerTransformer, Binning, and handling mixed & date/time variables.", tier: "free" },
          { num: "DAYS 35-46", title: "Missing Value Imputation & Outlier Engineering", desc: "Complete Case Analysis, Mean/Median/Arbitrary imputation, KNNImputer, IterativeImputer (MICE), Outlier detection using Z-score, IQR & percentiles, feature splitting.", tier: "free" },
          { num: "DAYS 47-57", title: "Dimensionality Reduction (PCA), Regression & Optimization Math", desc: "Geometric intuition of PCA, Simple & Multiple Linear Regression, OLS derivation, Batch/Stochastic/Mini-batch Gradient Descent, Polynomial Regression, Ridge, Lasso, and ElasticNet.", tier: "free" },
          { num: "DAYS 58-75", title: "Classification, Decision Boundaries & Advanced Ensemble Boosting", desc: "Logistic Regression, Multi-class Classification, Confusion Matrix, ROC-AUC, Decision Trees, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, CatBoost & Stacking.", tier: "free" },
          { num: "DAYS 76-85", title: "Unsupervised Clustering & High-Dimensional Anomaly Detection", desc: "K-Means centroid clustering, Elbow method, Silhouette score, Hierarchical agglomerative clustering, DBSCAN density clustering, and anomaly detection in sensor feeds.", tier: "free" },
          { num: "DAYS 86-100", title: "Production MLOps, Containerization & Drift Monitoring", desc: "Model serialization (Joblib/ONNX), Streamlit dashboards, FastAPI asynchronous microservices, Docker containerization, cloud deployment (AWS/GCP), data drift & model retraining.", tier: "free" },
          { num: "LABS TIER", title: "200 Applied Discipline-Specific Capstone Projects", desc: "200 data-driven capstone projects across 10 disciplines with datasets, Colab notebooks, and deployment guidance.", tier: "paid" }
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
        badge: "State-of-the-Art Neural Architectures & GPU Acceleration",
        freeDesc: "In-depth mathematical foundations of biological and artificial neurons, backpropagation calculus, CNN convolutions, sequence LSTMs, and Transformer self-attention.",
        paidDesc: "GPU Training Notebooks on Colab/Kaggle, PyTorch Lightning, YOLOv8/v11 Vision, TensorRT model optimization, and Docker microservice serving.",
        playlistUrl: "https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
        embedUrl: "https://www.youtube.com/embed/aircAruvnKk",
        playlistName: "Neural Networks & Deep Learning Foundations",
        freeFeatures: [
          "Neural Networks from Scratch: Perceptrons, Activations & Computational Graphs",
          "Backpropagation Calculus: Chain rule, loss gradients & loss landscapes",
          "PyTorch Foundations: Tensors, Autograd, Custom Loss & GPU training",
          "Computer Vision: Convolutions, ResNet, Transfer Learning & YOLO Object Detection",
          "Sequence Models: RNNs, LSTMs & GRUs for time-series and sensor telemetry",
          "Transformers: Scaled Dot-Product Self-Attention & Vision Transformers (ViT)"
        ],
        paidFeatures: [
          "GPU-Accelerated Colab/Kaggle Notebooks with Real-World Industry Datasets",
          "Real-Time Vision Pipelines: Custom YOLOv11 and 3D U-Net medical segmentation",
          "Vision Transformers (ViT) & HuggingFace Transformer Fine-Tuning",
          "Low-Latency Model Quantization: FP16 & INT8 inference using NVIDIA TensorRT",
          "Edge Deployment on NVIDIA Jetson & Raspberry Pi AI Accelerators",
          "AI Connectra Certified Deep Learning Specialist Credential"
        ],
        modules: [
          { num: "MODULE 01", title: "Perceptrons to Multilayer Neural Networks", desc: "Biological neurons, Perceptron learning rule, activation functions (Sigmoid, Tanh, ReLU, GELU), multi-layer perceptrons (MLP), and forward propagation.", tier: "free" },
          { num: "MODULE 02", title: "Backpropagation Calculus & Optimization Algorithms", desc: "The chain rule, loss surfaces, vanishing/exploding gradients, SGD with Momentum, RMSprop, Adam, AdamW, and regularization (Dropout, BatchNorm, LayerNorm).", tier: "free" },
          { num: "MODULE 03", title: "Computer Vision: Convolutions, ResNet & YOLO Detection", desc: "Convolutional filters, pooling, feature maps, ResNet residual skip connections, EfficientNet, Transfer Learning, and YOLO real-time object detection.", tier: "free" },
          { num: "MODULE 04", title: "Sequential Architectures: RNNs, LSTMs & Temporal Modeling", desc: "Recurrent neural networks, vanishing gradient in time, LSTM memory cells, GRUs, and multivariate sensor time-series forecasting.", tier: "free" },
          { num: "MODULE 05", title: "Transformers, Self-Attention & Vision Transformers (ViT)", desc: "Scaled dot-product attention, multi-head self-attention, positional encodings, encoder-decoder architectures, and Vision Transformers (ViT).", tier: "paid" },
          { num: "MODULE 06", title: "Production Serving: PyTorch Lightning, TensorRT & Edge AI", desc: "PyTorch Lightning modular structure, ONNX Runtime export, TensorRT INT8 quantization, and ultra-low latency edge serving on NVIDIA hardware.", tier: "paid" }
        ]
      }
    };

    // State Variables
    const heroDisciplineWhyAI = {
      eee: {
        deptKey: 'eee',
        name: 'Electrical & Electronic Engineering (EEE)',
        icon: '⚡',
        tagline: 'From Passive Circuits to Autonomous Energy Grids',
        classicalLimit: "Classical formulas (Kirchhoff's laws, Laplace power flow, thermal steady-state) assume stable, predictable rotating generators. They break down in modern microgrids under stochastic solar/wind intermittency, microsecond switching transients, and non-linear Li-ion battery degradation.",
        aiBreakthrough: "Time-series Transformers and XGBoost forecast renewable generation with 98.4% accuracy; Physics-Informed Neural Networks (PINNs) solve non-linear power flows in milliseconds; and 1D-CNNs localize grid waveform faults in <10ms.",
        careerWhy: "Energy utilities, EV manufacturers (Tesla, BYD), and grid operators no longer hire engineers for manual calculations. They demand EEE graduates who can build automated, software-defined, AI-driven power infrastructure.",
        blueprint: "Smart Grid Dynamic Load & Renewable Forecasting (LSTM + XGBoost) · 98.4% Accuracy",
        blueprintIndex: 0
      },
      ece: {
        deptKey: 'ece',
        name: 'Electronics & Communication Engineering (ECE / Telecom)',
        icon: '📡',
        tagline: 'From Hardcoded Radios to Cognitive 5G/6G Networks',
        classicalLimit: "In 5G/6G millimeter-wave communications with massive MIMO antenna arrays, calculating optimal channel matrices via classical linear algebra inversion takes longer than the channel coherence window itself. Hardcoded RF filters cannot adapt to dynamic spectrum congestion.",
        aiBreakthrough: "Deep Reinforcement Learning performs sub-5ms hybrid beamforming; neural autoencoders decode non-linearly distorted signals through severe fading; and deep RF fingerprinting identifies rogue transmitters for zero-trust physical-layer security.",
        careerWhy: "Qualcomm, Ericsson, Apple, and satellite operators are shifting entirely to AI-native Open-RAN and cognitive radios. ECE students with ML expertise command top-tier compensation in wireless and chip design.",
        blueprint: "5G/6G Massive MIMO Beamforming Optimization (Deep RL) · 4.2x Spectral Efficiency",
        blueprintIndex: 0
      },
      civil: {
        deptKey: 'civil',
        name: 'Civil & Structural Engineering',
        icon: '🏗️',
        tagline: 'From Manual Visual Audits to Autonomous Smart Infrastructure',
        classicalLimit: "Periodic manual visual inspections of bridges, dams, and tunnels miss internal microscopic spalling and seismic fatigue. Classical structural equations (Euler-Bernoulli beams, Terzaghi soil mechanics) cannot process terabytes of real-time IoT vibration telemetry or satellite radar subsidence data.",
        aiBreakthrough: "Drone-mounted Computer Vision (YOLOv11 & 3D U-Net) segments sub-millimeter cracks across bridge piers; LSTM networks predict seismic structural acceleration; and satellite InSAR time-series detect millimeter ground subsidence years before catastrophic collapse.",
        careerWhy: "Top civil engineering firms (Arup, AECOM, Bechtel) are digitizing construction and asset management. Civil engineers who know AI lead digital-twin modeling, smart city sensor grids, and automated safety compliance.",
        blueprint: "Automated Bridge Pier Crack & Spalling Segmentation (YOLOv11 + U-Net) · 96.8% mAP",
        blueprintIndex: 0
      },
      agriculture: {
        deptKey: 'agriculture',
        name: 'Smart Agriculture & Biosystems',
        icon: '🌱',
        tagline: 'From Blanket Chemical Spraying to Micro-Precision Agronomy',
        classicalLimit: "Blanket pesticide spraying and uniform water flooding waste up to 60% of farm inputs, poison topsoil, and deplete groundwater. Classical agronomy depends on manual field scouting that cannot cover hundreds of acres before fungal infestations destroy crops.",
        aiBreakthrough: "Edge AI camera systems on autonomous tractors differentiate weeds from cash crops in real-time at 15 km/h for targeted micro-spraying; satellite multispectral NDVI fusion forecasts crop yields; and vision transformers diagnose foliar pathogens from smartphone leaf photos.",
        careerWhy: "AgTech is undergoing a massive multi-billion-dollar robotic revolution (John Deere, Corteva, Syngenta). Agricultural graduates who master AI lead precision farming, autonomous harvesting, and climate-resilient crop genetics.",
        blueprint: "Autonomous Real-Time Weed vs. Crop Identification (YOLOv8 Edge Vision) · 45% Chemical Reduction",
        blueprintIndex: 0
      },
      business: {
        deptKey: 'business',
        name: 'FinTech, Business & Quantitative Management',
        icon: '📈',
        tagline: 'From Static Spreadsheets to Real-Time Predictive Engines',
        classicalLimit: "Traditional business decision-making relies on backward-looking Excel reports and linear regression that fail during macroeconomic shocks, sudden supply chain disruptions, or sophisticated multi-party financial fraud rings.",
        aiBreakthrough: "Gradient Boosted Trees (XGBoost/LightGBM) score credit card fraud in <20 milliseconds; Graph Neural Networks (GNNs) expose hidden money laundering syndicates; and Transformer NLP parses real-time earnings calls and market sentiment.",
        careerWhy: "Hedge funds, fintech disruptors, and global consultancies (Goldman Sachs, McKinsey, Stripe) prioritize quantitative graduates who can translate business intuition into robust, predictive machine learning pipelines.",
        blueprint: "Real-Time Credit Card Fraud Detection Microservice (XGBoost + SMOTE) · 99.4% ROC-AUC",
        blueprintIndex: 0
      },
      healthcare: {
        deptKey: 'healthcare',
        name: 'Healthcare, Biomedical & Life Sciences',
        icon: '🏥',
        tagline: 'From Reactive Triage to Autonomous Diagnostic Precision',
        classicalLimit: "Radiologists and ICU clinicians are overwhelmed by petabytes of high-resolution medical imaging and continuous telemetry. Human fatigue leads to delayed diagnoses, and static thresholds fail to catch subtle pre-septic vital signs.",
        aiBreakthrough: "3D U-Net architectures delineate brain tumors and lung nodules in CT/MRI with radiologist-level concordance; CNN-LSTMs detect cardiac arrhythmias across 12-lead ECGs; and survival analysis models give 6-hour advance warning for ICU sepsis.",
        careerWhy: "Medical device manufacturers, biotech firms, and healthcare networks urgently need bio-engineers and clinicians who understand AI to build FDA-cleared diagnostic algorithms, automated surgical tools, and genomic pipelines.",
        blueprint: "3D Brain Tumor & Lesion Volumetric Segmentation (3D U-Net on MRI) · 91.2% Dice Score",
        blueprintIndex: 0
      },
      cs: {
        deptKey: 'cs',
        name: 'Computer Science & Software Systems',
        icon: '💻',
        tagline: 'From Hardcoded Logic to Foundation Models & Autonomous Agents',
        classicalLimit: "Traditional software engineering is bound to rigid, brittle if-else rules that cannot handle unstructured human intent, multimodal image/audio inputs, or self-healing cloud microservice orchestration.",
        aiBreakthrough: "Large Language Models, Vision-Language Transformers, and autonomous agent loops synthesize complex workflows, optimize high-concurrency database queries, and convert natural language into production code.",
        careerWhy: "Classical CRUD web development is rapidly commoditizing. Modern software engineering demands proficiency in MLOps, vector embeddings, fine-tuning foundation models, and scalable AI infrastructure.",
        blueprint: "Cloud-Native MLOps Drift Detection & Automated Retraining (FastAPI + Triton) · 10x Scale",
        blueprintIndex: 0
      }
    };

