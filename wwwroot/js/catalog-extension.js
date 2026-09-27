// Expands the original 60-project catalog to 200 projects across 10 disciplines.
// Every item uses the same shared detail view in Index.cshtml; no per-project page is needed.
(function () {
  const additions = {
    eee: [
      'AI-Based Automatic Voltage Regulation', 'Transformer Oil Health Prediction',
      'Non-Intrusive Load Monitoring for Smart Homes', 'EV Charging Demand Forecasting',
      'Arc-Fault Detection from Current Waveforms', 'Wind-Turbine Generator Fault Diagnosis',
      'Electricity Theft Detection with Graph Learning', 'Optimal Power Flow with Reinforcement Learning',
      'Substation Thermal Anomaly Detection', 'Power-Quality Event Classification'
    ],
    ece: [
      'Neural Channel Estimation for OFDM', 'RF Spectrum Occupancy Forecasting',
      'LoRa Network Adaptive Data-Rate Controller', 'Satellite Link Rain-Fade Prediction',
      'Indoor Positioning from Wi-Fi CSI', 'Speech Enhancement for Low-Bandwidth Calls',
      'Network Intrusion Detection for 5G Core', 'Antenna Array Fault Localization',
      'Adaptive Video Streaming Quality Optimizer', 'Edge AI Modulation Classification'
    ],
    civil: [
      'Construction-Site PPE Compliance Detection', 'Concrete Strength Prediction from Mix Design',
      'Urban Flood Depth Forecasting', 'Road Pothole Detection and Severity Mapping',
      'Building Energy Demand Digital Twin', 'Landslide Susceptibility Mapping',
      'Construction Delay Risk Prediction', 'Traffic Signal Optimization with Multi-Agent RL',
      'BIM Clash Prioritization Assistant', 'Water-Pipeline Leakage Localization'
    ],
    agriculture: [
      'Rice Yield Forecasting from Satellite Imagery', 'Smart Irrigation Scheduling',
      'Livestock Health Monitoring from Wearables', 'Fruit Ripeness Grading with Computer Vision',
      'Greenhouse Climate Control with Reinforcement Learning', 'Soil Nutrient Recommendation Engine',
      'Pest Outbreak Early-Warning System', 'Autonomous Farm-Robot Row Navigation',
      'Cold-Chain Spoilage Prediction', 'Aquaculture Water-Quality Anomaly Detection'
    ],
    business: [
      'Customer Churn Intervention Engine', 'Demand Forecasting for Retail Inventory',
      'Invoice Fraud and Duplicate-Payment Detection', 'Dynamic Pricing Optimization',
      'Supply-Chain Disruption Early Warning', 'Employee Attrition Risk Analytics',
      'Contract Clause Risk Extraction', 'Cash-Flow Forecasting for Small Businesses',
      'Marketing Attribution with Causal ML', 'ESG Report Evidence Verification'
    ],
    healthcare: [
      'Diabetic Retinopathy Screening', 'Hospital Readmission Risk Prediction',
      'Medication Interaction Alert Assistant', 'Skin-Lesion Triage with Vision Transformers',
      'Emergency Department Wait-Time Forecasting', 'Wearable Fall Detection for Older Adults',
      'Chest X-Ray Report Drafting Assistant', 'Personalized Rehabilitation Exercise Scoring',
      'Blood-Donor Demand Forecasting', 'Pathology Slide Cell Segmentation'
    ]
  };

  const newDisciplines = {
    cs: {
      name: 'Computer Science & Software Engineering', short: 'CSE', icon: '💻', color: '#8b5cf6',
      description: 'Build dependable AI products, developer tools, cloud platforms, cybersecurity systems, and human-centered software.',
      titles: [
        'Repository-Aware Coding Assistant', 'Cloud Cost Anomaly Detector', 'Log-Based Incident Root-Cause Analyzer',
        'Automated API Test Generator', 'Malware Family Classification', 'Phishing Website Detection',
        'Semantic Code Search Engine', 'Database Query Optimization Advisor', 'Document Question-Answering Platform',
        'Kubernetes Failure Prediction', 'Software Defect Prediction', 'Privacy-Preserving Federated Learning',
        'Multimodal Accessibility Assistant', 'Synthetic Test-Data Generator', 'LLM Hallucination Evaluation Harness',
        'Intelligent Cache Eviction Policy', 'Network Traffic Anomaly Detection', 'Requirements-to-Test Traceability Engine',
        'Open-Source Dependency Risk Scorer', 'MLOps Drift Detection and Automated Retraining'
      ]
    },
    mechanical: {
      name: 'Mechanical & Manufacturing Engineering', short: 'Mechanical', icon: '⚙️', color: '#f97316',
      description: 'Apply AI to machines, factories, robotics, thermal systems, materials, and predictive maintenance.',
      titles: [
        'CNC Tool-Wear Prediction', 'Robotic Arm Grasp Planning', 'Predictive Maintenance for Industrial Pumps',
        'Additive Manufacturing Defect Detection', 'HVAC Energy Optimization', 'Bearing Remaining-Useful-Life Estimation',
        'Digital Twin for a Heat Exchanger', 'Weld Quality Inspection with Vision', 'Autonomous Mobile Robot Navigation',
        'Compressor Fault Diagnosis', 'Topology Optimization Surrogate Model', 'Factory Production Scheduling with RL',
        'Vibration-Based Gearbox Diagnostics', 'Thermal Comfort Personalization', 'Sheet-Metal Surface Defect Detection',
        'Fluid-Flow Surrogate Modeling', 'Industrial Robot Collision Prediction', 'Machine Acoustic Anomaly Detection',
        'Warehouse Picking Route Optimization', 'Energy-Efficient Chiller Control'
      ]
    },
    environment: {
      name: 'Environmental & Earth Sciences', short: 'Environment', icon: '🌍', color: '#10b981',
      description: 'Use AI for climate resilience, pollution monitoring, biodiversity, water security, and geospatial decision support.',
      titles: [
        'Urban Air-Quality Forecasting', 'Wildfire Risk Prediction', 'Plastic Waste Detection in Rivers',
        'Groundwater Level Forecasting', 'Satellite Deforestation Change Detection', 'Coastal Erosion Monitoring',
        'Biodiversity Acoustic Species Recognition', 'Industrial Emission Anomaly Detection', 'Drought Severity Forecasting',
        'Waste-Sorting Computer Vision', 'Methane Plume Detection from Satellite Data', 'Water-Quality Index Prediction',
        'Heat-Island Mapping for Cities', 'Coral Reef Health Classification', 'Flood Inundation Rapid Mapping',
        'Solar Farm Site-Suitability Analysis', 'Environmental Impact Report Assistant', 'Soil Erosion Susceptibility Mapping',
        'Smart Building Carbon Dashboard', 'Climate-Risk Scoring for Infrastructure'
      ]
    },
    education: {
      name: 'Education & Learning Sciences', short: 'Education', icon: '📚', color: '#ec4899',
      description: 'Create responsible learning analytics, accessible tutoring, assessment, and student-support systems.',
      titles: [
        'Adaptive Learning Path Recommender', 'Early Student Dropout Risk Detection', 'Automated Short-Answer Feedback',
        'Lecture Transcript Study Assistant', 'Learning-Outcome Coverage Analyzer', 'Accessible Reading-Level Rewriter',
        'Academic Integrity Evidence Dashboard', 'Personalized Practice Question Generator', 'Classroom Engagement Analytics',
        'Course Timetable Optimization', 'Student Support Ticket Router', 'Pronunciation Coach with Speech AI',
        'Rubric-Based Project Assessment Assistant', 'Curriculum Skill-Gap Mapper', 'Sign-Language Learning Companion',
        'Virtual Science Lab Tutor', 'Multilingual Parent Communication Assistant', 'Library Resource Recommender',
        'Exam Question Difficulty Calibration', 'Learning Analytics Privacy Auditor'
      ]
    }
  };

  const profiles = {
    eee: ['sensor and power-system telemetry', 'time-series forecasting and control', 'Python, PyTorch, Pandapower, Plotly'],
    ece: ['RF, network, and signal measurements', 'signal processing and deep learning', 'Python, GNU Radio, PyTorch, FastAPI'],
    civil: ['images, BIM records, and infrastructure sensors', 'computer vision and predictive modeling', 'Python, OpenCV, YOLO, GeoPandas'],
    agriculture: ['field sensors, drone imagery, and weather data', 'computer vision and forecasting', 'Python, TensorFlow, Rasterio, Streamlit'],
    business: ['transactional, operational, and text data', 'machine learning and decision analytics', 'Python, XGBoost, SHAP, FastAPI'],
    healthcare: ['de-identified clinical signals and medical images', 'validated predictive modeling', 'Python, PyTorch, MONAI, FHIR'],
    cs: ['code, logs, traces, and system events', 'LLMs, anomaly detection, and MLOps', 'Python, PyTorch, FastAPI, Docker'],
    mechanical: ['vibration, acoustic, thermal, and machine telemetry', 'predictive maintenance and control', 'Python, PyTorch, OpenCV, MQTT'],
    environment: ['satellite, GIS, sensor, and climate data', 'geospatial ML and forecasting', 'Python, GeoPandas, Rasterio, XGBoost'],
    education: ['privacy-safe learning events and course content', 'responsible learning analytics and NLP', 'Python, scikit-learn, Transformers, Streamlit']
  };

  function makeProject(key, title, index) {
    const profile = profiles[key];
    const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const difficulty = index % 4 === 0 ? 'Advanced' : 'Intermediate';
    return {
      id: `${key}-${String(index + 1).padStart(2, '0')}`,
      slug,
      title,
      difficulty,
      duration: index % 3 === 0 ? '10–12 weeks' : '8–10 weeks',
      format: 'Guided, self-paced capstone',
      summary: `Design and evaluate a practical ${title.toLowerCase()} system using ${profile[0]}.`,
      problem: `Current workflows for ${title.toLowerCase()} are often manual, reactive, or difficult to scale. Build a reproducible prototype that turns ${profile[0]} into timely, explainable decisions.`,
      ai_technique: profile[1],
      tech_stack: profile[2].split(', '),
      data_pipeline: `Collect or source representative ${profile[0]}; document consent and licensing; clean, split, version, and validate the data before modeling.`,
      hardware_target: 'Google Colab for development; laptop, cloud, or edge deployment as appropriate.',
      impact: `A validated ${title.toLowerCase()} prototype with measurable performance, documented limitations, and a deployment-ready demonstration.`,
      audience: `${newDisciplines[key]?.short || projectsData[key].short || key.toUpperCase()} students who want an applied AI portfolio project.`,
      prerequisites: 'Basic Python, introductory statistics, and foundational machine-learning concepts.',
      performance_metrics: 'Report task-appropriate accuracy and error metrics, latency, robustness, and comparison with a simple baseline.',
      code_snippet: `# ${title}\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(\n    features, target, test_size=0.2, random_state=42\n)\nmodel.fit(X_train, y_train)\nmetrics = evaluate(model, X_test, y_test)\nprint(metrics)`
    };
  }

  Object.entries(additions).forEach(([key, titles]) => {
    const start = projectsData[key].projects.length;
    projectsData[key].projects.push(...titles.map((title, i) => makeProject(key, title, start + i)));
  });

  Object.entries(newDisciplines).forEach(([key, discipline]) => {
    projectsData[key] = {
      name: discipline.name,
      short: discipline.short,
      icon: discipline.icon,
      color: discipline.color,
      tagline: discipline.description,
      description: discipline.description,
      projects: discipline.titles.map((title, i) => makeProject(key, title, i))
    };
    deptMetadata[key] = {
      name: discipline.name,
      shortName: discipline.short,
      icon: discipline.icon,
      color: discipline.color,
      bg: `${discipline.color}22`,
      glow: `${discipline.color}66`,
      synergy: discipline.description
    };
  });

  // Normalize the original 60 records so the shared page can render old and new data identically.
  Object.values(projectsData).forEach((discipline) => {
    discipline.projects.forEach((project, index) => {
      project.summary = project.summary || project.description;
      project.description = project.description || project.summary;
      project.challenge = project.challenge || project.problem;
      project.architecture = project.architecture || project.ai_technique;
      project.performance_metrics = project.performance_metrics || project.impact;
      project.business_impact = project.business_impact || project.impact;
      project.duration = project.duration || (index % 3 === 0 ? '10–12 weeks' : '8–10 weeks');
      project.format = project.format || 'Guided, self-paced capstone';
      project.audience = project.audience || `${discipline.short || discipline.name} students building an applied AI portfolio.`;
      project.prerequisites = project.prerequisites || 'Basic Python, introductory statistics, and foundational machine-learning concepts.';
      project.hardware_target = project.hardware_target || 'Google Colab or an equivalent development environment.';
    });
  });
})();
