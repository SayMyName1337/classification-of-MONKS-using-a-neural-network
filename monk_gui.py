import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QSlider, QPushButton, QSpinBox, QTabWidget,
                             QFileDialog, QMessageBox, QGroupBox, QFormLayout, QDoubleSpinBox,
                             QProgressBar, QSplitter, QCheckBox, QRadioButton, QButtonGroup,
                             QLineEdit, QGridLayout, QTextEdit, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette

import logging
from tensorflow.keras.models import load_model

# Configure path for saving figures
RESULTS_DIR = "monk_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Setup logging
logging.basicConfig(
    filename='monk_gui_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Worker thread for running the experiments
class ExperimentWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    experiment_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.status_updated.emit("Loading data...")
            # Import necessary functions from your monk.py
            from Monks import (load_monks_data, add_noise_to_features, handle_missing_values,
                             create_ensemble_parallel, ensemble_predict, evaluate_model,
                             benchmark_algorithms)
            
            # Load data
            X_train, y_train, X_test, y_test = load_monks_data(self.params['problem_number'])
            if X_train is None:
                self.error_occurred.emit("Failed to load dataset. Please ensure the Monk's dataset files are in the current directory.")
                return
            
            self.status_updated.emit("Preparing data...")
            self.progress_updated.emit(10)
            
            # Prepare data
            X_train_encoded = X_train.values.astype(np.float32)
            X_test_encoded = X_test.values.astype(np.float32)
            y_train_encoded = y_train.values.astype(np.float32)
            y_test_encoded = y_test.values.astype(np.float32)
            
            # Split for validation if training ensemble
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train_encoded, y_train_encoded, test_size=0.2, random_state=42
            )
            
            self.status_updated.emit("Creating ensemble models...")
            self.progress_updated.emit(30)
            
            # Create ensemble models
            input_dim = X_train_encoded.shape[1]
            ensemble_models = create_ensemble_parallel(
                X_train_split, y_train_split, X_val, y_val,
                input_dim, self.params['problem_number'], 
                num_models=self.params['num_ensemble_models']
            )
            
            self.status_updated.emit("Running experiments...")
            self.progress_updated.emit(60)
            
            # Evaluate baseline performance
            _, baseline_predictions = ensemble_predict(ensemble_models, X_test_encoded)
            baseline_metrics = evaluate_model(y_test_encoded, baseline_predictions, "Baseline Ensemble")
            baseline_accuracy = baseline_metrics['accuracy']
            
            # Run noise experiments
            results = {
                'noise_levels': self.params['noise_levels'],
                'noise_type': self.params['noise_type'],
                'mean_accuracies': [],
                'std_accuracies': [],
                'baseline_accuracy': baseline_accuracy,
                'problem_number': self.params['problem_number']
            }
            
            # For each noise level
            for i, noise_level in enumerate(self.params['noise_levels']):
                level_accuracies = []
                status_msg = f"Testing noise level: {noise_level*100:.1f}%"
                self.status_updated.emit(status_msg)
                progress = 60 + int((i / len(self.params['noise_levels'])) * 30)
                self.progress_updated.emit(progress)
                
                # Run multiple experiments per noise level
                for exp in range(self.params['num_experiments']):
                    # Add noise
                    X_test_noisy = add_noise_to_features(X_test_encoded, noise_level, self.params['noise_type'])
                    
                    # Handle missing values if needed
                    if self.params['noise_type'] == 'missing':
                        X_test_noisy = handle_missing_values(X_test_noisy, strategy='mean')
                    
                    # Predict and evaluate
                    _, y_pred = ensemble_predict(ensemble_models, X_test_noisy)
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_test_encoded, y_pred) * 100
                    level_accuracies.append(accuracy)
                
                # Store results
                results['mean_accuracies'].append(np.mean(level_accuracies))
                results['std_accuracies'].append(np.std(level_accuracies))
            
            # Run algorithm comparison if requested
            if self.params.get('run_benchmark', False):
                self.status_updated.emit("Running algorithm benchmarks...")
                self.progress_updated.emit(90)
                
                # No noise benchmark
                bench_results = benchmark_algorithms(
                    X_train_encoded, y_train_encoded,
                    X_test_encoded, y_test_encoded,
                    self.params['problem_number']
                )
                results['benchmark_results'] = bench_results.to_dict('records')
                
                # With noise benchmark
                if self.params['noise_levels'][-1] > 0:
                    noise_bench_results = benchmark_algorithms(
                        X_train_encoded, y_train_encoded,
                        X_test_encoded, y_test_encoded,
                        self.params['problem_number'],
                        noise_level=self.params['noise_levels'][-1],
                        noise_type=self.params['noise_type']
                    )
                    results['noise_benchmark_results'] = noise_bench_results.to_dict('records')
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Experiment completed successfully!")
            
            # Save models if requested
            if self.params.get('save_models', False):
                self.status_updated.emit("Saving models...")
                model_dir = os.path.join(RESULTS_DIR, f"monk{self.params['problem_number']}_models")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                for i, model_path in enumerate(ensemble_models):
                    model = load_model(model_path)
                    save_path = os.path.join(model_dir, f"ensemble_model_{i+1}.h5")
                    model.save(save_path)
            
            # Emit results
            self.experiment_completed.emit(results)
            
        except Exception as e:
            logger.error(f"Error in experiment worker: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"An error occurred: {str(e)}")


# Custom matplotlib canvas for embedding plots
class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class MonkClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MONK Neural Network Classifier")
        self.setMinimumSize(1000, 800)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Configuration tab
        config_tab = QWidget()
        tabs.addTab(config_tab, "Configuration")
        
        # Results tab
        results_tab = QWidget()
        tabs.addTab(results_tab, "Results")
        
        # Benchmarks tab
        benchmarks_tab = QWidget()
        tabs.addTab(benchmarks_tab, "Benchmarks")
        
        # About tab
        about_tab = QWidget()
        tabs.addTab(about_tab, "About")
        
        # Setup each tab
        self.setup_config_tab(config_tab)
        self.setup_results_tab(results_tab)
        self.setup_benchmarks_tab(benchmarks_tab)
        self.setup_about_tab(about_tab)
        
        # Initialize results storage
        self.experiment_results = None
        
        # Set stylesheet for modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border: 1px solid #cccccc;
                border-bottom: 1px solid white;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1c5a85;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 4px;
                min-height: 25px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #e1e1e1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2980b9;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        
        # Status bar for showing current operation
        self.statusBar().showMessage("Ready")
        
        # Log initialization
        logger.info("MONK Classifier GUI initialized")

    def setup_config_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Dataset configuration
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QFormLayout()
        
        # Problem selection
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(["MONK-1", "MONK-2", "MONK-3"])
        dataset_layout.addRow("MONK Problem:", self.problem_combo)
        
        # Dataset info
        dataset_info = QLabel("Dataset will be loaded from the current directory")
        dataset_layout.addRow("Dataset Location:", dataset_info)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Noise configuration
        noise_group = QGroupBox("Noise Configuration")
        noise_layout = QFormLayout()
        
        # Noise type
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["gaussian", "uniform", "impulse", "missing"])
        noise_layout.addRow("Noise Type:", self.noise_type_combo)
        
        # Min noise level
        self.min_noise_spin = QDoubleSpinBox()
        self.min_noise_spin.setRange(0, 100)
        self.min_noise_spin.setValue(0)
        self.min_noise_spin.setSuffix("%")
        noise_layout.addRow("Minimum Noise:", self.min_noise_spin)
        
        # Max noise level
        self.max_noise_spin = QDoubleSpinBox()
        self.max_noise_spin.setRange(0, 100)
        self.max_noise_spin.setValue(50)
        self.max_noise_spin.setSuffix("%")
        noise_layout.addRow("Maximum Noise:", self.max_noise_spin)
        
        # Noise steps
        self.noise_steps_spin = QSpinBox()
        self.noise_steps_spin.setRange(1, 10)
        self.noise_steps_spin.setValue(5)
        noise_layout.addRow("Noise Steps:", self.noise_steps_spin)
        
        # Number of experiments
        self.num_experiments_spin = QSpinBox()
        self.num_experiments_spin.setRange(1, 100)
        self.num_experiments_spin.setValue(5)
        noise_layout.addRow("Experiments Per Level:", self.num_experiments_spin)
        
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        # Model configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        # Number of ensemble models
        self.ensemble_size_spin = QSpinBox()
        self.ensemble_size_spin.setRange(1, 10)
        self.ensemble_size_spin.setValue(5)
        model_layout.addRow("Ensemble Size:", self.ensemble_size_spin)
        
        # Run benchmark comparison
        self.run_benchmark_check = QCheckBox("Run algorithm benchmark comparison")
        self.run_benchmark_check.setChecked(True)
        model_layout.addRow("", self.run_benchmark_check)
        
        # Save models
        self.save_models_check = QCheckBox("Save trained models")
        self.save_models_check.setChecked(False)
        model_layout.addRow("", self.save_models_check)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Run button and progress area
        run_group = QGroupBox("Run Experiment")
        run_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to run experiment")
        run_layout.addWidget(self.status_label)
        
        # Run button
        self.run_button = QPushButton("Run Experiment")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.start_experiment)
        run_layout.addWidget(self.run_button)
        
        run_group.setLayout(run_layout)
        layout.addWidget(run_group)
        
        # Add stretch to bottom
        layout.addStretch()
        
        # Connect signal for min noise change
        self.min_noise_spin.valueChanged.connect(self.update_max_noise_min)
    
    def update_max_noise_min(self, value):
        self.max_noise_spin.setMinimum(value)

    def setup_results_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Results visualization
        results_group = QGroupBox("Experiment Results")
        results_layout = QVBoxLayout()
        
        # Canvas for plots
        self.results_canvas = MplCanvas(width=9, height=6)
        results_layout.addWidget(self.results_canvas)
        
        # Results summary
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        results_layout.addWidget(self.results_text)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.export_results)
        self.export_results_btn.setEnabled(False)
        export_layout.addWidget(self.export_results_btn)
        
        self.export_figure_btn = QPushButton("Export Figure")
        self.export_figure_btn.clicked.connect(self.export_figure)
        self.export_figure_btn.setEnabled(False)
        export_layout.addWidget(self.export_figure_btn)
        
        results_layout.addLayout(export_layout)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

    def setup_benchmarks_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Benchmark visualization
        benchmark_group = QGroupBox("Algorithm Benchmark Comparison")
        benchmark_layout = QVBoxLayout()
        
        # Canvas for benchmark plots
        self.benchmark_canvas = MplCanvas(width=9, height=6)
        benchmark_layout.addWidget(self.benchmark_canvas)
        
        # Benchmark configuration
        benchmark_controls = QHBoxLayout()
        
        # Toggle between no noise and with noise
        self.noise_toggle_group = QButtonGroup()
        self.no_noise_radio = QRadioButton("No Noise")
        self.with_noise_radio = QRadioButton("With Noise")
        self.no_noise_radio.setChecked(True)
        self.noise_toggle_group.addButton(self.no_noise_radio)
        self.noise_toggle_group.addButton(self.with_noise_radio)
        
        # Connect toggle signals
        self.no_noise_radio.toggled.connect(self.update_benchmark_plot)
        self.with_noise_radio.toggled.connect(self.update_benchmark_plot)
        
        benchmark_controls.addWidget(QLabel("Display:"))
        benchmark_controls.addWidget(self.no_noise_radio)
        benchmark_controls.addWidget(self.with_noise_radio)
        benchmark_controls.addStretch()
        
        # Export benchmark button
        self.export_benchmark_btn = QPushButton("Export Benchmark")
        self.export_benchmark_btn.clicked.connect(self.export_benchmark)
        self.export_benchmark_btn.setEnabled(False)
        benchmark_controls.addWidget(self.export_benchmark_btn)
        
        benchmark_layout.addLayout(benchmark_controls)
        benchmark_group.setLayout(benchmark_layout)
        layout.addWidget(benchmark_group)

    def setup_about_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Title and logo (placeholder)
        title_layout = QHBoxLayout()
        
        title_text = QVBoxLayout()
        app_name = QLabel("MONK Neural Network Classifier")
        app_name.setFont(QFont("Arial", 18, QFont.Bold))
        version = QLabel("Version 2.0 - Modern UI Edition")
        
        title_text.addWidget(app_name)
        title_text.addWidget(version)
        
        title_layout.addLayout(title_text)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # Description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setHtml("""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5;">
            <h2 style="color: #2980b9; font-weight: 600; margin-bottom: 20px; border-bottom: 2px solid #3498db; padding-bottom: 10px;">MONK Neural Network Classifier</h2>
            
            <p style="font-size: 15px; text-align: justify; margin-bottom: 15px;">
                Welcome to the <b>MONK Neural Network Classifier</b> — a powerful, intuitive tool designed to explore the fascinating intersection of machine learning, robustness analysis, and neural network ensemble methods. This application brings sophisticated AI research capabilities to your fingertips through an elegant, user-friendly interface.
            </p>
            
            <div style="background-color: rgba(41, 128, 185, 0.1); border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; border-radius: 0 5px 5px 0;">
                <p style="font-size: 15px; font-style: italic; margin: 0;">
                    The MONK's problems represent a landmark in machine learning research — a collection of carefully crafted classification challenges developed to rigorously benchmark learning algorithms. Named after the concept of finding underlying patterns in seemingly complex data (like monks seeking truth), these datasets have become standard testbeds for evaluating AI robustness and generalization capabilities.
                </p>
            </div>
            
            <h3 style="color: #2980b9; margin-top: 25px; font-weight: 600;">✨ Key Capabilities</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">
                <div style="background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(41, 128, 185, 0.2) 100%); padding: 15px; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #2980b9;">🧠 Ensemble Intelligence</h4>
                    <p style="margin-bottom: 0;">Harness the collective power of multiple neural networks working together to achieve superior accuracy and stability.</p>
                </div>
                
                <div style="background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(39, 174, 96, 0.2) 100%); padding: 15px; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #27ae60;">🛡️ Noise Resistance</h4>
                    <p style="margin-bottom: 0;">Test how robust your models are against various types of real-world data corruption and interference.</p>
                </div>
                
                <div style="background: linear-gradient(135deg, rgba(155, 89, 182, 0.1) 0%, rgba(142, 68, 173, 0.2) 100%); padding: 15px; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #8e44ad;">📊 Comprehensive Benchmarking</h4>
                    <p style="margin-bottom: 0;">Compare your neural network ensemble against traditional machine learning algorithms with intuitive visualizations.</p>
                </div>
                
                <div style="background: linear-gradient(135deg, rgba(230, 126, 34, 0.1) 0%, rgba(211, 84, 0, 0.2) 100%); padding: 15px; border-radius: 8px;">
                    <h4 style="margin-top: 0; color: #d35400;">📈 Interactive Visualization</h4>
                    <p style="margin-bottom: 0;">Explore your results through dynamic, customizable plots that reveal insights at a glance.</p>
                </div>
            </div>
            
            <h3 style="color: #2980b9; margin-top: 25px; font-weight: 600;">🔬 Noise Simulation Laboratory</h3>
            
            <p style="font-size: 15px; margin-bottom: 15px;">
                Prepare your models for real-world deployment by testing their resilience against four sophisticated noise types:
            </p>
            
            <table style="width: 100%; border-collapse: collapse; margin: 15px 0; border-radius: 8px; overflow: hidden;">
                <tr style="background-color: #2980b9; color: white;">
                    <th style="padding: 12px; text-align: left;">Noise Type</th>
                    <th style="padding: 12px; text-align: left;">Description</th>
                    <th style="padding: 12px; text-align: left;">Real-world Analog</th>
                    <th style="padding: 12px; text-align: left;">Explicit Example</th>
                </tr>
                <tr style="background-color: rgba(41, 128, 185, 0.1);">
                    <td style="padding: 12px; font-weight: bold;">Gaussian</td>
                    <td style="padding: 12px;">Random noise from a normal distribution, creating subtle variations across features</td>
                    <td style="padding: 12px;">Electronic sensor reading fluctuations, measurement errors</td>
                    <td style="padding: 12px;">If a feature value is 0.75, Gaussian noise might change it to 0.82 or 0.71, with smaller deviations being more likely than larger ones</td>
                </tr>
                <tr>
                    <td style="padding: 12px; font-weight: bold;">Uniform</td>
                    <td style="padding: 12px;">Equal probability noise across a range, creating consistent distortions</td>
                    <td style="padding: 12px;">Quantization errors, digital rounding effects</td>
                    <td style="padding: 12px;">A feature value of 1.5 might become any value between 1.3 and 1.7 with equal probability, regardless of how far from the original</td>
                </tr>
                <tr style="background-color: rgba(41, 128, 185, 0.1);">
                    <td style="padding: 12px; font-weight: bold;">Impulse</td>
                    <td style="padding: 12px;">Sudden extreme values at random positions, creating sharp anomalies</td>
                    <td style="padding: 12px;">Dead pixels in images, electrical spikes in signals</td>
                    <td style="padding: 12px;">Most values remain unchanged, but occasionally a value like 0.5 might suddenly become 0.0 or 1.0 (minimum or maximum possible value)</td>
                </tr>
                <tr>
                    <td style="padding: 12px; font-weight: bold;">Missing</td>
                    <td style="padding: 12px;">Randomly removes values entirely, creating gaps in the dataset</td>
                    <td style="padding: 12px;">Incomplete survey responses, connection dropouts</td>
                    <td style="padding: 12px;">Some feature values are replaced with NaN (Not a Number), requiring imputation before the model can process them</td>
                </tr>
            </table>
            
            <div style="display: flex; align-items: center; background-color: rgba(46, 204, 113, 0.1); padding: 15px; border-radius: 8px; margin: 20px 0;">
                <div style="background-color: #27ae60; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; margin-right: 15px; font-weight: bold;">TF</div>
                <p style="margin: 0; font-size: 15px;">
                    Powered by <b>TensorFlow</b> and <b>PyQt5</b>, this application combines cutting-edge machine learning capabilities with a responsive, modern user interface designed for researchers and practitioners alike.
                </p>
            </div>
            
            <p style="font-size: 14px; text-align: center; margin-top: 25px; font-style: italic; color: #7f8c8d;">
                Begin your journey into neural network resilience testing today — your models will thank you tomorrow.
            </p>
        </div>
        """)
        layout.addWidget(description)
        
        credits_layout = QHBoxLayout()
        credits_label = QLabel("© 2025 - A-06m-23 Shvets Grigoriy")
        credits_layout.addWidget(credits_label)
        credits_layout.addStretch()
        
        layout.addLayout(credits_layout)

    def start_experiment(self):
        # Collect parameters
        problem_number = self.problem_combo.currentIndex() + 1
        min_noise = self.min_noise_spin.value() / 100
        max_noise = self.max_noise_spin.value() / 100
        noise_steps = self.noise_steps_spin.value()
        noise_type = self.noise_type_combo.currentText()
        num_experiments = self.num_experiments_spin.value()
        num_ensemble_models = self.ensemble_size_spin.value()
        run_benchmark = self.run_benchmark_check.isChecked()
        save_models = self.save_models_check.isChecked()
        
        # Generate noise levels
        if noise_steps == 1:
            noise_levels = [min_noise]
        else:
            noise_levels = np.linspace(min_noise, max_noise, noise_steps)
        
        # Package parameters
        params = {
            'problem_number': problem_number,
            'noise_levels': noise_levels.tolist(),
            'noise_type': noise_type,
            'num_experiments': num_experiments,
            'num_ensemble_models': num_ensemble_models,
            'run_benchmark': run_benchmark,
            'save_models': save_models
        }
        
        # Update UI
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting experiment...")
        
        # Start worker thread
        self.worker = ExperimentWorker(params)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.experiment_completed.connect(self.process_results)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()
        
        # Log parameters
        logger.info(f"Starting experiment with parameters: {params}")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, status):
        self.status_label.setText(status)
        self.statusBar().showMessage(status)

    def handle_error(self, error_msg):
        self.run_button.setEnabled(True)
        self.status_label.setText("Error occurred")
        self.statusBar().showMessage("Error")
        
        # Show error dialog
        QMessageBox.critical(self, "Error", error_msg)
        logger.error(f"Error in experiment: {error_msg}")

    def process_results(self, results):
        self.experiment_results = results
        self.run_button.setEnabled(True)
        self.status_label.setText("Experiment completed")
        self.statusBar().showMessage("Ready")
        
        # Update results visualization
        self.update_results_plots()
        
        # Update benchmark plots if available
        if 'benchmark_results' in results:
            self.update_benchmark_plot()
            self.export_benchmark_btn.setEnabled(True)
        
        # Enable export buttons
        self.export_results_btn.setEnabled(True)
        self.export_figure_btn.setEnabled(True)
        
        # Switch to results tab
        self.centralWidget().findChild(QTabWidget).setCurrentIndex(1)
        
        # Log completion
        logger.info("Experiment completed and results processed")

    def update_results_plots(self):
        if not self.experiment_results:
            return
        
        # Clear previous plot
        self.results_canvas.axes.clear()
        
        # Extract data
        noise_levels = [level * 100 for level in self.experiment_results['noise_levels']]  # Convert to percentages
        mean_accuracies = self.experiment_results['mean_accuracies']
        std_accuracies = self.experiment_results['std_accuracies']
        baseline_accuracy = self.experiment_results['baseline_accuracy']
        noise_type = self.experiment_results['noise_type']
        problem_number = self.experiment_results['problem_number']
        
        # Plot mean accuracies with error bars
        self.results_canvas.axes.errorbar(
            noise_levels,
            mean_accuracies,
            yerr=std_accuracies,
            fmt='o-',
            capsize=5,
            label=f'{noise_type.capitalize()} Noise'
        )
        
        # Add baseline
        self.results_canvas.axes.axhline(
            y=baseline_accuracy, 
            color='g', 
            linestyle='--',
            label=f'Baseline (No Noise): {baseline_accuracy:.2f}%'
        )
        
        # Add labels and title
        self.results_canvas.axes.set_title(f'Impact of {noise_type.capitalize()} Noise on MONK-{problem_number} Classification Accuracy')
        self.results_canvas.axes.set_xlabel('Noise Level (%)')
        self.results_canvas.axes.set_ylabel('Accuracy (%)')
        self.results_canvas.axes.grid(True, alpha=0.3)
        self.results_canvas.axes.legend()
        
        # Refresh canvas
        self.results_canvas.draw()
        
        # Update results text
        result_text = f"<h3>MONK-{problem_number} Noise Resistance Experiment Results</h3>"
        result_text += f"<p><b>Baseline accuracy (no noise):</b> {baseline_accuracy:.2f}%</p>"
        result_text += f"<p><b>Noise type:</b> {noise_type.capitalize()}</p>"
        result_text += "<p><b>Accuracy at different noise levels:</b></p><ul>"
        
        for i, level in enumerate(noise_levels):
            result_text += f"<li>{level:.1f}% noise: {mean_accuracies[i]:.2f}% (±{std_accuracies[i]:.2f}%)</li>"
        
        result_text += "</ul>"
        
        self.results_text.setHtml(result_text)

    def update_benchmark_plot(self):
        if not self.experiment_results or 'benchmark_results' not in self.experiment_results:
            return
        
        # Clear previous plot
        self.benchmark_canvas.axes.clear()
        
        # Determine which results to plot
        if self.no_noise_radio.isChecked() or 'noise_benchmark_results' not in self.experiment_results:
            benchmark_data = self.experiment_results['benchmark_results']
            title_suffix = "without Noise"
        else:
            benchmark_data = self.experiment_results['noise_benchmark_results']
            title_suffix = f"with {self.experiment_results['noise_type'].capitalize()} Noise ({self.experiment_results['noise_levels'][-1]*100:.1f}%)"
        
        # Extract data
        algorithms = [item['Algorithm'] for item in benchmark_data]
        accuracies = [item['Accuracy'] for item in benchmark_data]
        
        # Create bar plot
        bars = self.benchmark_canvas.axes.bar(algorithms, accuracies, color='#3498db')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            self.benchmark_canvas.axes.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{height:.2f}%',
                ha='center',
                va='bottom'
            )
        
        # Add labels and title
        problem_number = self.experiment_results['problem_number']
        self.benchmark_canvas.axes.set_title(f'Algorithm Comparison for MONK-{problem_number} {title_suffix}')
        self.benchmark_canvas.axes.set_ylabel('Accuracy (%)')
        self.benchmark_canvas.axes.set_ylim(0, 105)  # Leave room for text
        self.benchmark_canvas.axes.grid(axis='y', alpha=0.3)
        
        # Adjust x-axis labels for better readability if many algorithms
        if len(algorithms) > 4:
            self.benchmark_canvas.axes.set_xticklabels(algorithms, rotation=45, ha='right')
            self.benchmark_canvas.fig.tight_layout()
        
        # Add a horizontal line for baseline accuracy if available
        if 'baseline_accuracy' in self.experiment_results:
            baseline = self.experiment_results['baseline_accuracy']
            self.benchmark_canvas.axes.axhline(
                y=baseline,
                color='r',
                linestyle='--',
                label=f'Baseline Accuracy: {baseline:.2f}%'
            )
            self.benchmark_canvas.axes.legend()
        
        # Color the best performing algorithm differently
        if len(accuracies) > 0:
            best_idx = accuracies.index(max(accuracies))
            bars[best_idx].set_color('#2ecc71')  # Green for best algorithm
            
            # Add annotation for best algorithm
            self.benchmark_canvas.axes.annotate(
                'Best Algorithm',
                xy=(best_idx, accuracies[best_idx]),
                xytext=(best_idx, accuracies[best_idx] + 5),
                ha='center'
            )
        
        # Refresh canvas
        self.benchmark_canvas.draw()
    
    def export_results(self):
        """Export experiment results to a CSV file"""
        if not self.experiment_results:
            return
        
        # Select file location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", 
            os.path.join(RESULTS_DIR, f"monk{self.experiment_results['problem_number']}_{self.experiment_results['noise_type']}_results.csv"),
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Create DataFrame
            results_df = pd.DataFrame({
                'Noise Level (%)': [level * 100 for level in self.experiment_results['noise_levels']],
                'Mean Accuracy (%)': self.experiment_results['mean_accuracies'],
                'Std Deviation (%)': self.experiment_results['std_accuracies']
            })
            
            # Add benchmark results if available
            if 'benchmark_results' in self.experiment_results:
                benchmark_df = pd.DataFrame(self.experiment_results['benchmark_results'])
                
                # Save to file - if Excel format
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        results_df.to_excel(writer, sheet_name='Noise Results', index=False)
                        benchmark_df.to_excel(writer, sheet_name='Algorithm Benchmark', index=False)
                        
                        # Add noise benchmark if available
                        if 'noise_benchmark_results' in self.experiment_results:
                            noise_bench_df = pd.DataFrame(self.experiment_results['noise_benchmark_results'])
                            noise_bench_df.to_excel(writer, sheet_name='Noisy Benchmark', index=False)
                    
                    self.statusBar().showMessage(f"Results exported to {file_path}")
                    logger.info(f"Results exported to {file_path}")
                else:
                    # Just save noise results to CSV
                    results_df.to_csv(file_path, index=False)
                    
                    # Save benchmark to separate file
                    bench_path = file_path.replace('.csv', '_benchmark.csv')
                    benchmark_df.to_csv(bench_path, index=False)
                    
                    self.statusBar().showMessage(f"Results exported to {file_path} and {bench_path}")
                    logger.info(f"Results exported to {file_path} and {bench_path}")
            else:
                # Just save noise results to CSV or Excel
                if file_path.endswith('.xlsx'):
                    results_df.to_excel(file_path, index=False)
                else:
                    results_df.to_csv(file_path, index=False)
                
                self.statusBar().showMessage(f"Results exported to {file_path}")
                logger.info(f"Results exported to {file_path}")
            
            # Show success message
            QMessageBox.information(self, "Export Successful", "Results have been exported successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")
            logger.error(f"Error exporting results: {str(e)}", exc_info=True)

    def export_figure(self):
        """Export the current figure to an image file"""
        if not self.experiment_results:
            return
        
        # Select file location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", 
            os.path.join(RESULTS_DIR, f"monk{self.experiment_results['problem_number']}_{self.experiment_results['noise_type']}_plot.png"),
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save figure
            self.results_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            # Show success message
            self.statusBar().showMessage(f"Figure exported to {file_path}")
            logger.info(f"Figure exported to {file_path}")
            QMessageBox.information(self, "Export Successful", "Figure has been exported successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting figure: {str(e)}")
            logger.error(f"Error exporting figure: {str(e)}", exc_info=True)

    def export_benchmark(self):
        """Export the benchmark comparison figure to an image file"""
        if not self.experiment_results or 'benchmark_results' not in self.experiment_results:
            return
        
        # Determine which results are being shown
        benchmark_type = "no_noise" if self.no_noise_radio.isChecked() else "with_noise"
        
        # Select file location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Benchmark Figure", 
            os.path.join(RESULTS_DIR, f"monk{self.experiment_results['problem_number']}_benchmark_{benchmark_type}.png"),
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save figure
            self.benchmark_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            # Show success message
            self.statusBar().showMessage(f"Benchmark figure exported to {file_path}")
            logger.info(f"Benchmark figure exported to {file_path}")
            QMessageBox.information(self, "Export Successful", "Benchmark figure has been exported successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting benchmark figure: {str(e)}")
            logger.error(f"Error exporting benchmark figure: {str(e)}", exc_info=True)

# New module for main function
def main():
    """Run the MONK Classifier GUI application"""
    # Enable high DPI scaling for better display on high-resolution monitors
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = MonkClassifierGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


# Execute the main function when the script is run
if __name__ == "__main__":
    main()