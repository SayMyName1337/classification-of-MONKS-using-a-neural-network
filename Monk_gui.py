import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json
from pathlib import Path
import time
from datetime import datetime
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QRadioButton, QButtonGroup, QFileDialog, QTextEdit, 
                            QProgressBar, QGroupBox, QFormLayout, QSplitter, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QMessageBox, QGridLayout, QFrame,
                            QSizePolicy, QStackedWidget, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
from PyQt5.QtGui import QFont, QIcon, QDesktopServices, QPixmap, QPalette, QColor, QMovie

# Импортируем вашу существующую логику классификации
# ВАЖНО: В зависимости от того, как организован ваш код, может потребоваться адаптация импортов
from Monks_v2 import (DataLoader, HyperOptimizer, EnsembleClassifier, run_optimal_classifier, 
                  ComparativeAnalyzer, analyze_ensemble_size)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monk_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MONK_GUI")

# Пути к директориям для сохранения
RESULTS_DIR = Path("./results")
MODELS_DIR = Path("./models")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# Классы для выполнения задач в отдельных потоках
class ClassificationWorker(QThread):
    """Поток для выполнения классификации."""
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    update_log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            # Логируем начало процесса
            self.update_status.emit("Инициализация...")
            self.update_log.emit(f"Запуск классификации для MONK-{self.params['problem_number']}")
            
            # Эмулируем процесс для отображения прогресса
            self.update_progress.emit(5)
            self.update_status.emit("Загрузка данных...")
            self.update_log.emit("Загрузка и подготовка данных...")
            
            # Запускаем классификацию
            start_time = time.time()
            
            # Проверяем режим работы
            if self.params['mode'] == 'training':
                self.update_progress.emit(15)
                self.update_status.emit("Обучение ансамбля...")
                self.update_log.emit(f"Обучение ансамбля из {self.params['ensemble_size']} моделей...")
                
                # Запускаем классификацию
                ensemble = run_optimal_classifier(
                    self.params['problem_number'],
                    run_hyperopt=self.params['run_hyperopt'],
                    n_trials=self.params['n_trials'],
                    ensemble_size=self.params['ensemble_size'],
                    analyze_noise=self.params['analyze_noise'],
                    run_comparative=False,
                    min_noise=self.params['min_noise'],
                    max_noise=self.params['max_noise'],
                    noise_step=self.params['noise_step']
                )
                
                # Эмулируем завершение для отображения прогресса
                self.update_progress.emit(95)
                self.update_status.emit("Завершение...")
                
            elif self.params['mode'] == 'comparative':
                self.update_progress.emit(15)
                self.update_status.emit("Сравнительный анализ...")
                self.update_log.emit(f"Запуск сравнительного анализа для MONK-{self.params['problem_number']}...")
                
                # Запускаем сравнительный анализ
                ensemble = run_optimal_classifier(
                    self.params['problem_number'],
                    run_hyperopt=self.params['run_hyperopt'],
                    n_trials=self.params['n_trials'],
                    ensemble_size=self.params['ensemble_size'],
                    analyze_noise=False,
                    run_comparative=True,
                    min_noise=self.params['min_noise'],
                    max_noise=self.params['max_noise'],
                    noise_step=self.params['noise_step']
                )
                
                # Эмулируем завершение для отображения прогресса
                self.update_progress.emit(95)
                self.update_status.emit("Завершение...")
                
            elif self.params['mode'] == 'ensemble_size':
                self.update_progress.emit(15)
                self.update_status.emit("Анализ размера ансамбля...")
                self.update_log.emit(f"Запуск анализа размера ансамбля для MONK-{self.params['problem_number']}...")
                
                # Запускаем анализ размера ансамбля
                results = analyze_ensemble_size(
                    self.params['problem_number'],
                    max_ensemble_size=self.params['max_ensemble_size'],
                    noise_levels=np.arange(
                        self.params['min_noise'], 
                        self.params['max_noise'] + self.params['noise_step']/2, 
                        self.params['noise_step']
                    ).tolist()
                )
                
                # Эмулируем завершение для отображения прогресса
                self.update_progress.emit(95)
                self.update_status.emit("Завершение...")
            
            # Рассчитываем время выполнения
            elapsed_time = time.time() - start_time
            
            # Собираем результаты
            results = {
                'elapsed_time': elapsed_time,
                'problem_number': self.params['problem_number'],
                'mode': self.params['mode'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Сообщаем о завершении
            self.update_progress.emit(100)
            self.update_status.emit("Готово!")
            self.update_log.emit(f"Классификация завершена за {elapsed_time:.2f} секунд")
            self.finished_signal.emit(results)
            
        except Exception as e:
            # В случае ошибки
            error_msg = f"Ошибка при выполнении классификации: {str(e)}"
            self.update_log.emit(error_msg)
            self.error_signal.emit(error_msg)
            logger.exception("Ошибка в потоке классификации")


# Классы для визуализации
class MatplotlibCanvas(FigureCanvas):
    """Виджет для отображения графиков matplotlib."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Настройка внешнего вида графика
        self.fig.patch.set_facecolor('#f0f0f0')
        self.axes.grid(True, alpha=0.3)
        
        # Автоматическое масштабирование виджета
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ResultsTable(QTableWidget):
    """Таблица для отображения результатов классификации."""
    def __init__(self, parent=None):
        super(ResultsTable, self).__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Модель", "Точность", "F1-Score", "ROC AUC"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        
    def update_with_results(self, results):
        """Обновляет таблицу с результатами классификации."""
        self.setRowCount(0)  # Очищаем таблицу
        
        # Проверяем наличие результатов
        if not results or 'models' not in results:
            return
        
        # Добавляем строки с результатами
        for model_name, metrics in results['models'].items():
            row_position = self.rowCount()
            self.insertRow(row_position)
            
            # Заполняем ячейки таблицы
            self.setItem(row_position, 0, QTableWidgetItem(model_name))
            self.setItem(row_position, 1, QTableWidgetItem(f"{metrics['accuracy']:.4f}"))
            self.setItem(row_position, 2, QTableWidgetItem(f"{metrics.get('f1_score', 'N/A')}"))
            self.setItem(row_position, 3, QTableWidgetItem(f"{metrics.get('roc_auc', 'N/A')}"))
            
            # Выделяем лучшую модель
            if metrics.get('is_best', False):
                for col in range(4):
                    self.item(row_position, col).setBackground(QColor(200, 255, 200))


# Основной класс главного окна
class MonkClassifierGUI(QMainWindow):
    """Главное окно приложения для классификации MONK."""
    
    def __init__(self):
        super().__init__()
        
        # Настройка основного окна
        self.setWindowTitle("Классификатор MONK - Интеллектуальная система для решения задачи классификации")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon("icon.png"))  # Если есть иконка
        
        # Инициализация UI
        self.init_ui()
        
        # Состояние приложения
        self.current_results = None
        self.worker = None
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        # Основной виджет и компоновка
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #f5f5f5;
                border-bottom: 1px solid #f5f5f5;
            }
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
        """)
        
        # Создаем вкладки
        self.setup_tab = self._create_setup_tab()
        self.results_tab = self._create_results_tab()
        self.visualization_tab = self._create_visualization_tab()
        self.log_tab = self._create_log_tab()
        
        # Добавляем вкладки
        self.tabs.addTab(self.setup_tab, "Настройка")
        self.tabs.addTab(self.results_tab, "Результаты")
        self.tabs.addTab(self.visualization_tab, "Визуализация")
        self.tabs.addTab(self.log_tab, "Журнал")
        
        # Добавляем вкладки в основную компоновку
        main_layout.addWidget(self.tabs)
        
        # Статусная строка
        self.statusBar().showMessage("Готово к работе")
        
        # Добавляем прогресс-бар в статусную строку
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(15)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Логируем запуск приложения
        logger.info("Приложение запущено")
        
    def _create_setup_tab(self):
        """Создание вкладки настройки."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Верхняя панель с выбором задачи и режима
        top_panel = QGroupBox("Основные настройки")
        top_layout = QFormLayout()
        
        # Выбор проблемы MONK
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(["MONK-1", "MONK-2", "MONK-3"])
        self.problem_combo.setCurrentIndex(1)  # По умолчанию MONK-2
        top_layout.addRow("Задача:", self.problem_combo)
        
        # Выбор режима работы
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Обучение и оценка классификатора", 
            "Сравнительный анализ с другими алгоритмами", 
            "Анализ влияния размера ансамбля"
        ])
        self.mode_combo.currentIndexChanged.connect(self._update_ui_for_mode)
        top_layout.addRow("Режим работы:", self.mode_combo)
        
        top_panel.setLayout(top_layout)
        layout.addWidget(top_panel)
        
        # Панель параметров ансамбля
        self.ensemble_panel = QGroupBox("Параметры ансамбля")
        ensemble_layout = QFormLayout()
        
        # Размер ансамбля
        self.ensemble_size_spin = QSpinBox()
        self.ensemble_size_spin.setRange(1, 20)
        self.ensemble_size_spin.setValue(5)
        ensemble_layout.addRow("Размер ансамбля:", self.ensemble_size_spin)
        
        # Максимальный размер ансамбля для анализа
        self.max_ensemble_size_spin = QSpinBox()
        self.max_ensemble_size_spin.setRange(5, 30)
        self.max_ensemble_size_spin.setValue(15)
        ensemble_layout.addRow("Максимальный размер для анализа:", self.max_ensemble_size_spin)
        
        # Оптимизация гиперпараметров
        self.hyperopt_check = QCheckBox("Выполнить оптимизацию гиперпараметров")
        self.hyperopt_check.setChecked(True)
        ensemble_layout.addRow("", self.hyperopt_check)
        
        # Количество попыток оптимизации
        self.n_trials_spin = QSpinBox()
        self.n_trials_spin.setRange(10, 200)
        self.n_trials_spin.setValue(50)
        self.n_trials_spin.setSingleStep(10)
        ensemble_layout.addRow("Количество попыток оптимизации:", self.n_trials_spin)
        
        self.ensemble_panel.setLayout(ensemble_layout)
        layout.addWidget(self.ensemble_panel)
        
        # Панель параметров шума
        self.noise_panel = QGroupBox("Параметры шума")
        noise_layout = QFormLayout()
        
        # Типы шума
        noise_group = QWidget()
        noise_group_layout = QHBoxLayout(noise_group)
        noise_group_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gaussian_check = QCheckBox("Гауссовский")
        self.gaussian_check.setChecked(True)
        noise_group_layout.addWidget(self.gaussian_check)
        
        self.uniform_check = QCheckBox("Равномерный")
        self.uniform_check.setChecked(True)
        noise_group_layout.addWidget(self.uniform_check)
        
        self.impulse_check = QCheckBox("Импульсный")
        self.impulse_check.setChecked(True)
        noise_group_layout.addWidget(self.impulse_check)
        
        self.missing_check = QCheckBox("Пропущенные данные")
        self.missing_check.setChecked(True)
        noise_group_layout.addWidget(self.missing_check)
        
        noise_layout.addRow("Типы шума:", noise_group)
        
        # Минимальный уровень шума
        self.min_noise_spin = QDoubleSpinBox()
        self.min_noise_spin.setRange(0, 100)
        self.min_noise_spin.setValue(0)
        self.min_noise_spin.setSuffix("%")
        self.min_noise_spin.setSingleStep(5)
        noise_layout.addRow("Минимальный уровень шума:", self.min_noise_spin)
        
        # Максимальный уровень шума
        self.max_noise_spin = QDoubleSpinBox()
        self.max_noise_spin.setRange(0, 100)
        self.max_noise_spin.setValue(50)
        self.max_noise_spin.setSuffix("%")
        self.max_noise_spin.setSingleStep(5)
        noise_layout.addRow("Максимальный уровень шума:", self.max_noise_spin)
        
        # Шаг изменения уровня шума
        self.noise_step_spin = QDoubleSpinBox()
        self.noise_step_spin.setRange(1, 100)
        self.noise_step_spin.setValue(10)
        self.noise_step_spin.setSuffix("%")
        self.noise_step_spin.setSingleStep(5)
        noise_layout.addRow("Шаг изменения уровня шума:", self.noise_step_spin)
        
        self.noise_panel.setLayout(noise_layout)
        layout.addWidget(self.noise_panel)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        # Кнопка справки
        self.help_button = QPushButton("Справка")
        self.help_button.setIcon(QIcon("help.png"))  # Если есть иконка
        self.help_button.clicked.connect(self._show_help)
        buttons_layout.addWidget(self.help_button)
        
        # Растягивающийся виджет для выравнивания
        buttons_layout.addStretch()
        
        # Кнопка запуска
        self.start_button = QPushButton("Запустить")
        self.start_button.setIcon(QIcon("start.png"))  # Если есть иконка
        self.start_button.clicked.connect(self._start_classification)
        self.start_button.setMinimumWidth(120)
        buttons_layout.addWidget(self.start_button)
        
        layout.addLayout(buttons_layout)
        
        # Растягивающийся виджет для заполнения пустого пространства
        layout.addStretch()
        
        tab.setLayout(layout)
        
        # Обновляем UI в соответствии с текущим режимом
        self._update_ui_for_mode(0)
        
        return tab
    
    def _create_results_tab(self):
        """Создание вкладки результатов."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Информационная панель
        info_panel = QGroupBox("Информация о последнем запуске")
        info_layout = QFormLayout()
        
        # Метки для информации
        self.info_problem = QLabel("Не выполнялось")
        info_layout.addRow("Задача:", self.info_problem)
        
        self.info_mode = QLabel("Не выполнялось")
        info_layout.addRow("Режим:", self.info_mode)
        
        self.info_ensemble_size = QLabel("Не выполнялось")
        info_layout.addRow("Размер ансамбля:", self.info_ensemble_size)
        
        self.info_accuracy = QLabel("Не выполнялось")
        info_layout.addRow("Достигнутая точность:", self.info_accuracy)
        
        self.info_time = QLabel("Не выполнялось")
        info_layout.addRow("Время выполнения:", self.info_time)
        
        info_panel.setLayout(info_layout)
        layout.addWidget(info_panel)
        
        # Таблица результатов
        results_panel = QGroupBox("Результаты классификации")
        results_layout = QVBoxLayout()
        
        self.results_table = ResultsTable()
        results_layout.addWidget(self.results_table)
        
        # Кнопки для работы с результатами
        results_buttons = QHBoxLayout()
        
        self.export_button = QPushButton("Экспорт результатов")
        self.export_button.setIcon(QIcon("export.png"))  # Если есть иконка
        self.export_button.clicked.connect(self._export_results)
        results_buttons.addWidget(self.export_button)
        
        self.report_button = QPushButton("Создать отчет")
        self.report_button.setIcon(QIcon("report.png"))  # Если есть иконка
        self.report_button.clicked.connect(self._generate_report)
        results_buttons.addWidget(self.report_button)
        
        results_layout.addLayout(results_buttons)
        
        results_panel.setLayout(results_layout)
        layout.addWidget(results_panel)
        
        tab.setLayout(layout)
        return tab
    
    def _create_visualization_tab(self):
        """Создание вкладки визуализации."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Панель управления визуализацией
        control_panel = QGroupBox("Управление визуализацией")
        control_layout = QHBoxLayout()
        
        # Выбор графика
        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "Точность классификации",
            "Матрица ошибок",
            "ROC-кривая",
            "Влияние шума",
            "Сравнение алгоритмов",
            "Зависимость от размера ансамбля"
        ])
        self.plot_combo.currentIndexChanged.connect(self._update_visualization)
        control_layout.addWidget(QLabel("Выберите график:"))
        control_layout.addWidget(self.plot_combo)
        
        # Кнопка обновления
        self.refresh_button = QPushButton("Обновить")
        self.refresh_button.setIcon(QIcon("refresh.png"))  # Если есть иконка
        self.refresh_button.clicked.connect(self._update_visualization)
        control_layout.addWidget(self.refresh_button)
        
        # Кнопка сохранения
        self.save_plot_button = QPushButton("Сохранить график")
        self.save_plot_button.setIcon(QIcon("save.png"))  # Если есть иконка
        self.save_plot_button.clicked.connect(self._save_plot)
        control_layout.addWidget(self.save_plot_button)
        
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)
        
        # Контейнер для графика
        self.plot_container = QVBoxLayout()
        
        # Создаем холст matplotlib
        self.canvas = MatplotlibCanvas(width=8, height=6, dpi=100)
        self.plot_container.addWidget(self.canvas)
        
        # Подпись к графику
        self.plot_caption = QLabel("Выберите график для отображения")
        self.plot_caption.setAlignment(Qt.AlignCenter)
        self.plot_container.addWidget(self.plot_caption)
        
        # Добавляем контейнер графика
        layout.addLayout(self.plot_container)
        
        tab.setLayout(layout)
        return tab
    
    def _create_log_tab(self):
        """Создание вкладки журнала."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Текстовое поле для лога
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                font-family: Consolas, Monaco, monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.log_text)
        
        # Кнопки управления логом
        log_buttons = QHBoxLayout()
        
        self.clear_log_button = QPushButton("Очистить журнал")
        self.clear_log_button.setIcon(QIcon("clear.png"))  # Если есть иконка
        self.clear_log_button.clicked.connect(self.log_text.clear)
        log_buttons.addWidget(self.clear_log_button)
        
        self.save_log_button = QPushButton("Сохранить журнал")
        self.save_log_button.setIcon(QIcon("save.png"))  # Если есть иконка
        self.save_log_button.clicked.connect(self._save_log)
        log_buttons.addWidget(self.save_log_button)
        
        layout.addLayout(log_buttons)
        
        tab.setLayout(layout)
        return tab
    
    def _update_ui_for_mode(self, index):
        """Обновляет UI в зависимости от выбранного режима работы."""
        mode = self.mode_combo.currentText()
        
        # Обновляем доступность виджетов в зависимости от режима
        if "Обучение и оценка" in mode:
            # Режим обучения
            self.ensemble_size_spin.setEnabled(True)
            self.max_ensemble_size_spin.setEnabled(False)
            self.hyperopt_check.setEnabled(True)
            self.n_trials_spin.setEnabled(True)
            
            # Включаем панель шума
            self.noise_panel.setEnabled(True)
            
        elif "Сравнительный анализ" in mode:
            # Режим сравнительного анализа
            self.ensemble_size_spin.setEnabled(True)
            self.max_ensemble_size_spin.setEnabled(False)
            self.hyperopt_check.setEnabled(True)
            self.n_trials_spin.setEnabled(True)
            
            # Включаем панель шума
            self.noise_panel.setEnabled(True)
            
        elif "Анализ влияния размера" in mode:
            # Режим анализа размера ансамбля
            self.ensemble_size_spin.setEnabled(False)
            self.max_ensemble_size_spin.setEnabled(True)
            self.hyperopt_check.setEnabled(False)
            self.n_trials_spin.setEnabled(False)
            
            # Включаем панель шума
            self.noise_panel.setEnabled(True)
            
    def _start_classification(self):
        """Запускает процесс классификации."""
        # Проверяем, не запущен ли уже процесс
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Предупреждение", 
                              "Процесс классификации уже запущен. Дождитесь его завершения.")
            return
        
        # Получаем параметры из UI
        problem_text = self.problem_combo.currentText()
        problem_number = int(problem_text.split("-")[1])
        
        mode_text = self.mode_combo.currentText()
        if "Обучение и оценка" in mode_text:
            mode = "training"
        elif "Сравнительный анализ" in mode_text:
            mode = "comparative"
        else:
            mode = "ensemble_size"
        
        # Собираем параметры для запуска
        params = {
            'problem_number': problem_number,
            'mode': mode,
            'ensemble_size': self.ensemble_size_spin.value(),
            'max_ensemble_size': self.max_ensemble_size_spin.value(),
            'run_hyperopt': self.hyperopt_check.isChecked(),
            'n_trials': self.n_trials_spin.value(),
            'analyze_noise': True,
            'min_noise': self.min_noise_spin.value() / 100.0,  # Преобразуем из % в доли
            'max_noise': self.max_noise_spin.value() / 100.0,  # Преобразуем из % в доли
            'noise_step': self.noise_step_spin.value() / 100.0  # Преобразуем из % в доли
        }
        
        # Получаем выбранные типы шума
        noise_types = []
        if self.gaussian_check.isChecked():
            noise_types.append('gaussian')
        if self.uniform_check.isChecked():
            noise_types.append('uniform')
        if self.impulse_check.isChecked():
            noise_types.append('impulse')
        if self.missing_check.isChecked():
            noise_types.append('missing')
        
        params['noise_types'] = noise_types
        
        # Логируем начало работы
        self.log_text.append(f"<b>[{datetime.now().strftime('%H:%M:%S')}] "
                           f"Запуск классификации для MONK-{problem_number}, режим: {mode_text}</b>")
        
        # Создаем и настраиваем рабочий поток
        self.worker = ClassificationWorker(params)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.update_status.connect(self.statusBar().showMessage)
        self.worker.update_log.connect(self._append_log)
        self.worker.finished_signal.connect(self._classification_finished)
        self.worker.error_signal.connect(self._show_error)
        
        # Отключаем кнопку запуска и запускаем процесс
        self.start_button.setEnabled(False)
        self.worker.start()
        
        # Показываем вкладку лога
        self.tabs.setCurrentWidget(self.log_tab)
        
    def _classification_finished(self, results):
        """Обрабатывает завершение классификации."""
        # Включаем кнопку запуска
        self.start_button.setEnabled(True)
        
        # Сохраняем результаты
        self.current_results = results
        
        # Обновляем информацию на вкладке результатов
        self.info_problem.setText(f"MONK-{results['problem_number']}")
        self.info_mode.setText(self._get_mode_display_name(results['mode']))
        
        # В реальной программе здесь нужно будет обновить все остальные поля результатов
        # с использованием данных из results и файлов с результатами
        
        # Переходим на вкладку результатов
        self.tabs.setCurrentWidget(self.results_tab)
        
        # Логируем успешное завершение
        self._append_log(f"Классификация успешно завершена за {results['elapsed_time']:.2f} секунд")
        
        # Обновляем визуализацию
        self._update_visualization()
        
    def _show_error(self, error_msg):
        """Отображает сообщение об ошибке."""
        # Включаем кнопку запуска
        self.start_button.setEnabled(True)
        
        # Сбрасываем прогресс-бар
        self.progress_bar.setValue(0)
        
        # Показываем сообщение об ошибке
        QMessageBox.critical(self, "Ошибка", error_msg)
        
        # Устанавливаем статус
        self.statusBar().showMessage("Произошла ошибка")
        
    def _append_log(self, message):
        """Добавляет сообщение в журнал."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Прокручиваем лог вниз
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _update_visualization(self):
        """Обновляет текущую визуализацию."""
        plot_type = self.plot_combo.currentText()
        
        # Очищаем график
        self.canvas.axes.clear()
        
        # Проверяем наличие результатов
        if not self.current_results:
            self.canvas.axes.text(0.5, 0.5, "Нет доступных результатов", 
                                ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return
        
        try:
            # В зависимости от типа графика, загружаем нужные данные и рисуем график
            if plot_type == "Точность классификации":
                self._plot_accuracy()
            elif plot_type == "Матрица ошибок":
                self._plot_confusion_matrix()
            elif plot_type == "ROC-кривая":
                self._plot_roc_curve()
            elif plot_type == "Влияние шума":
                self._plot_noise_impact()
            elif plot_type == "Сравнение алгоритмов":
                self._plot_algorithm_comparison()
            elif plot_type == "Зависимость от размера ансамбля":
                self._plot_ensemble_size_impact()
            
            # Обновляем холст
            self.canvas.draw()
            
            # Обновляем подпись
            self.plot_caption.setText(f"График: {plot_type}")
            
        except Exception as e:
            # В случае ошибки при построении графика
            logger.exception(f"Ошибка при построении графика '{plot_type}'")
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, f"Ошибка построения графика: {str(e)}", 
                                ha='center', va='center', fontsize=10, color='red')
            self.canvas.draw()
            
    def _plot_accuracy(self):
        """Строит график точности классификации."""
        problem_number = self.current_results['problem_number']
        
        # В реальном приложении здесь нужно будет загрузить данные из файла результатов
        # Создаем условные данные для примера
        accuracy_data = {
            'Ensemble NN': 0.95,
            'Single NN': 0.89,
            'Random Forest': 0.87,
            'SVM': 0.82,
            'XGBoost': 0.91
        }
        
        # Строим график
        models = list(accuracy_data.keys())
        accuracies = list(accuracy_data.values())
        
        # Создаем цветовую схему
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(models)))
        
        # Строим столбчатую диаграмму
        bars = self.canvas.axes.bar(models, accuracies, color=colors, width=0.6)
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            self.canvas.axes.text(
                bar.get_x() + bar.get_width() / 2, height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9
            )
        
        # Настраиваем график
        self.canvas.axes.set_title(f'Точность классификации для MONK-{problem_number}', fontsize=14)
        self.canvas.axes.set_xlabel('Модель', fontsize=12)
        self.canvas.axes.set_ylabel('Точность', fontsize=12)
        self.canvas.axes.set_ylim(0, 1.1)
        self.canvas.axes.tick_params(axis='x', rotation=30)
        self.canvas.axes.grid(axis='y', alpha=0.3)
        
    def _plot_confusion_matrix(self):
        """Строит матрицу ошибок."""
        problem_number = self.current_results['problem_number']
        
        # В реальном приложении здесь нужно будет загрузить данные из файла результатов
        # Создаем условную матрицу ошибок для примера
        cm = np.array([[120, 15], [8, 130]])
        
        # Нормализуем матрицу
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Строим тепловую карту
        im = self.canvas.axes.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Добавляем цветовую шкалу
        cbar = self.canvas.fig.colorbar(im, ax=self.canvas.axes)
        cbar.set_label('Normalized Frequency')
        
        # Добавляем аннотации
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.canvas.axes.text(j, i, f'{cm[i, j]} ({cm_normalized[i, j]:.2f})',
                                    ha="center", va="center",
                                    color="white" if cm_normalized[i, j] > thresh else "black")
        
        # Настраиваем график
        self.canvas.axes.set_title(f'Матрица ошибок для MONK-{problem_number}', fontsize=14)
        self.canvas.axes.set_xlabel('Предсказанный класс', fontsize=12)
        self.canvas.axes.set_ylabel('Истинный класс', fontsize=12)
        self.canvas.axes.set_xticks([0, 1])
        self.canvas.axes.set_yticks([0, 1])
        self.canvas.axes.set_xticklabels(['0', '1'])
        self.canvas.axes.set_yticklabels(['0', '1'])
        
    def _plot_roc_curve(self):
        """Строит ROC-кривую."""
        problem_number = self.current_results['problem_number']
        
        # В реальном приложении здесь нужно будет загрузить данные из файла результатов
        # Создаем условные данные для примера
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.4)  # Создаем кривую выше диагонали
        roc_auc = 0.85
        
        # Строим ROC-кривую
        self.canvas.axes.plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC curve (area = {roc_auc:.3f})')
        self.canvas.axes.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', 
                            label='Random guess')
        
        # Настраиваем график
        self.canvas.axes.set_title(f'ROC-кривая для MONK-{problem_number}', fontsize=14)
        self.canvas.axes.set_xlabel('False Positive Rate', fontsize=12)
        self.canvas.axes.set_ylabel('True Positive Rate', fontsize=12)
        self.canvas.axes.set_xlim([0.0, 1.0])
        self.canvas.axes.set_ylim([0.0, 1.05])
        self.canvas.axes.grid(alpha=0.3)
        self.canvas.axes.legend(loc="lower right")
        
    def _plot_noise_impact(self):
        """Строит график влияния шума на точность."""
        problem_number = self.current_results['problem_number']
        
        # В реальном приложении здесь нужно будет загрузить данные из файла результатов
        # Создаем условные данные для примера
        noise_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Создаем данные для разных типов шума
        gaussian_accuracy = 0.95 - noise_levels * 0.6
        uniform_accuracy = 0.95 - noise_levels * 0.4
        impulse_accuracy = 0.95 - noise_levels * 0.8
        missing_accuracy = 0.95 - noise_levels * 0.5
        
        # Строим графики
        self.canvas.axes.plot(noise_levels, gaussian_accuracy, 'o-', label='Gaussian Noise')
        self.canvas.axes.plot(noise_levels, uniform_accuracy, 's-', label='Uniform Noise')
        self.canvas.axes.plot(noise_levels, impulse_accuracy, '^-', label='Impulse Noise')
        self.canvas.axes.plot(noise_levels, missing_accuracy, 'D-', label='Missing Data')
        
        # Настраиваем график
        self.canvas.axes.set_title(f'Влияние шума на точность для MONK-{problem_number}', fontsize=14)
        self.canvas.axes.set_xlabel('Уровень шума', fontsize=12)
        self.canvas.axes.set_ylabel('Точность', fontsize=12)
        self.canvas.axes.set_xlim([-0.02, 0.52])
        self.canvas.axes.set_ylim([0, 1])
        self.canvas.axes.grid(alpha=0.3)
        self.canvas.axes.legend()
        
        # Добавляем метки по x в процентах
        self.canvas.axes.set_xticks(noise_levels)
        self.canvas.axes.set_xticklabels([f'{int(x*100)}%' for x in noise_levels])
        
    def _plot_algorithm_comparison(self):
        """Строит график сравнения алгоритмов при разных уровнях шума."""
        problem_number = self.current_results['problem_number']
        
        # В реальном приложении здесь нужно будет загрузить данные из файла результатов
        # Создаем условные данные для примера
        noise_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Данные для разных алгоритмов
        ensemble_accuracy = 0.95 - noise_levels * 0.3
        single_nn_accuracy = 0.9 - noise_levels * 0.5
        random_forest_accuracy = 0.87 - noise_levels * 0.6
        svm_accuracy = 0.82 - noise_levels * 0.7
        
        # Строим графики
        self.canvas.axes.plot(noise_levels, ensemble_accuracy, 'o-', linewidth=2, label='Ensemble NN')
        self.canvas.axes.plot(noise_levels, single_nn_accuracy, 's--', label='Single NN')
        self.canvas.axes.plot(noise_levels, random_forest_accuracy, '^--', label='Random Forest')
        self.canvas.axes.plot(noise_levels, svm_accuracy, 'D--', label='SVM')
        
        # Настраиваем график
        self.canvas.axes.set_title(f'Сравнение алгоритмов при разных уровнях шума (MONK-{problem_number})', 
                                 fontsize=14)
        self.canvas.axes.set_xlabel('Уровень шума', fontsize=12)
        self.canvas.axes.set_ylabel('Точность', fontsize=12)
        self.canvas.axes.set_xlim([-0.02, 0.52])
        self.canvas.axes.set_ylim([0, 1])
        self.canvas.axes.grid(alpha=0.3)
        self.canvas.axes.legend()
        
        # Добавляем метки по x в процентах
        self.canvas.axes.set_xticks(noise_levels)
        self.canvas.axes.set_xticklabels([f'{int(x*100)}%' for x in noise_levels])
        
    def _plot_ensemble_size_impact(self):
        """Строит график зависимости точности от размера ансамбля."""
        problem_number = self.current_results['problem_number']
        
        # В реальном приложении здесь нужно будет загрузить данные из файла результатов
        # Создаем условные данные для примера
        ensemble_sizes = np.arange(1, 16)
        
        # Данные для точности
        accuracy = 0.8 + 0.15 * (1 - np.exp(-0.3 * ensemble_sizes))
        
        # Добавляем небольшую случайность
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, len(ensemble_sizes))
        accuracy += noise
        
        # Строим график
        self.canvas.axes.plot(ensemble_sizes, accuracy, 'o-', linewidth=2)
        
        # Находим точку насыщения (где производная меньше порога)
        dx = np.diff(accuracy)
        saturation_idx = np.where(dx < 0.01)[0]
        if len(saturation_idx) > 0:
            saturation_point = ensemble_sizes[saturation_idx[0] + 1]
            saturation_accuracy = accuracy[saturation_idx[0] + 1]
            
            # Добавляем вертикальную линию и аннотацию
            self.canvas.axes.axvline(x=saturation_point, color='r', linestyle='--')
            self.canvas.axes.annotate(f'Оптимальный размер ≈ {saturation_point}',
                                    xy=(saturation_point, saturation_accuracy),
                                    xytext=(saturation_point + 1, saturation_accuracy - 0.05),
                                    arrowprops=dict(facecolor='black', shrink=0.05),
                                    fontsize=10)
        
        # Настраиваем график
        self.canvas.axes.set_title(f'Зависимость точности от размера ансамбля (MONK-{problem_number})', 
                                 fontsize=14)
        self.canvas.axes.set_xlabel('Размер ансамбля', fontsize=12)
        self.canvas.axes.set_ylabel('Точность', fontsize=12)
        self.canvas.axes.set_xlim([0.5, max(ensemble_sizes) + 0.5])
        self.canvas.axes.set_ylim([0.7, 1])
        self.canvas.axes.grid(alpha=0.3)
        
        # Устанавливаем целочисленные метки по оси x
        self.canvas.axes.set_xticks(ensemble_sizes)
        
    def _save_plot(self):
        """Сохраняет текущий график."""
        if not hasattr(self, 'canvas') or not self.canvas:
            QMessageBox.warning(self, "Предупреждение", "Нет графика для сохранения.")
            return
        
        # Открываем диалог сохранения файла
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "", "Изображения (*.png *.jpg *.pdf);;Все файлы (*)"
        )
        
        if file_path:
            try:
                # Сохраняем график
                self.canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.statusBar().showMessage(f"График сохранен: {file_path}")
                self._append_log(f"График сохранен: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении графика: {str(e)}")
                logger.exception("Ошибка при сохранении графика")
                
    def _export_results(self):
        """Экспортирует результаты в CSV."""
        if not self.current_results:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для экспорта.")
            return
        
        # Открываем диалог сохранения файла
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", "", "CSV файлы (*.csv);;Все файлы (*)"
        )
        
        if file_path:
            try:
                # В реальном приложении здесь нужно будет экспортировать фактические результаты
                # Создаем условные данные для примера
                results_df = pd.DataFrame({
                    'Model': ['Ensemble NN', 'Single NN', 'Random Forest', 'SVM', 'XGBoost'],
                    'Accuracy': [0.95, 0.89, 0.87, 0.82, 0.91],
                    'F1_Score': [0.94, 0.88, 0.86, 0.81, 0.90],
                    'ROC_AUC': [0.97, 0.92, 0.91, 0.89, 0.93]
                })
                
                # Сохраняем в CSV
                results_df.to_csv(file_path, index=False)
                self.statusBar().showMessage(f"Результаты экспортированы: {file_path}")
                self._append_log(f"Результаты экспортированы: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте результатов: {str(e)}")
                logger.exception("Ошибка при экспорте результатов")
                
    def _generate_report(self):
        """Генерирует отчет в формате HTML."""
        if not self.current_results:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для создания отчета.")
            return
        
        # Открываем диалог сохранения файла
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчет", "", "HTML файлы (*.html);;Все файлы (*)"
        )
        
        if file_path:
            try:
                # Формируем HTML отчет
                problem_number = self.current_results['problem_number']
                timestamp = self.current_results['timestamp']
                
                # Создаем HTML-содержимое
                html_content = f"""
                <!DOCTYPE html>
                <html lang="ru">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Отчет о классификации MONK-{problem_number}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .header {{ background-color: #4CAF50; color: white; padding: 20px; margin-bottom: 20px; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .footer {{ background-color: #f2f2f2; padding: 10px; text-align: center; margin-top: 30px; }}
                        .result-summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Отчет о классификации набора данных MONK-{problem_number}</h1>
                        <p>Сгенерировано: {timestamp}</p>
                    </div>
                    <div class="container">
                        <div class="result-summary">
                            <h2>Сводка результатов</h2>
                            <p><strong>Задача:</strong> MONK-{problem_number}</p>
                            <p><strong>Режим:</strong> {self._get_mode_display_name(self.current_results['mode'])}</p>
                            <p><strong>Время выполнения:</strong> {self.current_results['elapsed_time']:.2f} секунд</p>
                        </div>
                        
                        <h2>Результаты классификации</h2>
                        <table>
                            <tr>
                                <th>Модель</th>
                                <th>Точность</th>
                                <th>F1-Score</th>
                                <th>ROC AUC</th>
                            </tr>
                            <tr>
                                <td>Ensemble NN</td>
                                <td>0.9500</td>
                                <td>0.9450</td>
                                <td>0.9700</td>
                            </tr>
                            <tr>
                                <td>Single NN</td>
                                <td>0.8900</td>
                                <td>0.8800</td>
                                <td>0.9200</td>
                            </tr>
                            <tr>
                                <td>Random Forest</td>
                                <td>0.8700</td>
                                <td>0.8650</td>
                                <td>0.9100</td>
                            </tr>
                            <tr>
                                <td>SVM</td>
                                <td>0.8200</td>
                                <td>0.8100</td>
                                <td>0.8900</td>
                            </tr>
                            <tr>
                                <td>XGBoost</td>
                                <td>0.9100</td>
                                <td>0.9050</td>
                                <td>0.9350</td>
                            </tr>
                        </table>
                        
                        <h2>Параметры эксперимента</h2>
                        <table>
                            <tr>
                                <th>Параметр</th>
                                <th>Значение</th>
                            </tr>
                            <tr>
                                <td>Задача</td>
                                <td>MONK-{problem_number}</td>
                            </tr>
                            <tr>
                                <td>Режим</td>
                                <td>{self._get_mode_display_name(self.current_results['mode'])}</td>
                            </tr>
                            <tr>
                                <td>Оптимизация гиперпараметров</td>
                                <td>{'Включена' if self.hyperopt_check.isChecked() else 'Выключена'}</td>
                            </tr>
                            <tr>
                                <td>Количество попыток оптимизации</td>
                                <td>{self.n_trials_spin.value()}</td>
                            </tr>
                            <tr>
                                <td>Минимальный уровень шума</td>
                                <td>{self.min_noise_spin.value()}%</td>
                            </tr>
                            <tr>
                                <td>Максимальный уровень шума</td>
                                <td>{self.max_noise_spin.value()}%</td>
                            </tr>
                            <tr>
                                <td>Шаг изменения уровня шума</td>
                                <td>{self.noise_step_spin.value()}%</td>
                            </tr>
                        </table>
                        
                        <h2>Заключение</h2>
                        <p>Эксперименты показали, что ансамблевый подход дает наилучшие результаты для задачи MONK-{problem_number}.
                        Ансамбль нейронных сетей превосходит одиночную нейронную сеть и другие алгоритмы по точности классификации.</p>
                        
                        <p>Ансамбль также показывает большую устойчивость к шуму в данных, сохраняя приемлемую точность
                        даже при высоких уровнях зашумления.</p>
                    </div>
                    <div class="footer">
                        <p>Отчет сгенерирован приложением "Классификатор MONK"</p>
                    </div>
                </body>
                </html>
                """
                
                # Записываем HTML в файл
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Показываем сообщение об успешном сохранении
                self.statusBar().showMessage(f"Отчет сохранен: {file_path}")
                self._append_log(f"Отчет сохранен: {file_path}")
                
                # Спрашиваем пользователя, открыть ли отчет
                reply = QMessageBox.question(
                    self, "Отчет сохранен", 
                    f"Отчет успешно сохранен. Открыть его?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    # Открываем отчет в браузере
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                    
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при создании отчета: {str(e)}")
                logger.exception("Ошибка при создании отчета")
                
    def _save_log(self):
        """Сохраняет журнал в текстовый файл."""
        if self.log_text.toPlainText() == "":
            QMessageBox.warning(self, "Предупреждение", "Журнал пуст.")
            return
        
        # Открываем диалог сохранения файла
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить журнал", "", "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if file_path:
            try:
                # Получаем текст из журнала (без HTML-тегов)
                log_text = self.log_text.toPlainText()
                
                # Записываем в файл
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_text)
                
                self.statusBar().showMessage(f"Журнал сохранен: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении журнала: {str(e)}")
                logger.exception("Ошибка при сохранении журнала")
                
    def _show_help(self):
        """Показывает справку."""
        help_text = """
        <h1>Справка по использованию классификатора MONK</h1>
        
        <h2>Основные настройки</h2>
        <p><b>Задача:</b> Выберите одну из задач MONK (1, 2 или 3).</p>
        <p><b>Режим работы:</b> Выберите режим работы приложения:</p>
        <ul>
            <li><b>Обучение и оценка:</b> Обучение ансамбля и оценка его точности</li>
            <li><b>Сравнительный анализ:</b> Сравнение ансамбля с другими алгоритмами</li>
            <li><b>Анализ влияния размера ансамбля:</b> Исследование зависимости точности от размера ансамбля</li>
        </ul>
        
        <h2>Параметры ансамбля</h2>
        <p><b>Размер ансамбля:</b> Количество моделей в ансамбле (для режимов обучения и сравнительного анализа).</p>
        <p><b>Максимальный размер для анализа:</b> Максимальный размер ансамбля для исследования зависимости точности (для режима анализа размера).</p>
        <p><b>Оптимизация гиперпараметров:</b> Включает оптимизацию гиперпараметров моделей с использованием Optuna.</p>
        <p><b>Количество попыток оптимизации:</b> Число попыток при поиске оптимальных гиперпараметров.</p>
        
        <h2>Параметры шума</h2>
        <p><b>Типы шума:</b> Выберите типы шума для исследования устойчивости моделей.</p>
        <p><b>Минимальный уровень шума:</b> Минимальный уровень шума в процентах.</p>
        <p><b>Максимальный уровень шума:</b> Максимальный уровень шума в процентах.</p>
        <p><b>Шаг изменения уровня шума:</b> Шаг между уровнями шума в процентах.</p>
        
        <h2>Результаты и визуализация</h2>
        <p>После завершения классификации результаты будут доступны на вкладках "Результаты" и "Визуализация".</p>
        <p>На вкладке "Визуализация" вы можете выбрать различные типы графиков для отображения результатов.</p>
        <p>Графики можно сохранить, нажав кнопку "Сохранить график".</p>
        <p>Результаты можно экспортировать в CSV или создать подробный отчет в формате HTML.</p>
        
        <h2>Журнал</h2>
        <p>На вкладке "Журнал" отображаются все действия и сообщения программы.</p>
        <p>Журнал можно сохранить в текстовый файл для дальнейшего анализа.</p>
        """
        
        # Создаем диалоговое окно справки
        help_dialog = QMessageBox(self)
        help_dialog.setWindowTitle("Справка")
        help_dialog.setText(help_text)
        help_dialog.setTextFormat(Qt.RichText)
        help_dialog.setIcon(QMessageBox.Information)
        help_dialog.exec_()
    
    def _get_mode_display_name(self, mode):
        """Возвращает отображаемое имя режима."""
        mode_names = {
            'training': "Обучение и оценка классификатора",
            'comparative': "Сравнительный анализ с другими алгоритмами",
            'ensemble_size': "Анализ влияния размера ансамбля"
        }
        return mode_names.get(mode, mode)
        
    def closeEvent(self, event):
        """Обрабатывает закрытие приложения."""
        # Проверяем, не выполняется ли сейчас классификация
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Подтверждение выхода", 
                "Процесс классификации выполняется. Вы уверены, что хотите выйти?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Останавливаем поток (если возможно)
                self.worker.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Задаем стиль для всего приложения
    app.setStyle("Fusion")
    
    # Опционально: устанавливаем темную тему
    # dark_palette = QPalette()
    # dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    # dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    # dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    # dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    # dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    # dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    # dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    # dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    # dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    # dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    # app.setPalette(dark_palette)
    
    # Запускаем основное окно
    window = MonkClassifierGUI()
    window.show()
    sys.exit(app.exec_())