import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation, Add, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import logging
import warnings
from typing import Dict, List, Tuple, Union, Optional, Callable
import pickle
import json
import time
from pathlib import Path

# Подавляем предупреждения для более чистого вывода
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimal_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OptimalClassifier")

# Установка seed для воспроизводимости
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Создание директорий для моделей и результатов
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

class DataLoader:
    """Класс для загрузки и предобработки данных MONK."""
    
    @staticmethod
    def load_monks_data(problem_number: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Загружает данные MONK и выполняет расширенную предобработку признаков.
        
        Args:
            problem_number: Номер задачи MONK (1, 2 или 3)
            
        Returns:
            X_train, y_train, X_test, y_test: Обработанные признаки и метки для обучения и тестирования
        """
        logger.info(f"Загрузка данных MONK-{problem_number}")
        
        try:
            # Определение имен столбцов
            columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
            
            # Загрузка обучающих данных
            train_file = f"Dataset\\monks-{problem_number}.train"
            if not os.path.exists(train_file):
                logger.error(f"Файл набора данных не найден: {train_file}")
                raise FileNotFoundError(f"Файл набора данных не найден: {train_file}")
                
            train_data = pd.read_csv(train_file, sep=' ', names=columns)
            train_data = train_data.drop('id', axis=1)  # Удаление ID столбца
            
            # Загрузка тестовых данных
            test_file = f"Dataset\\monks-{problem_number}.test"
            if not os.path.exists(test_file):
                logger.error(f"Файл набора данных не найден: {test_file}")
                raise FileNotFoundError(f"Файл набора данных не найден: {test_file}")
                
            test_data = pd.read_csv(test_file, sep=' ', names=columns)
            test_data = test_data.drop('id', axis=1)  # Удаление ID столбца
            
            # Выполнение обогащения признаков в зависимости от проблемы
            train_data = DataLoader.enhance_features(train_data, problem_number)
            test_data = DataLoader.enhance_features(test_data, problem_number)
            
            # Извлечение признаков и целевой переменной
            X_train = train_data.drop('class', axis=1)
            y_train = train_data['class']
            
            X_test = test_data.drop('class', axis=1)
            y_test = test_data['class']
            
            logger.info(f"Данные успешно загружены: {len(X_train)} обучающих и {len(X_test)} тестовых примеров")
            logger.info(f"Признаки после расширения: {X_train.columns.tolist()}")
            
            # Преобразование в numpy массивы с правильными типами данных
            X_train_array = X_train.values.astype(np.float32)
            y_train_array = y_train.values.astype(np.float32)
            X_test_array = X_test.values.astype(np.float32)
            y_test_array = y_test.values.astype(np.float32)
            
            return X_train_array, y_train_array, X_test_array, y_test_array
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    @staticmethod
    def enhance_features(data: pd.DataFrame, problem_number: int) -> pd.DataFrame:
        """
        Расширяет набор признаков специфично для каждой задачи MONK.
        
        Args:
            data: DataFrame с исходными признаками
            problem_number: Номер задачи MONK (1, 2, 3)
            
        Returns:
            DataFrame с расширенным набором признаков
        """
        df = data.copy()
        
        # Одноразовое кодирование категориальных признаков
        for col in [f'a{i}' for i in range(1, 7)]:
            unique_values = df[col].unique()
            for val in unique_values:
                df[f'{col}_eq_{val}'] = (df[col] == val).astype(float)  # Используем float вместо int
        
        # Специфичное для задачи расширение
        if problem_number == 1:
            # MONK-1: (a1 = a2) OR (a5 = 1)
            df['a1_eq_a2'] = (df['a1'] == df['a2']).astype(float)
            df['a5_eq_1'] = (df['a5'] == 1).astype(float)
            df['rule_1_satisfied'] = ((df['a1'] == df['a2']) | (df['a5'] == 1)).astype(float)
            
            # Используем логические операции с булевыми значениями (не float)
            a1_eq_a2_bool = (df['a1'] == df['a2'])
            a5_eq_1_bool = (df['a5'] == 1)
            
            # Добавляем компоненты правила - с правильным приведением типов
            df['rule_component_1'] = (a1_eq_a2_bool & (~a5_eq_1_bool)).astype(float)
            df['rule_component_2'] = ((~a1_eq_a2_bool) & a5_eq_1_bool).astype(float)
            df['rule_component_3'] = (a1_eq_a2_bool & a5_eq_1_bool).astype(float)
            
            # Добавляем усиленный признак
            df['rule_1_amplified'] = df['rule_1_satisfied'] ** 2
            
            # Дополнительные полезные признаки
            for i in range(1, 7):
                for j in range(i + 1, 7):
                    df[f'a{i}_eq_a{j}'] = (df[f'a{i}'] == df[f'a{j}']).astype(float)
        
        elif problem_number == 2:
            # MONK-2: Ровно два атрибута имеют значение 1
            # Эта проблема сложнее, добавляем больше признаков
            
            # Подсчет атрибутов со значением 1
            df['count_1s'] = ((df['a1'] == 1) + (df['a2'] == 1) + 
                            (df['a3'] == 1) + (df['a4'] == 1) + 
                            (df['a5'] == 1) + (df['a6'] == 1)).astype(float)
            
            # Ключевой признак - флаг "ровно 2"
            df['exactly_two_1s'] = (df['count_1s'] == 2).astype(float)
            
            # Добавляем все возможные пары атрибутов со значением 1
            for i in range(1, 6):
                for j in range(i+1, 7):
                    df[f'a{i}_and_a{j}_eq_1'] = ((df[f'a{i}'] == 1) & (df[f'a{j}'] == 1)).astype(float)
            
            # Добавляем признаки для случаев, когда условие не выполняется
            df['less_than_two_1s'] = (df['count_1s'] < 2).astype(float)
            df['more_than_two_1s'] = (df['count_1s'] > 2).astype(float)
            
            # Добавляем расстояние до идеальных шаблонов
            mask_exactly_two = df['exactly_two_1s'] == 1
            df['distance_from_pattern'] = 0.0  # Используем float
            
            for i in range(1, 7):
                # Для атрибутов со значением 1, когда всего единиц > 2
                df.loc[df['more_than_two_1s'] == 1, 'distance_from_pattern'] += \
                    df.loc[df['more_than_two_1s'] == 1, f'a{i}'] * (df.loc[df['more_than_two_1s'] == 1, 'count_1s'] - 2)
                
                # Для атрибутов со значением 0, когда всего единиц < 2
                df.loc[df['less_than_two_1s'] == 1, 'distance_from_pattern'] += \
                    (1 - df.loc[df['less_than_two_1s'] == 1, f'a{i}']) * (2 - df.loc[df['less_than_two_1s'] == 1, 'count_1s'])
        
        elif problem_number == 3:
            # MONK-3: (a5 = 3 AND a4 = 1) OR (a5 ≠ 4 AND a2 ≠ 3)
            df['a5_eq_3_and_a4_eq_1'] = ((df['a5'] == 3) & (df['a4'] == 1)).astype(float)
            df['a5_neq_4_and_a2_neq_3'] = ((df['a5'] != 4) & (df['a2'] != 3)).astype(float)
            df['rule_3_satisfied'] = ((df['a5'] == 3) & (df['a4'] == 1)) | ((df['a5'] != 4) & (df['a2'] != 3))
            df['rule_3_satisfied'] = df['rule_3_satisfied'].astype(float)
            
            # Создаем булевы версии для безопасной логической операции NOT
            rule_part1_bool = (df['a5'] == 3) & (df['a4'] == 1)
            rule_part2_bool = (df['a5'] != 4) & (df['a2'] != 3)
            
            # Добавляем детализированные признаки
            df['rule_part1_only'] = (rule_part1_bool & (~rule_part2_bool)).astype(float)
            df['rule_part2_only'] = ((~rule_part1_bool) & rule_part2_bool).astype(float)
            df['rule_both_parts'] = (rule_part1_bool & rule_part2_bool).astype(float)
            
            # Квадрат признака правила для его усиления
            df['rule_3_amplified'] = df['rule_3_satisfied'] ** 2
            
            # Добавляем комбинации для первой части правила
            df['a5_eq_3'] = (df['a5'] == 3).astype(float)
            df['a4_eq_1'] = (df['a4'] == 1).astype(float)
            
            # Добавляем комбинации для второй части правила
            df['a5_neq_4'] = (df['a5'] != 4).astype(float)
            df['a2_neq_3'] = (df['a2'] != 3).astype(float)
        
        # Для всех проблем добавляем попарные взаимодействия
        for i in range(1, 6):
            for j in range(i+1, 7):
                # Разница между значениями атрибутов
                df[f'diff_a{i}_a{j}'] = (df[f'a{i}'] - df[f'a{j}']).astype(float)
                
                # Произведение атрибутов (для обнаружения нелинейных взаимодействий)
                df[f'prod_a{i}_a{j}'] = (df[f'a{i}'] * df[f'a{j}']).astype(float)
        
        return df

class HyperOptimizer:
    """Оптимизация гиперпараметров с использованием Optuna."""
    
    @staticmethod
    def optimize_hyperparameters(X_train, y_train, X_val, y_val, problem_number, n_trials=100):
        """
        Выполняет оптимизацию гиперпараметров для нейронной сети.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
            problem_number: Номер задачи MONK для специфичной настройки
            n_trials: Количество попыток оптимизации
            
        Returns:
            Лучшие найденные гиперпараметры
        """
        logger.info(f"Запуск оптимизации гиперпараметров для MONK-{problem_number} с {n_trials} попытками")
        
        # Проверяем и приводим данные к правильному типу
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        
        # Определяем целевую функцию для оптимизации
        def objective(trial):
            # Подбираем гиперпараметры с помощью Optuna
            params = {
                # Параметры архитектуры сети
                'n_layers': trial.suggest_int('n_layers', 2, 5),
                'units_first': trial.suggest_categorical('units_first', [32, 64, 128, 256]),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'swish', 'tanh']),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'use_residual': trial.suggest_categorical('use_residual', [True, False]),
                
                # Параметры обучения
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
                
                # Регуляризация
                'use_regularization': trial.suggest_categorical('use_regularization', [True, False]),
                'l1_factor': trial.suggest_float('l1_factor', 1e-6, 1e-3, log=True) if trial.params.get('use_regularization', False) else 0,
                'l2_factor': trial.suggest_float('l2_factor', 1e-6, 1e-3, log=True) if trial.params.get('use_regularization', False) else 0,
            }
            
            # Создаем модель с выбранными гиперпараметрами
            model = HyperOptimizer._create_model_from_params(params, X_train.shape[1])
            
            # Компилируем модель
            if params['optimizer'] == 'adam':
                optimizer = Adam(learning_rate=float(params['learning_rate']))
            elif params['optimizer'] == 'rmsprop':
                optimizer = RMSprop(learning_rate=float(params['learning_rate']))
            else:
                optimizer = SGD(learning_rate=float(params['learning_rate']), momentum=0.9)
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Колбэки для обучения
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                TFKerasPruningCallback(trial, 'val_accuracy')
            ]
            
            # Важный момент: batch_size должен быть int, не float
            batch_size = int(params['batch_size'])
            
            # Обучаем модель
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,  # Большое количество эпох, но с ранней остановкой
                    batch_size=batch_size,  # Используем int
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Возвращаем метрику для оптимизации (последнее значение валидационной точности)
                return float(history.history['val_accuracy'][-1])
            except Exception as e:
                logger.error(f"Ошибка при обучении модели: {str(e)}")
                raise
        
        # Создаем исследование Optuna с настройкой на максимизацию точности
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(
            direction='maximize',
            pruner=pruner,
            study_name=f'monk{problem_number}_optimization'
        )
        
        # Запускаем оптимизацию
        study.optimize(objective, n_trials=n_trials)
        
        # Логируем результаты
        logger.info(f"Лучшие гиперпараметры для MONK-{problem_number}: {study.best_params}")
        logger.info(f"Достигнутая точность: {study.best_value:.4f}")
        
        # Сохраняем результаты исследования
        with open(RESULTS_DIR / f"hyperopt_results_monk{problem_number}.pkl", "wb") as f:
            pickle.dump(study, f)
        
        # Визуализация результатов
        try:
            # Импортируем нужные модули здесь, чтобы они не были обязательными
            import plotly
            
            # Сохраняем график важности параметров
            param_importances = optuna.visualization.plot_param_importances(study)
            fig_importance = param_importances.update_layout(
                title=f"Важность параметров для MONK-{problem_number}", 
                width=1000, 
                height=600
            )
            fig_importance.write_image(RESULTS_DIR / f"param_importance_monk{problem_number}.png")
            
            # Сохраняем оптимизационную историю
            optimization_history = optuna.visualization.plot_optimization_history(study)
            fig_history = optimization_history.update_layout(
                title=f"История оптимизации для MONK-{problem_number}", 
                width=1000, 
                height=600
            )
            fig_history.write_image(RESULTS_DIR / f"optimization_history_monk{problem_number}.png")
        except Exception as viz_error:
            logger.warning(f"Не удалось создать визуализации оптимизации: {str(viz_error)}")
        
        return study.best_params
    
    @staticmethod
    def _create_model_from_params(params, input_dim):
        """
        Создает модель нейронной сети на основе словаря параметров.
        
        Args:
            params: Словарь с гиперпараметрами
            input_dim: Размерность входных данных
            
        Returns:
            Скомпилированная модель Keras
        """
        # Входной слой
        inputs = Input(shape=(input_dim,))
        
        # Определяем регуляризатор, если используется
        regularizer = None
        if params['use_regularization']:
            regularizer = l1_l2(l1=float(params['l1_factor']), l2=float(params['l2_factor']))
        
        # Первый слой
        x = Dense(
            int(params['units_first']),  # Убедимся, что это int
            kernel_regularizer=regularizer
        )(inputs)
        
        if params['activation'] == 'leaky_relu':
            x = LeakyReLU(alpha=0.1)(x)
        else:
            x = Activation(params['activation'])(x)
            
        if params['use_batch_norm']:
            x = BatchNormalization()(x)
            
        if params['dropout_rate'] > 0:
            x = Dropout(float(params['dropout_rate']))(x)
        
        # Скрытые слои с уменьшающимся количеством нейронов
        for i in range(params['n_layers'] - 1):
            # Постепенно уменьшаем количество нейронов
            units = int(params['units_first']) // (2 ** (i + 1))
            units = max(units, 16)  # Минимальное количество нейронов
            
            # Запоминаем вход слоя для возможного соединения
            layer_input = x
            
            # Добавляем полносвязный слой
            x = Dense(
                units, 
                kernel_regularizer=regularizer
            )(x)
            
            # Активация
            if params['activation'] == 'leaky_relu':
                x = LeakyReLU(alpha=0.1)(x)
            else:
                x = Activation(params['activation'])(x)
                
            # Нормализация и дропаут
            if params['use_batch_norm']:
                x = BatchNormalization()(x)
                
            if params['dropout_rate'] > 0:
                x = Dropout(float(params['dropout_rate']))(x)
            
            # Добавляем остаточное соединение, если выбрано и размерности совпадают
            if params['use_residual'] and i > 0 and layer_input.shape[-1] == x.shape[-1]:
                x = Add()([layer_input, x])
        
        # Выходной слой
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Создаем модель
        model = Model(inputs, outputs)
        
        return model

class EnsembleClassifier:
    """Построение ансамбля разнообразных моделей для повышения точности."""
    
    def __init__(self, problem_number, base_params=None, num_models=5):
        """
        Инициализация классификатора.
        
        Args:
            problem_number: Номер задачи MONK
            base_params: Базовые гиперпараметры для вариации
            num_models: Количество моделей в ансамбле
        """
        self.problem_number = problem_number
        self.base_params = base_params
        self.num_models = num_models
        self.models = []
        self.input_shape = None
        
        logger.info(f"Инициализирован EnsembleClassifier для MONK-{problem_number} с {num_models} моделями")
    
    def build_ensemble(self, X_train, y_train, X_val, y_val):
        """
        Строит ансамбль моделей с разными архитектурами.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
        """
        logger.info(f"Построение ансамбля из {self.num_models} моделей")
        self.input_shape = X_train.shape[1]
        
        # Проверяем типы данных - важно для TensorFlow
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        
        # Создаем директорию для моделей ансамбля
        ensemble_dir = MODELS_DIR / f"monk{self.problem_number}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        
        # УЛУЧШЕНИЕ: Более разнообразные архитектуры для моделей
        activations = ['relu', 'elu', 'selu', 'tanh', 'sigmoid']
        
        # Значительно разнообразим конфигурации слоев
        if self.problem_number == 2:  # Более сложная задача MONK-2
            layer_configs = [
                [int(self.base_params['units_first'] * 4), int(self.base_params['units_first'] * 2), int(self.base_params['units_first'])],  # Широкая сеть
                [int(self.base_params['units_first']), int(self.base_params['units_first'] * 2), int(self.base_params['units_first'])],  # Расширяющаяся в середине
                [int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5)],  # Узкая и глубокая
                [int(self.base_params['units_first'] * 2), int(self.base_params['units_first'] * 0.5)],  # Резкое сужение
                [int(self.base_params['units_first']), int(self.base_params['units_first'] * 0.75), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.25)]  # Плавное сужение
            ]
        else:  # Более простые задачи MONK-1 и MONK-3
            layer_configs = [
                [int(self.base_params['units_first'] * 2), int(self.base_params['units_first'])],  # Широкая и короткая
                [int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5)],  # Узкая и глубокая
                [int(self.base_params['units_first']), int(self.base_params['units_first'] * 2), int(self.base_params['units_first'])],  # Бутылочная структура
                [int(self.base_params['units_first'] * 1.5), int(self.base_params['units_first'] * 0.3)],  # Большое сужение
                [int(self.base_params['units_first']), int(self.base_params['units_first']), int(self.base_params['units_first'] * 0.5)]  # Плато и спуск
            ]
        
        # Массив для хранения валидационных точностей моделей
        val_accuracies = []
        
        # Для каждой модели в ансамбле
        for i in range(self.num_models):
            # Создаем вариацию гиперпараметров
            params = self.base_params.copy()
            
            # Выбираем вариации для разнообразия
            params['activation'] = activations[i % len(activations)]
            
            # УЛУЧШЕНИЕ: Более разнообразные параметры обучения
            # Изменяем скорость обучения более агрессивно
            learning_rate_factor = 0.1 + np.random.exponential(1.0)  # Экспоненциальное распределение для большего разнообразия
            params['learning_rate'] = float(self.base_params['learning_rate'] * learning_rate_factor)
            
            # Существенно варьируем размер батча
            batch_size_options = [8, 16, 32, 64, 128]
            params['batch_size'] = int(batch_size_options[i % len(batch_size_options)])
            
            # Варьируем dropout для разнообразия регуляризации
            params['dropout_rate'] = float(np.clip(np.random.normal(
                self.base_params.get('dropout_rate', 0.3), 0.1), 0.1, 0.6))
            
            # Иногда используем L1, иногда L2, иногда комбинацию, иногда без регуляризации
            regularization_type = i % 4
            if regularization_type == 0:  # Без регуляризации
                params['use_regularization'] = False
                params['l1_factor'] = 0.0
                params['l2_factor'] = 0.0
            elif regularization_type == 1:  # Только L1
                params['use_regularization'] = True
                params['l1_factor'] = float(np.random.choice([1e-6, 1e-5, 1e-4]))
                params['l2_factor'] = 0.0
            elif regularization_type == 2:  # Только L2
                params['use_regularization'] = True
                params['l1_factor'] = 0.0
                params['l2_factor'] = float(np.random.choice([1e-6, 1e-5, 1e-4]))
            else:  # L1 + L2
                params['use_regularization'] = True
                params['l1_factor'] = float(np.random.choice([1e-6, 1e-5]))
                params['l2_factor'] = float(np.random.choice([1e-6, 1e-5]))
            
            # Выбираем конфигурацию слоев
            layer_config = layer_configs[i % len(layer_configs)]
            
            logger.info(f"Обучение модели {i+1}/{self.num_models} с архитектурой {layer_config}, "
                        f"активация={params['activation']}, LR={params['learning_rate']:.6f}")
            
            # Создаем модель с текущей конфигурацией
            model = self._create_model_with_layers(params, layer_config)
            
            # Выбираем оптимизатор - больше разнообразия
            opt_choice = i % 4
            if opt_choice == 0:
                optimizer = Adam(learning_rate=float(params['learning_rate']))
            elif opt_choice == 1:
                optimizer = RMSprop(learning_rate=float(params['learning_rate']))
            elif opt_choice == 2:
                optimizer = SGD(learning_rate=float(params['learning_rate']), momentum=0.9)
            elif opt_choice == 3:
                # Добавляем AdamW для еще большего разнообразия, если доступен
                try:
                    # Пробуем новый путь (TF 2.11+)
                    optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=float(params['learning_rate']),
                        weight_decay=0.001
                    )
                except AttributeError:
                    try:
                        # Пробуем старый путь (до TF 2.11)
                        optimizer = tf.keras.optimizers.experimental.AdamW(
                            learning_rate=float(params['learning_rate']),
                            weight_decay=0.001
                        )
                    except (AttributeError, ImportError):
                        # Если AdamW недоступен, используем обычный Adam
                        logger.warning("AdamW не найден, используем Adam с L2-регуляризацией")
                        optimizer = Adam(
                            learning_rate=float(params['learning_rate']),
                            beta_1=0.9,
                            beta_2=0.999
                        )
            else:
                # Добавляем AdamW для еще большего разнообразия
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=float(params['learning_rate']),
                    weight_decay=0.001
                )
            
            # Компилируем модель
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Колбэки для обучения
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=30,  # Увеличенная patience для более стабильного обучения
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6
                ),
                # Сохраняем лучшую модель
                ModelCheckpoint(
                    filepath=str(ensemble_dir / f"model_{i+1}.h5"),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=0
                )
            ]
            
            # Добавляем разнообразие в схему обучения
            if i % 3 == 0:
                # Стандартное обучение с ранней остановкой
                max_epochs = 300  # Увеличиваем количество эпох
            elif i % 3 == 1:
                # Обучение с циклической скоростью обучения
                def cosine_annealing(epoch):
                    max_epochs = 300
                    return params['learning_rate'] * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
                    
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_annealing))
                max_epochs = 300
            else:
                # Обучение с перезапусками (периодические сбросы learning rate)
                def step_decay(epoch):
                    initial_lr = params['learning_rate']
                    drop = 0.5
                    epochs_drop = 50.0
                    lr = initial_lr * np.power(drop, np.floor(epoch/epochs_drop))
                    return lr
                    
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(step_decay))
                max_epochs = 300
            
            # Обучаем модель
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=max_epochs,  # Больше эпох, но с ранней остановкой
                    batch_size=int(params['batch_size']),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Загружаем лучшую модель
                model = load_model(ensemble_dir / f"model_{i+1}.h5")
                
                # Оцениваем на валидационной выборке
                _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                val_accuracies.append(float(val_accuracy))
                
                # Сохраняем метаданные модели
                model_meta = {
                    'params': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
                    'layer_config': [int(u) for u in layer_config],
                    'val_accuracy': float(max(history.history['val_accuracy'])),
                    'epochs_trained': len(history.history['loss'])
                }
                with open(ensemble_dir / f"model_{i+1}_meta.json", 'w') as f:
                    json.dump(model_meta, f, indent=2)
                
                # Добавляем модель в ансамбль
                self.models.append(model)
                
                # Выводим информацию о точности
                logger.info(f"Модель {i+1} обучена: val_accuracy={val_accuracy:.4f}, epochs={model_meta['epochs_trained']}")
            
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {i+1}: {str(e)}")
                # Продолжаем со следующей моделью
                continue
        
        # Сохраняем валидационные точности для взвешенного голосования
        self.val_accuracies = val_accuracies
        
        # УЛУЧШЕНИЕ: Обучение с отбором - удаляем самые слабые модели
        if len(self.models) > 3:  # Проверяем, что у нас достаточно моделей для отбора
            # Сортируем модели по валидационной точности
            sorted_indices = np.argsort(val_accuracies)
            
            # Удаляем 20% худших моделей если их доля ниже определенного порога
            threshold = 0.8 * np.max(val_accuracies)  # 80% от лучшей точности
            models_to_keep = []
            weights_to_keep = []
            
            for i, model_idx in enumerate(sorted_indices):
                if val_accuracies[model_idx] >= threshold:
                    models_to_keep.append(self.models[model_idx])
                    weights_to_keep.append(val_accuracies[model_idx])
            
            # Проверяем, что у нас осталось хотя бы 50% моделей
            if len(models_to_keep) >= len(self.models) // 2:
                logger.info(f"Отбор моделей: оставляем {len(models_to_keep)} из {len(self.models)} моделей")
                self.models = models_to_keep
                self.val_accuracies = weights_to_keep
            else:
                logger.info("Отбор моделей: все модели показали примерно одинаковую точность, сохраняем все")

    
    def predict(self, X):
        """
        Выполняет прогнозирование ансамблем моделей с адаптивной стратегией голосования в зависимости от задачи.
        
        Args:
            X: Данные для прогнозирования
            
        Returns:
            Прогнозы меток и вероятности классов
        """
        if not self.models:
            logger.error("Ансамбль не содержит моделей. Сначала выполните build_ensemble().")
            raise ValueError("Ансамбль моделей пуст")
        
        # Проверяем и приводим данные к правильному типу
        X = np.asarray(X, dtype=np.float32)
        
        # Получаем предсказания от всех моделей
        all_predictions = []
        for model in self.models:
            try:
                pred = model.predict(X, verbose=0)
                all_predictions.append(np.array(pred, dtype=np.float32).flatten())
            except Exception as e:
                logger.warning(f"Ошибка при получении предсказания от модели: {str(e)}")
                # В случае ошибки добавляем массив заполненный 0.5 (неуверенное предсказание)
                all_predictions.append(np.full(X.shape[0], 0.5, dtype=np.float32))
        
        # Преобразуем в numpy массив
        all_predictions = np.array(all_predictions)  # [n_models, n_samples]
        
        # Вычисляем уверенность каждой модели для каждого образца как расстояние от 0.5
        confidences = np.abs(all_predictions - 0.5)
        
        # Нормализуем уверенности в веса (по каждому образцу)
        # Обработка случая, когда все уверенности для образца равны 0
        sum_conf = np.sum(confidences, axis=0)
        sum_conf = np.where(sum_conf == 0, 1.0, sum_conf)  # Избегаем деления на ноль
        
        weights = confidences / sum_conf
        
        # Применяем веса к предсказаниям для каждого образца
        ensemble_predictions = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            ensemble_predictions[i] = np.sum(all_predictions[:, i] * weights[:, i])
        
        # Преобразуем в бинарные метки
        binary_predictions = (ensemble_predictions > 0.5).astype(int)
        
        return binary_predictions, ensemble_predictions.reshape(-1, 1)
    
    def build_snapshot_ensemble(self, X_train, y_train, X_val, y_val, n_snapshots=5, max_epochs=200):
        """
        Строит ансамбль на основе снимков одной модели в разных точках обучения.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
            n_snapshots: Количество снимков (моделей в ансамбле)
            max_epochs: Максимальное количество эпох обучения
        """
        logger.info(f"Построение Snapshot Ensemble из {n_snapshots} моделей")
        self.input_shape = X_train.shape[1]
        
        # Проверяем типы данных
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        
        # Создаем директорию для моделей ансамбля
        ensemble_dir = MODELS_DIR / f"monk{self.problem_number}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        
        # Настраиваем базовые слои и параметры
        layer_units = [int(self.base_params.get('units_first', 64)), 
                    int(self.base_params.get('units_first', 64) // 2), 
                    int(self.base_params.get('units_first', 64) // 4)]
        
        # Создаем базовую модель
        model = self._create_model_with_layers(self.base_params, layer_units)
        
        # Компилируем модель
        model.compile(
            optimizer=Adam(learning_rate=float(self.base_params.get('learning_rate', 0.001))),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Цикл обучения с сохранением снимков
        self.models = []
        self.val_accuracies = []
        
        # Расчет эпох для снимков
        epochs_per_snapshot = max_epochs // n_snapshots
        snapshot_epochs = [(i+1) * epochs_per_snapshot for i in range(n_snapshots)]
        
        # Колбэки для обучения
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Обучаем модель на весь период
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=snapshot_epochs[0],  # Обучаем до первого снимка
            batch_size=int(self.base_params.get('batch_size', 32)),
            callbacks=callbacks,
            verbose=0
        )
        
        # Сохраняем первый снимок
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        model_path = ensemble_dir / f"model_1.h5"
        model.save(model_path)
        self.models.append(model)
        self.val_accuracies.append(float(val_acc))
        
        logger.info(f"Snapshot 1/{n_snapshots}, эпоха {snapshot_epochs[0]}, val_acc: {val_acc:.4f}")
        
        # Продолжаем обучение и делаем остальные снимки
        for i in range(1, n_snapshots):
            # Создаем новую модель с другой архитектурой
            new_model = self._create_model_with_layers(self.base_params, layer_units)
            new_model.compile(
                optimizer=Adam(learning_rate=float(self.base_params.get('learning_rate', 0.001)) * (0.8 ** i)),  # Уменьшаем LR
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Обучаем модель
            history = new_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs_per_snapshot,
                batch_size=int(self.base_params.get('batch_size', 32)),
                callbacks=callbacks,
                verbose=0
            )
            
            # Оцениваем и сохраняем
            _, val_acc = new_model.evaluate(X_val, y_val, verbose=0)
            model_path = ensemble_dir / f"model_{i+1}.h5"
            new_model.save(model_path)
            self.models.append(new_model)
            self.val_accuracies.append(float(val_acc))
            
            logger.info(f"Snapshot {i+1}/{n_snapshots}, эпоха {(i+1)*epochs_per_snapshot}, val_acc: {val_acc:.4f}")
        
        logger.info(f"Snapshot Ensemble построен, {len(self.models)} моделей")

    def evaluate(self, X, y_true):
        """
        Оценивает производительность ансамбля.
        
        Args:
            X: Признаки для оценки
            y_true: Истинные метки
            
        Returns:
            Словарь с метриками производительности
        """
        # Проверяем и приводим данные к правильному типу
        X = np.asarray(X, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        
        y_pred, y_prob = self.predict(X)
        
        # Расчет метрик
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Расчет ROC AUC, если данные не полностью сбалансированы
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = np.nan
            fpr, tpr = None, None
            
        # Сохраняем метрики и визуализации
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        # Выводим основные метрики
        logger.info(f"Оценка ансамбля: accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}")
        
        return results
    
    def save(self):
        """Сохраняет весь ансамбль на диск."""
        if not self.models:
            logger.warning("Нет моделей для сохранения")
            return
        
        # Сохраняем каждую модель
        ensemble_dir = MODELS_DIR / f"monk{self.problem_number}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = ensemble_dir / f"model_{i+1}.h5"
            model.save(model_path)
        
        # Сохраняем метаинформацию ансамбля
        meta = {
            'problem_number': int(self.problem_number),
            'num_models': int(self.num_models),
            'base_params': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in self.base_params.items()} if self.base_params else {},
            'input_shape': int(self.input_shape) if self.input_shape is not None else None
        }
        
        with open(ensemble_dir / "ensemble_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Ансамбль из {self.num_models} моделей сохранен в {ensemble_dir}")
    
    @classmethod
    def load(cls, problem_number):
        """
        Загружает ансамбль моделей с диска.
        
        Args:
            problem_number: Номер задачи MONK
            
        Returns:
            Экземпляр EnsembleClassifier с загруженными моделями
        """
        ensemble_dir = MODELS_DIR / f"monk{problem_number}_ensemble"
        
        if not ensemble_dir.exists():
            logger.error(f"Директория ансамбля не найдена: {ensemble_dir}")
            raise FileNotFoundError(f"Директория ансамбля не найдена: {ensemble_dir}")
            
        # Загружаем метаинформацию
        try:
            with open(ensemble_dir / "ensemble_meta.json", 'r') as f:
                meta = json.load(f)
        except:
            logger.error(f"Не удалось загрузить метаданные ансамбля из {ensemble_dir}")
            raise
            
        # Создаем экземпляр класса
        ensemble = cls(
            problem_number=meta['problem_number'],
            base_params=meta.get('base_params', {}),
            num_models=meta['num_models']
        )
        ensemble.input_shape = meta.get('input_shape')
        
        # Загружаем модели
        ensemble.models = []
        for i in range(meta['num_models']):
            model_path = ensemble_dir / f"model_{i+1}.h5"
            if model_path.exists():
                model = load_model(model_path)
                ensemble.models.append(model)
            else:
                logger.warning(f"Файл модели не найден: {model_path}")
        
        logger.info(f"Загружен ансамбль из {len(ensemble.models)} моделей для MONK-{problem_number}")
        
        return ensemble
    
    def _create_model_with_layers(self, params, layer_units):
        """
        Создает модель с указанной конфигурацией слоев и большим разнообразием архитектур.
        
        Args:
            params: Словарь гиперпараметров
            layer_units: Список с количеством нейронов в каждом слое
            
        Returns:
            Модель Keras
        """
        # Определяем регуляризатор
        regularizer = None
        if params.get('use_regularization', False):
            regularizer = l1_l2(l1=float(params.get('l1_factor', 0)), l2=float(params.get('l2_factor', 0)))
        
        # Выбираем тип архитектуры случайным образом для разнообразия
        architecture_type = np.random.choice(['standard', 'wide_narrow', 'varied_activation'])
        
        # Создаем последовательную модель
        model = Sequential()
        
        # Входной слой
        model.add(Input(shape=(self.input_shape,)))
        
        if architecture_type == 'standard':
            # Стандартная архитектура с прямыми соединениями
            for i, units in enumerate(layer_units):
                model.add(Dense(
                    int(units),
                    kernel_regularizer=regularizer
                ))
                
                # Активация
                if params.get('activation') == 'leaky_relu':
                    model.add(LeakyReLU(alpha=0.1))
                else:
                    model.add(Activation(params.get('activation', 'relu')))
                
                # Нормализация и регуляризация
                if params.get('use_batch_norm', True):
                    model.add(BatchNormalization())
                    
                dropout_rate = float(params.get('dropout_rate', 0.3))
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
        
        elif architecture_type == 'wide_narrow':
            # Модель с чередованием широких и узких слоев
            for i, units in enumerate(layer_units):
                # Чередуем увеличение и уменьшение размеров
                if i % 2 == 0:
                    actual_units = int(units * 1.5)  # Широкий слой
                else:
                    actual_units = int(units * 0.75)  # Узкий слой
                    
                model.add(Dense(
                    max(16, actual_units),  # Минимум 16 нейронов
                    kernel_regularizer=regularizer
                ))
                
                # Активация - используем исходную, чтобы избежать проблем
                if params.get('activation') == 'leaky_relu':
                    model.add(LeakyReLU(alpha=0.1))
                else:
                    model.add(Activation(params.get('activation', 'relu')))
                
                # Нормализация и регуляризация
                if params.get('use_batch_norm', True):
                    model.add(BatchNormalization())
                    
                dropout_rate = float(params.get('dropout_rate', 0.3))
                if dropout_rate > 0:
                    # Разные уровни dropout для разных слоев
                    actual_dropout = dropout_rate * (0.8 + 0.4 * (i % 2))
                    model.add(Dropout(min(0.5, actual_dropout)))  # Ограничиваем максимальный dropout
        
        elif architecture_type == 'varied_activation':
            # Модель с разными активациями для каждого слоя
            activations = ['relu', 'elu', 'tanh', 'selu']
            
            for i, units in enumerate(layer_units):
                model.add(Dense(
                    int(units),
                    kernel_regularizer=regularizer
                ))
                
                # Выбираем разные активации для разных слоев
                activation = activations[i % len(activations)]
                if activation == 'leaky_relu':
                    model.add(LeakyReLU(alpha=0.1))
                else:
                    model.add(Activation(activation))
                
                # Нормализация и регуляризация
                if params.get('use_batch_norm', True) and i < len(layer_units) - 1:  # Избегаем BN перед выходным слоем
                    model.add(BatchNormalization())
                    
                dropout_rate = float(params.get('dropout_rate', 0.3))
                if dropout_rate > 0 and i < len(layer_units) - 1:  # Избегаем Dropout перед выходным слоем
                    model.add(Dropout(dropout_rate))
        
        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))
        
        return model
    
    def build_stacked_ensemble(self, X_train, y_train, X_val, y_val, initial_ensemble_size=10, final_ensemble_size=5):
        """
        Строит ансамбль с динамическим отбором лучших моделей по принципу стекинга.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
            initial_ensemble_size: Начальное количество моделей для отбора
            final_ensemble_size: Финальное количество моделей в ансамбле
        """
        logger.info(f"Построение стекингового ансамбля: начинаем с {initial_ensemble_size} моделей, "
                f"отбираем {final_ensemble_size} лучших")
        
        # Сохраняем текущее значение num_models
        original_num_models = self.num_models
        
        # Устанавливаем большее количество для первоначального обучения
        self.num_models = initial_ensemble_size
        
        # Обучаем начальный набор моделей
        self.build_ensemble(X_train, y_train, X_val, y_val)
        
        # Если у нас уже меньше моделей, чем требуется в финальном ансамбле, оставляем как есть
        if len(self.models) <= final_ensemble_size:
            logger.info(f"После обучения осталось {len(self.models)} моделей, меньше или равно требуемым {final_ensemble_size}")
            self.num_models = original_num_models  # Восстанавливаем исходное значение
            return
        
        logger.info(f"Начальный ансамбль из {len(self.models)} моделей обучен, приступаем к отбору")
        
        # Прогнозы для валидационной выборки от каждой модели
        val_predictions = []
        for i, model in enumerate(self.models):
            # Получаем probabilistic predictions, не бинарные
            pred = model.predict(X_val, verbose=0)
            val_predictions.append(pred)
        
        # Преобразуем в numpy массив для удобства
        val_predictions = np.array(val_predictions).squeeze()  # [n_models, n_samples]
        
        # Подготовка для greedy selection
        selected_models = []  # Индексы выбранных моделей
        remaining_models = list(range(len(self.models)))  # Начинаем со всех моделей
        
        # Итеративно выбираем модели, снижающие ошибку ансамбля
        for i in range(final_ensemble_size):
            best_accuracy = 0
            best_model_idx = -1
            
            # Проверяем каждую оставшуюся модель
            for model_idx in remaining_models:
                # Временно добавляем модель к выбранным
                temp_selected = selected_models + [model_idx]
                
                # Вычисляем среднее предсказание для текущего набора моделей
                if len(temp_selected) > 0:
                    temp_ensemble_pred = np.mean(val_predictions[temp_selected], axis=0)
                    temp_binary_pred = (temp_ensemble_pred > 0.5).astype(int)
                    
                    # Вычисляем точность
                    accuracy = accuracy_score(y_val, temp_binary_pred)
                    
                    # Если это лучшая модель на текущий момент, запоминаем её
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_idx = model_idx
            
            # Добавляем лучшую модель к выбранным и удаляем из оставшихся
            if best_model_idx != -1:
                selected_models.append(best_model_idx)
                remaining_models.remove(best_model_idx)
                logger.info(f"Шаг {i+1}: Добавлена модель {best_model_idx} с точностью {best_accuracy:.4f}")
            else:
                logger.warning(f"Не удалось найти модель, улучшающую ансамбль на шаге {i+1}")
                break
        
        # Обновляем ансамбль, оставляя только выбранные модели
        self.models = [self.models[idx] for idx in selected_models]
        
        # Если у нас есть валидационные точности, обновляем и их
        if hasattr(self, 'val_accuracies'):
            self.val_accuracies = [self.val_accuracies[idx] for idx in selected_models]
        
        # Восстанавливаем исходное значение num_models
        self.num_models = original_num_models
        
        logger.info(f"Финальный ансамбль состоит из {len(self.models)} моделей")

def run_optimal_classifier(problem_number, run_hyperopt=True, n_trials=50, ensemble_size=5, 
                          analyze_noise=True, run_comparative=True,
                          min_noise=0.0, max_noise=0.5, noise_step=0.1,
                          ensemble_type='standard'):
    """
    Запускает полный процесс оптимизации, обучения и анализа классификатора.
    
    Args:
        problem_number: Номер задачи MONK (1, 2, или 3)
        run_hyperopt: Выполнять ли оптимизацию гиперпараметров
        n_trials: Количество попыток для оптимизации гиперпараметров
        ensemble_size: Размер ансамбля моделей
        analyze_noise: Выполнять ли анализ устойчивости к шуму
        run_comparative: Выполнять ли сравнительный анализ с другими алгоритмами
        min_noise: Минимальный уровень шума (0.0 - 1.0)
        max_noise: Максимальный уровень шума (0.0 - 1.0)
        noise_step: Шаг изменения уровня шума (0.0 - 1.0)
        
    Returns:
        EnsembleClassifier: Обученный ансамбль моделей
    """
    logger.info(f"=== Запуск оптимального классификатора для MONK-{problem_number} ===")
    start_time = time.time()
    
    # Загрузка и подготовка данных
    X_train, y_train, X_test, y_test = DataLoader.load_monks_data(problem_number)
    logger.info(f"Загружены данные: {X_train.shape[1]} признаков")
    
    # Разделение на обучающую и валидационную выборки
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Убедимся, что данные имеют правильные типы
    X_train_subset = np.asarray(X_train_subset, dtype=np.float32)
    y_train_subset = np.asarray(y_train_subset, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    
    # Оптимизация гиперпараметров или загрузка ранее найденных
    if run_hyperopt:
        logger.info("Запуск оптимизации гиперпараметров")
        best_params = HyperOptimizer.optimize_hyperparameters(
            X_train_subset, y_train_subset, X_val, y_val, 
            problem_number, n_trials=n_trials
        )
    else:
        # Попытка загрузить ранее сохраненные результаты оптимизации
        try:
            result_path = RESULTS_DIR / f"hyperopt_results_monk{problem_number}.pkl"
            if result_path.exists():
                with open(result_path, "rb") as f:
                    study = pickle.load(f)
                best_params = study.best_params
                logger.info(f"Загружены ранее найденные гиперпараметры: {best_params}")
            else:
                logger.warning("Файл с результатами оптимизации не найден, используем значения по умолчанию")
                # Значения по умолчанию для каждой задачи
                if problem_number == 1:
                    # MONK-1: (a1 = a2) OR (a5 = 1)
                    # Относительно простое правило, подходит среднего размера сеть с ELU
                    default_params = {
                        'n_layers': 2,
                        'units_first': 32,
                        'activation': 'tanh',  # Different activation
                        'use_batch_norm': True,
                        'dropout_rate': 0.1,  # Lower dropout
                        'use_residual': False,
                        'learning_rate': 0.003,
                        'batch_size': 16,
                        'optimizer': 'adam',
                        'use_regularization': False,  # No regularization for simpler problem
                        'l1_factor': 0,
                        'l2_factor': 0
                    }
                elif problem_number == 2:
                    # MONK-2: Ровно два атрибута имеют значение 1
                    # Более сложное правило XOR-типа, требует более глубокой сети
                    default_params = {
                        'n_layers': 4,  # Более глубокая сеть для сложной задачи
                        'units_first': 128,  # Больше нейронов
                        'activation': 'selu',  # Self-normalizing функция активации
                        'use_batch_norm': True,
                        'dropout_rate': 0.3,
                        'use_residual': True,  # Остаточные связи помогают с градиентами
                        'learning_rate': 0.0005,  # Меньшая скорость обучения для более стабильного обучения
                        'batch_size': 16,  # Меньший размер батча для лучшего обобщения
                        'optimizer': 'adam',
                        'use_regularization': True,
                        'l1_factor': 1e-5,
                        'l2_factor': 1e-4,
                    }
                else:  # MONK-3
                    # MONK-3: (a5 = 3 AND a4 = 1) OR (a5 ≠ 4 AND a2 ≠ 3)
                    # Комбинация AND и OR с отрицаниями
                    default_params = {
                        'n_layers': 2,
                        'units_first': 48,
                        'activation': 'relu',
                        'use_batch_norm': True,
                        'dropout_rate': 0.15,
                        'use_residual': False,
                        'learning_rate': 0.002,
                        'batch_size': 24,
                        'optimizer': 'adam',
                        'use_regularization': False,
                        'l1_factor': 0,
                        'l2_factor': 0
                    }

                # Присваиваем параметры по умолчанию, если не удалось загрузить
                best_params = default_params

        except Exception as e:
            logger.error(f"Ошибка при загрузке гиперпараметров: {str(e)}")
            # Используем значения по умолчанию
            best_params = {
                'n_layers': 3,
                'units_first': 64,
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3,
                'use_residual': False,
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'use_regularization': False,
                'l1_factor': 0,
                'l2_factor': 0
            }
    
    # Построение и обучение ансамбля моделей
    logger.info(f"Создание ансамбля из {ensemble_size} моделей")
    ensemble = EnsembleClassifier(problem_number, base_params=best_params, num_models=ensemble_size)

    # Выбор метода построения ансамбля в зависимости от типа
    if ensemble_type == 'snapshot':
        ensemble.build_snapshot_ensemble(X_train, y_train, X_val, y_val, n_snapshots=ensemble_size)
    elif ensemble_type == 'stacked':
        # Используем существующий метод build_stacked_ensemble
        ensemble.build_stacked_ensemble(X_train, y_train, X_val, y_val, 
                                    initial_ensemble_size=min(ensemble_size * 2, 20),
                                    final_ensemble_size=ensemble_size)
    else:  # 'standard'
        # Для небольших ансамблей используем обычный подход
        ensemble.build_ensemble(X_train, y_train, X_val, y_val)

    # Если размер ансамбля больше 3, используем стекинговый подход
    if ensemble_size > 3:
        initial_size = min(ensemble_size * 2, 100)  # Обучаем в 2 раза больше моделей для отбора
        ensemble.build_stacked_ensemble(X_train, y_train, X_val, y_val, 
                                    initial_ensemble_size=initial_size, 
                                    final_ensemble_size=ensemble_size)
    else:
        # Для небольших ансамблей используем обычный подход
        ensemble.build_ensemble(X_train, y_train, X_val, y_val)
    
    # Оценка на тестовой выборке
    logger.info("Оценка ансамбля на тестовой выборке")
    results = ensemble.evaluate(X_test, y_test)
    
    # Визуализация результатов
    plt.figure(figsize=(12, 10))
    
    # Матрица ошибок
    plt.subplot(2, 2, 1)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Матрица ошибок")
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    
    # ROC-кривая (если применимо)
    if not np.isnan(results.get('roc_auc', np.nan)) and results.get('fpr') is not None and results.get('tpr') is not None:
        plt.subplot(2, 2, 2)
        plt.plot(results['fpr'], results['tpr'], label=f'ROC curve (AUC = {results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
    
    # Сохраняем визуализацию
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"results_monk{problem_number}.png", dpi=300)
    plt.close()
    
    # Сохраняем ансамбль моделей
    ensemble.save()
    
    # Сравнительный анализ с другими алгоритмами
    if run_comparative:
        logger.info("Сравнительный анализ с другими алгоритмами")
        comparative = ComparativeAnalyzer(ensemble, problem_number)
        comparative.train_baseline_models(X_train, y_train)
        
        # Создаем массив уровней шума на основе параметров
        noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()
        
        # Проводим сравнительный анализ
        comparative_results = comparative.analyze_noise_resistance(
            X_test, y_test,
            noise_types=['gaussian', 'uniform', 'impulse', 'missing'],
            noise_levels=noise_levels,
            n_experiments=3
        )
    
    # 5. Анализ устойчивости к шуму отдельной модели
    if analyze_noise and not run_comparative:
        logger.info("Анализ устойчивости к шуму")
        
        # Создаем класс для анализа шума
        class NoiseAnalyzer:
            @staticmethod
            def add_noise(X, noise_level, noise_type='gaussian'):
                logger.info(f"Добавление {noise_type} шума с уровнем {noise_level:.2f}")
                
                if noise_level <= 0:
                    return X
                    
                X_noisy = X.copy()
                n_samples, n_features = X.shape
                
                # Количество примеров для добавления шума (согласно уровню шума)
                n_noisy_samples = int(noise_level * n_samples)
                noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
                
                # Статистики признаков для реалистичного шума
                feature_means = np.mean(X, axis=0)
                feature_stds = np.std(X, axis=0)
                feature_mins = np.min(X, axis=0)
                feature_maxs = np.max(X, axis=0)
                
                # Для каждого зашумляемого примера
                for idx in noisy_indices:
                    # Выбираем случайное количество признаков для зашумления
                    n_features_to_noise = np.random.randint(1, n_features + 1)
                    features_to_noise = np.random.choice(n_features, n_features_to_noise, replace=False)
                    
                    # Добавляем шум к выбранным признакам
                    for feature_idx in features_to_noise:
                        feature_range = feature_maxs[feature_idx] - feature_mins[feature_idx]
                        
                        if noise_type == 'gaussian':
                            # Гауссовский шум, масштабируемый к стандартному отклонению признака
                            noise = np.random.normal(0, feature_stds[feature_idx])
                            X_noisy[idx, feature_idx] += noise
                            
                        elif noise_type == 'uniform':
                            # Равномерный шум в пределах диапазона признака
                            noise = np.random.uniform(-0.5, 0.5) * feature_range
                            X_noisy[idx, feature_idx] += noise
                            
                        elif noise_type == 'impulse':
                            # Импульсный шум - замена на экстремальные значения
                            impulse_type = np.random.choice(['min', 'max', 'extreme'])
                            if impulse_type == 'min':
                                X_noisy[idx, feature_idx] = feature_mins[feature_idx]
                            elif impulse_type == 'max':
                                X_noisy[idx, feature_idx] = feature_maxs[feature_idx]
                            else:  # extreme
                                extreme_factor = np.random.choice([-2, 2])
                                X_noisy[idx, feature_idx] = feature_means[feature_idx] + extreme_factor * feature_stds[feature_idx]
                                
                        elif noise_type == 'missing':
                            # Замена на NaN (требует предобработки перед использованием)
                            X_noisy[idx, feature_idx] = np.nan
                
                # Если есть пропущенные значения, заполняем их средними (импутация)
                if noise_type == 'missing':
                    for j in range(n_features):
                        mask = np.isnan(X_noisy[:, j])
                        X_noisy[mask, j] = np.mean(X_noisy[~mask, j])
                
                return X_noisy
            
            @staticmethod
            def analyze_noise_resistance(ensemble, X, y, noise_types=None, noise_levels=None, n_experiments=5):
                if noise_types is None:
                    noise_types = ['gaussian', 'uniform', 'impulse']
                    
                if noise_levels is None:
                    noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()
                    
                logger.info(f"Анализ устойчивости к шуму: {len(noise_types)} типов, {len(noise_levels)} уровней, {n_experiments} экспериментов")
                
                # Получаем базовую точность (без шума)
                baseline_preds, _ = ensemble.predict(X)
                baseline_accuracy = accuracy_score(y, baseline_preds)
                logger.info(f"Базовая точность (без шума): {baseline_accuracy:.4f}")
                
                # Структура для хранения результатов
                results = {
                    'noise_types': noise_types,
                    'noise_levels': noise_levels,
                    'baseline_accuracy': baseline_accuracy,
                    'accuracies': {}
                }
                
                # Для каждого типа шума
                for noise_type in noise_types:
                    results['accuracies'][noise_type] = []
                    
                    # Для каждого уровня шума
                    for noise_level in noise_levels:
                        level_accuracies = []
                        
                        # Повторяем эксперимент несколько раз
                        for exp in range(n_experiments):
                            # Добавляем шум
                            X_noisy = NoiseAnalyzer.add_noise(X, noise_level, noise_type)
                            
                            # Делаем предсказание
                            preds, _ = ensemble.predict(X_noisy)
                            
                            # Вычисляем точность
                            accuracy = accuracy_score(y, preds)
                            level_accuracies.append(accuracy)
                        
                        # Сохраняем средний результат
                        mean_accuracy = np.mean(level_accuracies)
                        std_accuracy = np.std(level_accuracies)
                        
                        results['accuracies'][noise_type].append({
                            'level': float(noise_level),
                            'mean_accuracy': float(mean_accuracy),
                            'std_accuracy': float(std_accuracy),
                            'experiments': [float(acc) for acc in level_accuracies]
                        })
                        
                        logger.info(f"Шум '{noise_type}' уровня {noise_level:.1f}: точность = {mean_accuracy:.4f} (±{std_accuracy:.4f})")
                
                # Визуализация результатов
                NoiseAnalyzer.visualize_noise_resistance(results, ensemble.problem_number)
                
                # Сохраняем результаты
                with open(RESULTS_DIR / f"noise_analysis_monk{ensemble.problem_number}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                    
                return results
            
            @staticmethod
            def visualize_noise_resistance(results, problem_number):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Строим график для каждого типа шума
                for noise_type in results['noise_types']:
                    noise_data = results['accuracies'][noise_type]
                    levels = [data['level'] for data in noise_data]
                    accuracies = [data['mean_accuracy'] for data in noise_data]
                    errors = [data['std_accuracy'] for data in noise_data]
                    
                    ax.errorbar(
                        levels, 
                        accuracies, 
                        yerr=errors, 
                        marker='o', 
                        linestyle='-', 
                        capsize=5, 
                        label=f'{noise_type.capitalize()} Noise'
                    )
                
                # Добавляем базовую линию (без шума)
                ax.axhline(
                    y=results['baseline_accuracy'], 
                    color='black', 
                    linestyle='--', 
                    label=f'Baseline (No Noise): {results["baseline_accuracy"]:.4f}'
                )
                
                # Настраиваем график
                ax.set_title(f'Устойчивость к шуму для MONK-{problem_number}', fontsize=16)
                ax.set_xlabel('Уровень шума', fontsize=14)
                ax.set_ylabel('Точность', fontsize=14)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(-0.02, max(results['noise_levels']) + 0.02)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12)
                
                # Сохраняем график
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"noise_resistance_monk{problem_number}.png", dpi=300)
                plt.close()
        
        # Создаем массив уровней шума на основе параметров
        noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()

        # Анализируем устойчивость к шуму
        noise_analysis = NoiseAnalyzer.analyze_noise_resistance(
            ensemble, X_test, y_test,
            noise_types=['gaussian', 'uniform', 'impulse', 'missing'],  # Избегаем 'missing', требующего специальной обработки
            noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
            n_experiments=3
        )
    
    # Выводим сводную информацию
    elapsed_time = time.time() - start_time
    logger.info(f"=== Результаты для MONK-{problem_number} ===")
    logger.info(f"Точность: {results['accuracy']:.4f}")
    logger.info(f"ROC AUC: {results.get('roc_auc', 'N/A')}")
    
    # Получаем детальную информацию из отчета
    if 'classification_report' in results:
        for cls, metrics in results['classification_report'].items():
            if cls in ['0', '1']:
                logger.info(f"Класс {cls}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")
    
    logger.info(f"Время выполнения: {elapsed_time:.2f} секунд")
    
    return ensemble

class ComparativeAnalyzer:
    """Класс для сравнительного анализа различных алгоритмов в условиях шума."""
    
    def __init__(self, ensemble_model, problem_number):
        """
        Инициализация анализатора.
        
        Args:
            ensemble_model: Обученный ансамбль моделей
            problem_number: Номер задачи MONK
        """
        self.ensemble_model = ensemble_model
        self.problem_number = problem_number
        self.models = {}
        self.model_names = []
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Добавляем ансамбль в словарь моделей
        self.models["Ensemble NN"] = ensemble_model
        self.model_names.append("Ensemble NN")
        
        # Логирование
        logger.info(f"Инициализирован ComparativeAnalyzer для MONK-{problem_number}")
    
    def train_baseline_models(self, X_train, y_train):
        """
        Обучает базовые модели для сравнения.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
        """
        logger.info("Обучение базовых моделей для сравнения")
        
        # Приводим данные к нужному типу
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        
        # Настройки для разных моделей MONK
        if self.problem_number == 2:
            # Более сложная задача MONK-2
            rf_estimators = 200
            gb_estimators = 200
            mlp_hidden_layer_sizes = (100, 50, 25)
        else:
            # Более простые задачи MONK-1 и MONK-3
            rf_estimators = 100
            gb_estimators = 100
            mlp_hidden_layer_sizes = (64, 32)
        
        # Одиночная нейронная сеть
        logger.info("Обучение одиночной нейронной сети")
        single_nn = MLPClassifier(
            hidden_layer_sizes=mlp_hidden_layer_sizes, 
            activation='relu', 
            solver='adam', 
            alpha=0.0001,
            batch_size='auto', 
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        single_nn.fit(X_train, y_train)
        self.models["Single NN"] = single_nn
        self.model_names.append("Single NN")
        
        # Random Forest
        logger.info("Обучение Random Forest")
        rf = RandomForestClassifier(
            n_estimators=rf_estimators, 
            max_depth=None, 
            min_samples_split=2,
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models["Random Forest"] = rf
        self.model_names.append("Random Forest")
        
        # Support Vector Machine
        logger.info("Обучение SVM")
        svm = SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            probability=True,
            random_state=42
        )
        svm.fit(X_train, y_train)
        self.models["SVM"] = svm
        self.model_names.append("SVM")
        
        # Gradient Boosting
        logger.info("Обучение Gradient Boosting")
        gb = GradientBoostingClassifier(
            n_estimators=gb_estimators, 
            learning_rate=0.1, 
            max_depth=3,
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.models["Gradient Boosting"] = gb
        self.model_names.append("Gradient Boosting")
        
        # k-Nearest Neighbors
        logger.info("Обучение KNN")
        knn = KNeighborsClassifier(
            n_neighbors=5, 
            weights='distance', 
            n_jobs=-1
        )
        knn.fit(X_train, y_train)
        self.models["KNN"] = knn
        self.model_names.append("KNN")
        
        # Логистическая регрессия
        logger.info("Обучение Logistic Regression")
        lr = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42, 
            n_jobs=-1
        )
        lr.fit(X_train, y_train)
        self.models["Logistic Regression"] = lr
        self.model_names.append("Logistic Regression")
        
        logger.info(f"Обучено {len(self.models)} моделей для сравнения")
    
    def predict_with_model(self, model, X):
        """
        Делает предсказания моделью, обрабатывая особые случаи.
        
        Args:
            model: Модель для прогнозирования
            X: Данные для прогнозирования
            
        Returns:
            Предсказанные метки
        """
        X = np.asarray(X, dtype=np.float32)
        
        # Особый случай для ансамбля
        if model == self.ensemble_model:
            y_pred, _ = model.predict(X)
            return y_pred
        
        # Для SVM и логистической регрессии
        if isinstance(model, (SVC, LogisticRegression)):
            return model.predict(X)
        
        # Для остальных моделей
        return model.predict(X)
    
    def analyze_noise_resistance(self, X_test, y_test, noise_types=None, noise_levels=None, n_experiments=3):
        """
        Анализирует устойчивость различных моделей к шуму.
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовые метки
            noise_types: Список типов шума для тестирования
            noise_levels: Список уровней шума для тестирования
            n_experiments: Количество экспериментов для каждой комбинации
            
        Returns:
            Словарь с результатами анализа
        """
        if noise_types is None:
            noise_types = ['gaussian', 'uniform', 'impulse', 'missing']
            
        if noise_levels is None:
            noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        logger.info(f"Сравнительный анализ устойчивости к шуму: {len(self.models)} моделей, " 
                  f"{len(noise_types)} типов шума, {len(noise_levels)} уровней, {n_experiments} экспериментов")
        
        # Проверяем, что у нас есть модели для сравнения
        if len(self.models) <= 1:
            logger.warning("Нет базовых моделей для сравнения. Сначала выполните train_baseline_models().")
            return None
        
        # Подготавливаем структуру для результатов
        results = {
            'noise_types': noise_types,
            'noise_levels': noise_levels,
            'model_names': self.model_names,
            'accuracies': {}
        }
        
        # Для каждого типа шума
        for noise_type in noise_types:
            results['accuracies'][noise_type] = {}
            
            # Для каждой модели
            for model_name in self.model_names:
                results['accuracies'][noise_type][model_name] = []
                
                # Для каждого уровня шума
                for noise_level in noise_levels:
                    level_accuracies = []
                    
                    # Повторяем эксперимент несколько раз
                    for exp in range(n_experiments):
                        try:
                            # Добавляем шум к тестовым данным
                            X_noisy = self.add_noise(X_test, noise_level, noise_type)
                            
                            # Получаем предсказания
                            model = self.models[model_name]
                            y_pred = self.predict_with_model(model, X_noisy)
                            
                            # Рассчитываем точность
                            accuracy = accuracy_score(y_test, y_pred)
                            level_accuracies.append(accuracy)
                        except Exception as e:
                            logger.error(f"Ошибка при тестировании {model_name} с шумом {noise_type} {noise_level}: {str(e)}")
                            level_accuracies.append(np.nan)
                    
                    # Сохраняем средний результат
                    mean_accuracy = np.nanmean(level_accuracies)
                    std_accuracy = np.nanstd(level_accuracies)
                    
                    results['accuracies'][noise_type][model_name].append({
                        'level': float(noise_level),
                        'mean_accuracy': float(mean_accuracy),
                        'std_accuracy': float(std_accuracy),
                        'experiments': [float(acc) for acc in level_accuracies]
                    })
                    
                    logger.info(f"Модель {model_name}, шум '{noise_type}' уровня {noise_level:.1f}: "
                             f"точность = {mean_accuracy:.4f} (±{std_accuracy:.4f})")
        
        # Сохраняем результаты в JSON
        import json
        with open(self.results_dir / f"comparative_analysis_monk{self.problem_number}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Визуализируем и сохраняем результаты
        self.visualize_comparative_results(results)
        
        return results
    
    def add_noise(self, X, noise_level, noise_type='gaussian'):
        """
        Добавляет шум к признакам.
        
        Args:
            X: Входные признаки
            noise_level: Уровень шума (0-1)
            noise_type: Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
            
        Returns:
            Данные с добавленным шумом
        """
        if noise_level <= 0:
            return X
                
        X_noisy = X.copy()
        n_samples, n_features = X.shape
        
        # Количество примеров для добавления шума (согласно уровню шума)
        n_noisy_samples = int(noise_level * n_samples)
        noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
        
        # Статистики признаков для реалистичного шума
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        feature_mins = np.min(X, axis=0)
        feature_maxs = np.max(X, axis=0)
        
        # Для каждого зашумляемого примера
        for idx in noisy_indices:
            # Выбираем случайное количество признаков для зашумления
            n_features_to_noise = np.random.randint(1, n_features + 1)
            features_to_noise = np.random.choice(n_features, n_features_to_noise, replace=False)
            
            # Добавляем шум к выбранным признакам
            for feature_idx in features_to_noise:
                feature_range = feature_maxs[feature_idx] - feature_mins[feature_idx]
                
                if noise_type == 'gaussian':
                    # Гауссовский шум, масштабируемый к стандартному отклонению признака
                    noise = np.random.normal(0, feature_stds[feature_idx])
                    X_noisy[idx, feature_idx] += noise
                    
                elif noise_type == 'uniform':
                    # Равномерный шум в пределах диапазона признака
                    noise = np.random.uniform(-0.5, 0.5) * feature_range
                    X_noisy[idx, feature_idx] += noise
                    
                elif noise_type == 'impulse':
                    # Импульсный шум - замена на экстремальные значения
                    impulse_type = np.random.choice(['min', 'max', 'extreme'])
                    if impulse_type == 'min':
                        X_noisy[idx, feature_idx] = feature_mins[feature_idx]
                    elif impulse_type == 'max':
                        X_noisy[idx, feature_idx] = feature_maxs[feature_idx]
                    else:  # extreme
                        extreme_factor = np.random.choice([-2, 2])
                        X_noisy[idx, feature_idx] = feature_means[feature_idx] + extreme_factor * feature_stds[feature_idx]
                        
                elif noise_type == 'missing':
                    # Замена на NaN (требует предобработки перед использованием)
                    X_noisy[idx, feature_idx] = np.nan
        
        # Если есть пропущенные значения, заполняем их средними
        if noise_type == 'missing':
            for j in range(n_features):
                mask = np.isnan(X_noisy[:, j])
                X_noisy[mask, j] = np.mean(X_noisy[~mask, j])
        
        return X_noisy
    
    def visualize_comparative_results(self, results):
        """
        Визуализирует результаты сравнительного анализа.
        
        Args:
            results: Словарь с результатами анализа
        """
        noise_types = results['noise_types']
        noise_levels = results['noise_levels']
        model_names = results['model_names']
        
        # Создаем отдельный график для каждого типа шума
        for noise_type in noise_types:
            plt.figure(figsize=(12, 8))
            
            # Цветовая схема для моделей
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
            
            # Для каждой модели
            for i, model_name in enumerate(model_names):
                # Извлекаем данные
                model_data = results['accuracies'][noise_type][model_name]
                levels = [data['level'] for data in model_data]
                accuracies = [data['mean_accuracy'] for data in model_data]
                errors = [data['std_accuracy'] for data in model_data]
                
                # Строим линию с ошибками
                plt.errorbar(
                    levels, 
                    accuracies, 
                    yerr=errors, 
                    marker='o' if model_name == "Ensemble NN" else 's',
                    linestyle='-' if model_name == "Ensemble NN" else '--',
                    linewidth=2 if model_name == "Ensemble NN" else 1,
                    markersize=8 if model_name == "Ensemble NN" else 6,
                    capsize=5,
                    color=colors[i],
                    label=model_name
                )
            
            # Настраиваем график
            plt.title(f'Сравнительный анализ устойчивости к шуму {noise_type.capitalize()} для MONK-{self.problem_number}', fontsize=16)
            plt.xlabel('Уровень шума', fontsize=14)
            plt.ylabel('Точность', fontsize=14)
            plt.ylim(0, 1.05)
            plt.xlim(-0.02, max(noise_levels) + 0.02)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            # Аннотации для наилучшей и наихудшей моделей на каждом уровне шума
            for i, level in enumerate(noise_levels):
                if level == 0:
                    continue  # Пропускаем базовый уровень
                
                # Находим лучшую и худшую модель для этого уровня
                level_accuracies = {}
                for model_name in model_names:
                    level_accuracies[model_name] = results['accuracies'][noise_type][model_name][i]['mean_accuracy']
                
                best_model = max(level_accuracies, key=level_accuracies.get)
                worst_model = min(level_accuracies, key=level_accuracies.get)
                
                # Аннотация для лучшей модели
                if i == len(noise_levels) - 1:  # Только для последнего уровня
                    plt.annotate(
                        f"Лучшая: {best_model}",
                        xy=(level, level_accuracies[best_model]),
                        xytext=(level, level_accuracies[best_model] + 0.05),
                        fontsize=10,
                        arrowprops=dict(arrowstyle="->", color='green'),
                        color='green'
                    )
            
            # Сохраняем график
            plt.tight_layout()
            plt.savefig(self.results_dir / f"comparative_{noise_type}_monk{self.problem_number}.png", dpi=300)
            plt.close()
        
        # Сводный график по всем типам шума для ансамбля
        plt.figure(figsize=(12, 8))
        
        # Цветовая схема для типов шума
        colors = plt.cm.tab10(np.linspace(0, 1, len(noise_types)))
        
        # Для каждого типа шума
        for i, noise_type in enumerate(noise_types):
            # Извлекаем данные для ансамбля
            model_data = results['accuracies'][noise_type]["Ensemble NN"]
            levels = [data['level'] for data in model_data]
            accuracies = [data['mean_accuracy'] for data in model_data]
            errors = [data['std_accuracy'] for data in model_data]
            
            # Строим линию с ошибками
            plt.errorbar(
                levels, 
                accuracies, 
                yerr=errors, 
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=8,
                capsize=5,
                color=colors[i],
                label=f'{noise_type.capitalize()} Noise'
            )
        
        # Настраиваем график
        plt.title(f'Устойчивость ансамбля к различным типам шума для MONK-{self.problem_number}', fontsize=16)
        plt.xlabel('Уровень шума', fontsize=14)
        plt.ylabel('Точность', fontsize=14)
        plt.ylim(0, 1.05)
        plt.xlim(-0.02, max(noise_levels) + 0.02)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Сохраняем график
        plt.tight_layout()
        plt.savefig(self.results_dir / f"ensemble_noise_comparison_monk{self.problem_number}.png", dpi=300)
        plt.close()

def analyze_ensemble_size(problem_number, max_ensemble_size=15, noise_levels=None):
    """
    Запускает анализ зависимости точности от размера ансамбля.
    
    Args:
        problem_number: Номер задачи MONK
        max_ensemble_size: Максимальный размер ансамбля для исследования
        noise_levels: Список уровней шума
    
    Returns:
        Результаты анализа
    """
    logger.info(f"=== Запуск анализа размера ансамбля для MONK-{problem_number} ===")
    start_time = time.time()
    
    # Загрузка данных
    X_train, y_train, X_test, y_test = DataLoader.load_monks_data(problem_number)
    
    # Разделение на тренировочную и валидационную выборки
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Установка базовых гиперпараметров на основе задачи
    if problem_number == 2:  # Сложная задача
        base_params = {
            'n_layers': 3,
            'units_first': 128,
            'activation': 'elu',
            'use_batch_norm': True,
            'dropout_rate': 0.3,
            'use_residual': True,
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'use_regularization': True,
            'l1_factor': 1e-5,
            'l2_factor': 1e-4,
        }
    else:  # Более простые задачи
        base_params = {
            'n_layers': 2,
            'units_first': 64,
            'activation': 'relu',
            'use_batch_norm': True,
            'dropout_rate': 0.2,
            'use_residual': False,
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'use_regularization': False,
            'l1_factor': 0,
            'l2_factor': 0
        }
    
    # Результаты для разных размеров ансамбля
    results = {
        'ensemble_sizes': list(range(1, max_ensemble_size + 1)),
        'problem_number': problem_number,
        'accuracies': {}
    }
    
    # Уровни шума по умолчанию
    if noise_levels is None:
        noise_levels = [0, 0.1, 0.3, 0.5]
    
    results['noise_levels'] = noise_levels
    
    # Для каждого уровня шума
    for noise_level in noise_levels:
        noise_key = f"noise_{int(noise_level*100)}"
        results['accuracies'][noise_key] = []
        
        # Для каждого размера ансамбля
        for size in range(1, max_ensemble_size + 1):
            # Создаем и обучаем ансамбль
            ensemble = EnsembleClassifier(problem_number, base_params=base_params, num_models=size)
            ensemble.build_ensemble(X_train, y_train, X_val, y_val)
            
            # Если нужно добавить шум
            if noise_level > 0:
                X_test_noisy = add_noise_to_features(X_test, noise_level, 'gaussian')
                predictions, _ = ensemble.predict(X_test_noisy)
            else:
                predictions, _ = ensemble.predict(X_test)
                
            # Вычисляем точность
            accuracy = accuracy_score(y_test, predictions)
            
            # Сохраняем результат
            results['accuracies'][noise_key].append({
                'size': size,
                'accuracy': float(accuracy)
            })
            
            logger.info(f"Ансамбль размера {size}, шум {noise_level*100}%: точность = {accuracy:.4f}")
    
    # Визуализируем результаты
    try:
        # Для случая без шума
        plt.figure(figsize=(12, 8))
        noise_key = "noise_0"
        sizes = [item['size'] for item in results['accuracies'][noise_key]]
        accuracies = [item['accuracy'] for item in results['accuracies'][noise_key]]
        
        plt.plot(sizes, accuracies, 'o-', linewidth=2)
        
        plt.title(f'Зависимость точности от размера ансамбля (MONK-{problem_number})', fontsize=14)
        plt.xlabel('Размер ансамбля', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.grid(alpha=0.3)
        plt.xticks(sizes)
        
        plt.savefig(RESULTS_DIR / f"ensemble_size_impact_monk{problem_number}.png", dpi=300)
        plt.close()
        
        # Для разных уровней шума
        plt.figure(figsize=(12, 8))
        
        for noise_level in noise_levels:
            noise_key = f"noise_{int(noise_level*100)}"
            if noise_key in results['accuracies']:
                sizes = [item['size'] for item in results['accuracies'][noise_key]]
                accuracies = [item['accuracy'] for item in results['accuracies'][noise_key]]
                
                plt.plot(sizes, accuracies, 'o-', label=f'Шум {int(noise_level*100)}%')
        
        plt.title(f'Влияние размера ансамбля на устойчивость к шуму (MONK-{problem_number})', fontsize=14)
        plt.xlabel('Размер ансамбля', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.grid(alpha=0.3)
        plt.xticks(sizes)
        plt.legend()
        
        plt.savefig(RESULTS_DIR / f"ensemble_size_noise_impact_monk{problem_number}.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Ошибка при визуализации результатов: {str(e)}")
    
    # Сохраняем результаты в JSON
    try:
        with open(RESULTS_DIR / f"ensemble_size_analysis_monk{problem_number}.json", 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.warning(f"Ошибка при сохранении результатов: {str(e)}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"=== Анализ размера ансамбля завершен. Время выполнения: {elapsed_time:.2f} секунд ===")
    
    return results

def add_noise_to_features(X, noise_level, noise_type='gaussian'):
    """Добавляет шум к признакам."""
    if noise_level <= 0:
        return X
            
    X_noisy = X.copy()
    n_samples, n_features = X.shape
    
    # Количество примеров для добавления шума (согласно уровню шума)
    n_noisy_samples = int(noise_level * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
    
    # Статистики признаков для реалистичного шума
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    feature_mins = np.min(X, axis=0)
    feature_maxs = np.max(X, axis=0)
    
    # Для каждого зашумляемого примера
    for idx in noisy_indices:
        # Выбираем случайное количество признаков для зашумления
        n_features_to_noise = np.random.randint(1, n_features + 1)
        features_to_noise = np.random.choice(n_features, n_features_to_noise, replace=False)
        
        # Добавляем шум к выбранным признакам
        for feature_idx in features_to_noise:
            feature_range = feature_maxs[feature_idx] - feature_mins[feature_idx]
            
            if noise_type == 'gaussian':
                # Гауссовский шум
                noise = np.random.normal(0, feature_stds[feature_idx])
                X_noisy[idx, feature_idx] += noise
                
            elif noise_type == 'uniform':
                # Равномерный шум
                noise = np.random.uniform(-0.5, 0.5) * feature_range
                X_noisy[idx, feature_idx] += noise
                
            elif noise_type == 'impulse':
                # Импульсный шум
                impulse_type = np.random.choice(['min', 'max', 'extreme'])
                if impulse_type == 'min':
                    X_noisy[idx, feature_idx] = feature_mins[feature_idx]
                elif impulse_type == 'max':
                    X_noisy[idx, feature_idx] = feature_maxs[feature_idx]
                else:
                    extreme_factor = np.random.choice([-2, 2])
                    X_noisy[idx, feature_idx] = feature_means[feature_idx] + extreme_factor * feature_stds[feature_idx]
                    
            elif noise_type == 'missing':
                # Пропущенные значения
                X_noisy[idx, feature_idx] = np.nan
    
    # Заполняем пропущенные значения
    if noise_type == 'missing':
        for j in range(n_features):
            mask = np.isnan(X_noisy[:, j])
            if np.any(mask):
                X_noisy[mask, j] = np.mean(X_noisy[~mask, j])
    
    return X_noisy

def run_super_ensemble(problem_number, nn_ensemble_size=5, run_hyperopt=False, n_trials=50,
                    analyze_noise=True, run_comparative=True,
                    min_noise=0.0, max_noise=0.5, noise_step=0.1):
    """
    Запускает расширенный суперансамбль для задачи MONK с анализом шума.
    
    Args:
        problem_number: Номер задачи MONK (1, 2, или 3)
        nn_ensemble_size: Размер ансамбля нейронных сетей
        run_hyperopt: Выполнять ли оптимизацию гиперпараметров
        n_trials: Количество попыток для оптимизации гиперпараметров
        analyze_noise: Выполнять ли анализ устойчивости к шуму
        run_comparative: Выполнять ли сравнительный анализ с другими алгоритмами
        min_noise: Минимальный уровень шума (0.0 - 1.0)
        max_noise: Максимальный уровень шума (0.0 - 1.0)
        noise_step: Шаг изменения уровня шума (0.0 - 1.0)
        
    Returns:
        SuperEnsemble: Обученный суперансамбль
    """
    from super_ensemble import SuperEnsemble
    
    logger.info(f"=== Запуск SuperEnsemble для MONK-{problem_number} ===")
    start_time = time.time()
    
    # Загрузка и подготовка данных
    X_train, y_train, X_test, y_test = DataLoader.load_monks_data(problem_number)
    logger.info(f"Загружены данные: {X_train.shape[1]} признаков")
    
    # Разделение на обучающую и валидационную выборки
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Оптимизация гиперпараметров или загрузка ранее найденных
    if run_hyperopt:
        logger.info("Запуск оптимизации гиперпараметров")
        best_params = HyperOptimizer.optimize_hyperparameters(
            X_train_subset, y_train_subset, X_val, y_val, 
            problem_number, n_trials=n_trials
        )
    else:
        # Попытка загрузить ранее сохраненные результаты оптимизации
        try:
            result_path = RESULTS_DIR / f"hyperopt_results_monk{problem_number}.pkl"
            if result_path.exists():
                with open(result_path, "rb") as f:
                    study = pickle.load(f)
                best_params = study.best_params
                logger.info(f"Загружены ранее найденные гиперпараметры: {best_params}")
            else:
                logger.warning("Файл с результатами оптимизации не найден, используем значения по умолчанию")
                # Значения по умолчанию
                best_params = {
                    'n_layers': 3,
                    'units_first': 64,
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'dropout_rate': 0.3,
                    'use_residual': False,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'optimizer': 'adam',
                    'use_regularization': False,
                    'l1_factor': 0,
                    'l2_factor': 0
                }
        except Exception as e:
            logger.error(f"Ошибка при загрузке гиперпараметров: {str(e)}")
            best_params = {
                'n_layers': 3,
                'units_first': 64,
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3,
                'use_residual': False,
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'use_regularization': False,
                'l1_factor': 0,
                'l2_factor': 0
            }
    
    # Создание и обучение SuperEnsemble
    super_ensemble = SuperEnsemble(
        problem_number=problem_number,
        base_params=best_params,
        nn_ensemble_size=nn_ensemble_size
    )
    
    # Обучаем на всей обучающей выборке
    super_ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Оценка на тестовой выборке
    logger.info("Оценка SuperEnsemble на тестовой выборке")
    results = super_ensemble.evaluate(X_test, y_test)
    
    # Визуализация результатов суперансамбля
    plt.figure(figsize=(20, 15))

    # 1. Матрица ошибок (верхний левый угол)
    plt.subplot(2, 3, 1)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Матрица ошибок SuperEnsemble", fontsize=14)
    plt.ylabel("Истинный класс", fontsize=12)
    plt.xlabel("Предсказанный класс", fontsize=12)

    # 2. ROC-кривая (верхний средний)
    if not np.isnan(results.get('roc_auc', np.nan)) and results.get('fpr') is not None and results.get('tpr') is not None:
        plt.subplot(2, 3, 2)
        plt.plot(results['fpr'], results['tpr'], 'b-', 
                label=f'SuperEnsemble ROC (AUC = {results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve SuperEnsemble', fontsize=14)
        plt.legend(loc="lower right")

    # 3. Важность компонентов ансамбля (верхний правый)
    plt.subplot(2, 3, 3)
    model_weights = []
    model_names = []

    # Собираем веса моделей если доступны
    if hasattr(super_ensemble, 'optimal_weights') and super_ensemble.optimal_weights is not None:
        weights = super_ensemble.optimal_weights
        
        # Получаем имена и веса всех компонентов
        nn_count = len(super_ensemble.nn_ensemble.models)
        for i in range(nn_count):
            model_names.append(f"NN_{i+1}")
            model_weights.append(weights[i] if i < len(weights) else 0)
        
        # Добавляем другие модели
        other_idx = nn_count
        for name in super_ensemble.additional_models.keys():
            if other_idx < len(weights):
                model_names.append(name)
                model_weights.append(weights[other_idx])
                other_idx += 1

        # Сортируем по важности
        sorted_idx = np.argsort(model_weights)
        sorted_names = [model_names[i] for i in sorted_idx[-10:]]  # Берем топ-10
        sorted_weights = [model_weights[i] for i in sorted_idx[-10:]]
        
        # Визуализируем
        bars = plt.barh(sorted_names, sorted_weights, color='skyblue')
        plt.title('Важность компонентов ансамбля', fontsize=14)
        plt.xlabel('Относительная важность', fontsize=12)
        plt.gca().invert_yaxis()  # Наибольшая важность сверху

    # 4. Детализированный отчет (нижний левый)
    plt.subplot(2, 3, 4)
    report_str = "Детальный отчет по классам:\n\n"
    for cls, metrics in results['classification_report'].items():
        if cls in ['0', '1', 0, 1]:
            report_str += f"Класс {cls}:\n"
            report_str += f"  Precision: {metrics['precision']:.4f}\n"
            report_str += f"  Recall: {metrics['recall']:.4f}\n"
            report_str += f"  F1-Score: {metrics['f1-score']:.4f}\n"
            report_str += f"  Support: {metrics['support']}\n\n"
    report_str += f"Accuracy: {results['accuracy']:.4f}\n"
    report_str += f"Macro Avg F1: {results['classification_report']['macro avg']['f1-score']:.4f}\n"
    report_str += f"Weighted Avg F1: {results['classification_report']['weighted avg']['f1-score']:.4f}\n"
    plt.text(0.1, 0.1, report_str, fontsize=12, va='top', ha='left')
    plt.axis('off')

    # 5. Сравнение компонентов по точности (нижний средний)
    plt.subplot(2, 3, 5)
    # Здесь добавим сравнение компонентов по точности
    # Сначала нужно вычислить точность каждого компонента на тестовых данных
    component_accuracies = []
    component_names = []

    # Оцениваем каждую нейронную сеть
    for i, nn_model in enumerate(super_ensemble.nn_ensemble.models):
        pred = nn_model.predict(X_test)
        pred_binary = (pred > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, pred_binary)
        component_accuracies.append(acc)
        component_names.append(f"NN_{i+1}")

    # Оцениваем другие модели
    for name, model in super_ensemble.additional_models.items():
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_test)[:, 1]
                pred_binary = (pred > 0.5).astype(int)
            else:
                pred_binary = model.predict(X_test)
            acc = accuracy_score(y_test, pred_binary)
            component_accuracies.append(acc)
            component_names.append(name)
        except Exception as e:
            print(f"Ошибка при оценке модели {name}: {str(e)}")

    # Добавляем точность самого суперансамбля
    component_accuracies.append(results['accuracy'])
    component_names.append("SuperEnsemble")

    # Сортируем по точности
    sorted_idx = np.argsort(component_accuracies)
    sorted_names = [component_names[i] for i in sorted_idx[-10:]]  # Берем топ-10
    sorted_accuracies = [component_accuracies[i] for i in sorted_idx[-10:]]

    # Визуализируем
    bars = plt.barh(sorted_names, sorted_accuracies, color='lightgreen')
    plt.title('Точность компонентов', fontsize=14)
    plt.xlabel('Точность', fontsize=12)
    plt.xlim([0.5, 1.0])  # Ограничиваем диапазон для лучшей видимости
    plt.gca().invert_yaxis()  # Наибольшая точность сверху

    # 6. Дополнительная информация (нижний правый)
    plt.subplot(2, 3, 6)
    info_str = "Информация о SuperEnsemble:\n\n"
    info_str += f"Число нейронных сетей: {super_ensemble.nn_ensemble_size}\n"
    info_str += f"Число дополнительных моделей: {len(super_ensemble.additional_models)}\n"
    info_str += f"Общее число компонентов: {len(super_ensemble.nn_ensemble.models) + len(super_ensemble.additional_models)}\n\n"
    info_str += "Дополнительные модели:\n"
    for name in super_ensemble.additional_models.keys():
        info_str += f"  - {name}\n"

    if hasattr(super_ensemble, 'meta_model') and super_ensemble.meta_model is not None:
        info_str += "\nИспользуется мета-модель для стекинга"
    elif hasattr(super_ensemble, 'optimal_weights') and super_ensemble.optimal_weights is not None:
        info_str += "\nИспользуется оптимизированное взвешенное голосование"
    else:
        info_str += "\nИспользуется простое голосование"

    plt.text(0.1, 0.1, info_str, fontsize=12, va='top', ha='left')
    plt.axis('off')

    # Сохраняем полную визуализацию
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"superensemble_metrics_monk{problem_number}.png", dpi=300)
    plt.close()
    
    # Анализ устойчивости суперансамбля к шуму
    robustness_results = analyze_superensemble_robustness(
        super_ensemble, X_test, y_test, problem_number,
        noise_types=['gaussian', 'uniform', 'impulse', 'missing'],
        noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    # Сохраняем результаты анализа устойчивости
    with open(RESULTS_DIR / f"superensemble_robustness_monk{problem_number}.json", 'w') as f:
        json.dump(robustness_results, f, indent=2)

    # Сравнительный анализ с другими алгоритмами
    if run_comparative:
        logger.info("Сравнительный анализ суперансамбля с другими алгоритмами")
        comparative = ComparativeAnalyzer(super_ensemble, problem_number)
        comparative.train_baseline_models(X_train, y_train)
        
        # Создаем массив уровней шума на основе параметров
        noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()
        
        # Проводим сравнительный анализ
        comparative_results = comparative.analyze_noise_resistance(
            X_test, y_test,
            noise_types=['gaussian', 'uniform', 'impulse', 'missing'],
            noise_levels=noise_levels,
            n_experiments=3
        )
    
    # Анализ устойчивости к шуму отдельной модели
    if analyze_noise and not run_comparative:
        logger.info("Анализ устойчивости суперансамбля к шуму")
        
        # Создаем класс для анализа шума
        class NoiseAnalyzer:
            @staticmethod
            def add_noise(X, noise_level, noise_type='gaussian'):
                logger.info(f"Добавление {noise_type} шума с уровнем {noise_level:.2f}")
                
                if noise_level <= 0:
                    return X
                    
                X_noisy = X.copy()
                n_samples, n_features = X.shape
                
                # Количество примеров для добавления шума (согласно уровню шума)
                n_noisy_samples = int(noise_level * n_samples)
                noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
                
                # Статистики признаков для реалистичного шума
                feature_means = np.mean(X, axis=0)
                feature_stds = np.std(X, axis=0)
                feature_mins = np.min(X, axis=0)
                feature_maxs = np.max(X, axis=0)
                
                # Для каждого зашумляемого примера
                for idx in noisy_indices:
                    # Выбираем случайное количество признаков для зашумления
                    n_features_to_noise = np.random.randint(1, n_features + 1)
                    features_to_noise = np.random.choice(n_features, n_features_to_noise, replace=False)
                    
                    # Добавляем шум к выбранным признакам
                    for feature_idx in features_to_noise:
                        feature_range = feature_maxs[feature_idx] - feature_mins[feature_idx]
                        
                        if noise_type == 'gaussian':
                            # Гауссовский шум, масштабируемый к стандартному отклонению признака
                            noise = np.random.normal(0, feature_stds[feature_idx])
                            X_noisy[idx, feature_idx] += noise
                            
                        elif noise_type == 'uniform':
                            # Равномерный шум в пределах диапазона признака
                            noise = np.random.uniform(-0.5, 0.5) * feature_range
                            X_noisy[idx, feature_idx] += noise
                            
                        elif noise_type == 'impulse':
                            # Импульсный шум - замена на экстремальные значения
                            impulse_type = np.random.choice(['min', 'max', 'extreme'])
                            if impulse_type == 'min':
                                X_noisy[idx, feature_idx] = feature_mins[feature_idx]
                            elif impulse_type == 'max':
                                X_noisy[idx, feature_idx] = feature_maxs[feature_idx]
                            else:  # extreme
                                extreme_factor = np.random.choice([-2, 2])
                                X_noisy[idx, feature_idx] = feature_means[feature_idx] + extreme_factor * feature_stds[feature_idx]
                                
                        elif noise_type == 'missing':
                            # Замена на NaN (требует предобработки перед использованием)
                            X_noisy[idx, feature_idx] = np.nan
                
                # Если есть пропущенные значения, заполняем их средними (импутация)
                if noise_type == 'missing':
                    for j in range(n_features):
                        mask = np.isnan(X_noisy[:, j])
                        X_noisy[mask, j] = np.mean(X_noisy[~mask, j])
                
                return X_noisy
            
            @staticmethod
            def analyze_noise_resistance(ensemble, X, y, noise_types=None, noise_levels=None, n_experiments=5):
                if noise_types is None:
                    noise_types = ['gaussian', 'uniform', 'impulse']
                    
                if noise_levels is None:
                    noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()
                    
                logger.info(f"Анализ устойчивости к шуму: {len(noise_types)} типов, {len(noise_levels)} уровней, {n_experiments} экспериментов")
                
                # Получаем базовую точность (без шума)
                baseline_preds, _ = ensemble.predict(X)
                baseline_accuracy = accuracy_score(y, baseline_preds)
                logger.info(f"Базовая точность (без шума): {baseline_accuracy:.4f}")
                
                # Структура для хранения результатов
                results = {
                    'noise_types': noise_types,
                    'noise_levels': noise_levels,
                    'baseline_accuracy': baseline_accuracy,
                    'accuracies': {}
                }
                
                # Для каждого типа шума
                for noise_type in noise_types:
                    results['accuracies'][noise_type] = []
                    
                    # Для каждого уровня шума
                    for noise_level in noise_levels:
                        level_accuracies = []
                        
                        # Повторяем эксперимент несколько раз
                        for exp in range(n_experiments):
                            # Добавляем шум
                            X_noisy = NoiseAnalyzer.add_noise(X, noise_level, noise_type)
                            
                            # Делаем предсказание
                            preds, _ = ensemble.predict(X_noisy)
                            
                            # Вычисляем точность
                            accuracy = accuracy_score(y, preds)
                            level_accuracies.append(accuracy)
                        
                        # Сохраняем средний результат
                        mean_accuracy = np.mean(level_accuracies)
                        std_accuracy = np.std(level_accuracies)
                        
                        results['accuracies'][noise_type].append({
                            'level': float(noise_level),
                            'mean_accuracy': float(mean_accuracy),
                            'std_accuracy': float(std_accuracy),
                            'experiments': [float(acc) for acc in level_accuracies]
                        })
                        
                        logger.info(f"Шум '{noise_type}' уровня {noise_level:.1f}: точность = {mean_accuracy:.4f} (±{std_accuracy:.4f})")
                
                # Визуализация результатов
                NoiseAnalyzer.visualize_noise_resistance(results, ensemble.problem_number)
                
                # Сохраняем результаты
                with open(RESULTS_DIR / f"noise_analysis_monk{ensemble.problem_number}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                    
                return results
            
            @staticmethod
            def visualize_noise_resistance(results, problem_number):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Строим график для каждого типа шума
                for noise_type in results['noise_types']:
                    noise_data = results['accuracies'][noise_type]
                    levels = [data['level'] for data in noise_data]
                    accuracies = [data['mean_accuracy'] for data in noise_data]
                    errors = [data['std_accuracy'] for data in noise_data]
                    
                    ax.errorbar(
                        levels, 
                        accuracies, 
                        yerr=errors, 
                        marker='o', 
                        linestyle='-', 
                        capsize=5, 
                        label=f'{noise_type.capitalize()} Noise'
                    )
                
                # Добавляем базовую линию (без шума)
                ax.axhline(
                    y=results['baseline_accuracy'], 
                    color='black', 
                    linestyle='--', 
                    label=f'Baseline (No Noise): {results["baseline_accuracy"]:.4f}'
                )
                
                # Настраиваем график
                ax.set_title(f'Устойчивость к шуму для MONK-{problem_number}', fontsize=16)
                ax.set_xlabel('Уровень шума', fontsize=14)
                ax.set_ylabel('Точность', fontsize=14)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(-0.02, max(results['noise_levels']) + 0.02)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12)
                
                # Сохраняем график
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"noise_resistance_monk{problem_number}.png", dpi=300)
                plt.close()
        
        # Создаем массив уровней шума на основе параметров
        noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()

        # Анализируем устойчивость к шуму
        noise_analysis = NoiseAnalyzer.analyze_noise_resistance(
            super_ensemble, X_test, y_test,
            noise_types=['gaussian', 'uniform', 'impulse', 'missing'],
            noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
            n_experiments=3
        )
    
    return super_ensemble

def analyze_superensemble_robustness(super_ensemble, X_test, y_test, problem_number, 
                                    noise_types=['gaussian', 'uniform', 'impulse'], 
                                    noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Анализирует устойчивость суперансамбля к шуму по сравнению с компонентами.
    
    Args:
        super_ensemble: Обученный SuperEnsemble
        X_test: Тестовые данные
        y_test: Истинные метки
        problem_number: Номер задачи MONK
        noise_types: Типы шума для анализа
        noise_levels: Уровни шума для анализа
    """
        # Выбираем несколько компонентов для сравнения
    models_to_compare = {
        "SuperEnsemble": super_ensemble
    }
    
    # Добавляем несколько ключевых компонентов (первая НС, лучшая традиционная модель)
    if len(super_ensemble.nn_ensemble.models) > 0:
        models_to_compare["Best NN"] = super_ensemble.nn_ensemble.models[0]
    
    # Выбираем лучшую традиционную модель
    if super_ensemble.additional_models:
        best_traditional_model = None
        best_accuracy = 0
        
        for name, model in super_ensemble.additional_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_test)[:, 1]
                    pred_binary = (pred > 0.5).astype(int)
                else:
                    pred_binary = model.predict(X_test)
                    # Убедимся, что предсказания бинарные
                    if np.issubdtype(pred_binary.dtype, np.floating):
                        pred_binary = (pred_binary > 0.5).astype(int)
                acc = accuracy_score(y_test, pred_binary)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_traditional_model = (name, model)
            except:
                continue
        
        if best_traditional_model:
            models_to_compare[best_traditional_model[0]] = best_traditional_model[1]
    
    # Результаты для каждого типа шума
    results = {}
    
    # Для каждого типа шума
    for noise_type in noise_types:
        results[noise_type] = {}
        
        # Для каждой модели
        for model_name, model in models_to_compare.items():
            results[noise_type][model_name] = []
            
            # Для каждого уровня шума
            for noise_level in noise_levels:
                # Добавляем шум к тестовым данным
                X_noisy = add_noise_to_features(X_test, noise_level, noise_type)
                
                # Делаем предсказания
                if model_name == "SuperEnsemble":
                    y_pred, y_prob = model.predict(X_noisy)
                    # Убедимся, что предсказания бинарные
                    if np.issubdtype(y_pred.dtype, np.floating):
                        y_pred = (y_pred > 0.5).astype(int)
                elif hasattr(model, 'predict_proba'):  # Для моделей с вероятностями
                    y_prob = model.predict(X_noisy)
                    y_pred = (y_prob > 0.5).astype(int).flatten()
                else:  # Для остальных моделей
                    y_pred = model.predict(X_noisy)
                    # Убедимся, что предсказания бинарные
                    if np.issubdtype(y_pred.dtype, np.floating):
                        y_pred = (y_pred > 0.5).astype(int)
                    if y_pred.ndim > 1:
                        y_pred = y_pred.flatten()
                
                # Вычисляем точность
                accuracy = accuracy_score(y_test, y_pred)
                
                results[noise_type][model_name].append({
                    'level': noise_level,
                    'accuracy': float(accuracy)
                })
                
                logger.info(f"Шум '{noise_type}' уровня {noise_level}: {model_name} точность = {accuracy:.4f}")
    
    # Визуализация результатов
    for noise_type in noise_types:
        plt.figure(figsize=(10, 6))
        
        for model_name in models_to_compare.keys():
            noise_data = results[noise_type][model_name]
            levels = [data['level'] for data in noise_data]
            accuracies = [data['accuracy'] for data in noise_data]
            
            line_style = '-' if model_name == "SuperEnsemble" else '--'
            line_width = 2 if model_name == "SuperEnsemble" else 1
            
            plt.plot(levels, accuracies, marker='o', linestyle=line_style, 
                     linewidth=line_width, label=model_name)
        
        plt.title(f'Устойчивость к шуму {noise_type.capitalize()} для MONK-{problem_number}', fontsize=14)
        plt.xlabel('Уровень шума', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.ylim(0.5, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"superensemble_{noise_type}_robustness_monk{problem_number}.png", dpi=300)
        plt.close()
    
    # Комбинированная визуализация для финального сравнения
    plt.figure(figsize=(15, 10))
    
    for i, noise_type in enumerate(noise_types):
        plt.subplot(2, 2, i+1)
        
        for model_name in models_to_compare.keys():
            noise_data = results[noise_type][model_name]
            levels = [data['level'] for data in noise_data]
            accuracies = [data['accuracy'] for data in noise_data]
            
            line_style = '-' if model_name == "SuperEnsemble" else '--'
            line_width = 2 if model_name == "SuperEnsemble" else 1
            
            plt.plot(levels, accuracies, marker='o', linestyle=line_style, 
                     linewidth=line_width, label=model_name)
        
        plt.title(f'Шум {noise_type.capitalize()}', fontsize=12)
        plt.xlabel('Уровень шума', fontsize=10)
        plt.ylabel('Точность', fontsize=10)
        plt.ylim(0.5, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
    
    plt.suptitle(f'Сравнение устойчивости к шуму для MONK-{problem_number}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(RESULTS_DIR / f"superensemble_comparative_robustness_monk{problem_number}.png", dpi=300)
    plt.close()
    
    return results

if __name__ == "__main__":
    print("=== Запуск оптимального классификатора для набора данных MONK ===")
    
    # Запрос выбора задачи
    while True:
        try:
            problem_number = int(input("Выберите задачу MONK (1, 2, или 3): "))
            if problem_number in [1, 2, 3]:
                break
            else:
                print("Пожалуйста, введите 1, 2 или 3.")
        except ValueError:
            print("Пожалуйста, введите число.")
    
    # Запрос параметров запуска
    run_hyperopt = input("Выполнить оптимизацию гиперпараметров? (y/n, по умолчанию: y): ").lower() != 'n'
    
    if run_hyperopt:
        while True:
            try:
                n_trials = int(input("Количество попыток для оптимизации (по умолчанию: 50): ") or "50")
                if n_trials > 0:
                    break
                else:
                    print("Пожалуйста, введите положительное число.")
            except ValueError:
                print("Пожалуйста, введите целое число.")
    else:
        n_trials = 50
    
    while True:
        try:
            ensemble_size = int(input("Размер ансамбля моделей (по умолчанию: 5): ") or "5")
            if ensemble_size > 0:
                break
            else:
                print("Пожалуйста, введите положительное число.")
        except ValueError:
            print("Пожалуйста, введите целое число.")
    
    # Опция выбора типа анализа (только шум или сравнительный)
    analyze_option = input("Выберите тип анализа:\n1. Только анализ шума\n2. Сравнительный анализ с другими алгоритмами\nВыбор (1/2, по умолчанию: 2): ") or "2"
    analyze_noise = analyze_option == "1"
    run_comparative = analyze_option == "2"
    
    # Если выбран анализ шума, запрашиваем параметры шума
    min_noise = 0.0
    max_noise = 0.5
    noise_step = 0.1
    
    if analyze_noise or run_comparative:
        print("\n=== Настройка параметров шума ===")
        while True:
            try:
                min_noise = float(input("Минимальный уровень шума в % (по умолчанию: 0): ") or "0") / 100
                if 0 <= min_noise <= 1:
                    break
                else:
                    print("Пожалуйста, введите значение от 0 до 100.")
            except ValueError:
                print("Пожалуйста, введите число.")
                
        while True:
            try:
                max_noise = float(input("Максимальный уровень шума в % (по умолчанию: 50): ") or "50") / 100
                if min_noise <= max_noise <= 1:
                    break
                else:
                    print(f"Пожалуйста, введите значение от {min_noise*100} до 100.")
            except ValueError:
                print("Пожалуйста, введите число.")
                
        while True:
            try:
                noise_step = float(input("Шаг изменения уровня шума в % (по умолчанию: 10): ") or "10") / 100
                if 0 < noise_step <= (max_noise - min_noise + 0.0001):
                    break
                else:
                    print(f"Пожалуйста, введите положительное значение, не превышающее {(max_noise - min_noise) * 100:.1f}%.")
            except ValueError:
                print("Пожалуйста, введите число.")
        
        # Показываем пользователю рассчитанные уровни шума
        noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step)
        print(f"\nБудут рассчитаны следующие уровни шума: {[f'{level*100:.1f}%' for level in noise_levels]}")
    
    # Выбор типа ансамбля
    ensemble_type = input("\nВыберите тип ансамбля:\n1. Базовый ансамбль\n2. SuperEnsemble (с дополнительными алгоритмами)\nВыбор (1/2, по умолчанию: 1): ") or "1"
    
    # Запуск выбранного типа ансамбля
    if ensemble_type == "1":
        # Базовый ансамбль
        print(f"\nЗапуск базового ансамбля для MONK-{problem_number}...")
        try:
            ensemble = run_optimal_classifier(
                problem_number, 
                run_hyperopt=run_hyperopt, 
                n_trials=n_trials, 
                ensemble_size=ensemble_size,
                analyze_noise=analyze_noise,
                run_comparative=run_comparative,
                min_noise=min_noise,
                max_noise=max_noise,
                noise_step=noise_step
            )
            
            print("\n=== Обучение и оценка завершены ===")
            print(f"Результаты сохранены в директории: {RESULTS_DIR}")
            
        except Exception as e:
            print(f"\nОшибка: {str(e)}")
            logger.exception("Необработанное исключение в основной программе")
            raise
    else:
        # SuperEnsemble
        print(f"\nЗапуск SuperEnsemble для MONK-{problem_number}...")
        try:
            super_ensemble = run_super_ensemble(
                problem_number,
                nn_ensemble_size=ensemble_size,
                run_hyperopt=run_hyperopt,
                n_trials=n_trials,
                analyze_noise=analyze_noise,
                run_comparative=run_comparative,
                min_noise=min_noise,
                max_noise=max_noise,
                noise_step=noise_step
            )
            
            print("\n=== Обучение и оценка SuperEnsemble завершены ===")
            print(f"Результаты сохранены в директории: {RESULTS_DIR}")
            
        except Exception as e:
            print(f"\nОшибка: {str(e)}")
            logger.exception("Необработанное исключение в основной программе")
            raise