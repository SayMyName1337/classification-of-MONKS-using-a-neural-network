import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Model, Sequential, load_model, save_model
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
        
        # Различные архитектуры для разнообразия моделей
        activations = ['relu', 'elu', 'selu', 'swish', 'tanh']
        layer_configs = [
            [int(self.base_params['units_first'] * 2), int(self.base_params['units_first']), int(self.base_params['units_first'] // 2)],
            [int(self.base_params['units_first']), int(self.base_params['units_first'] // 2), int(self.base_params['units_first'] // 4)],
            [int(self.base_params['units_first'] * 1.5), int(self.base_params['units_first'] * 0.75), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.25)],
            [int(self.base_params['units_first']), int(self.base_params['units_first']), int(self.base_params['units_first'] // 2)],
            [int(self.base_params['units_first'] // 2), int(self.base_params['units_first']), int(self.base_params['units_first'] // 2)]
        ]
        
        # Для каждой модели в ансамбле
        for i in range(self.num_models):
            # Создаем вариацию гиперпараметров
            params = self.base_params.copy()
            
            # Выбираем вариации для разнообразия
            params['activation'] = activations[i % len(activations)]
            params['learning_rate'] = float(self.base_params['learning_rate'] * (0.5 + i * 0.1))
            params['batch_size'] = int(self.base_params.get('batch_size', 32) * (1 + i % 3))
            params['dropout_rate'] = float(min(0.5, self.base_params.get('dropout_rate', 0.3) + 0.05 * (i % 5)))
            
            # Выбираем конфигурацию слоев
            layer_config = layer_configs[i % len(layer_configs)]
            
            logger.info(f"Обучение модели {i+1}/{self.num_models} с {params['activation']} и LR={params['learning_rate']:.6f}")
            
            # Создаем модель с текущей конфигурацией
            model = self._create_model_with_layers(params, layer_config)
            
            # Выбираем оптимизатор
            if i % 3 == 0:
                optimizer = Adam(learning_rate=float(params['learning_rate']))
            elif i % 3 == 1:
                optimizer = RMSprop(learning_rate=float(params['learning_rate']))
            else:
                optimizer = SGD(learning_rate=float(params['learning_rate']), momentum=0.9)
            
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
                    patience=20,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
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
            
            # Обучаем модель
            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,  # Большое количество эпох, но с ранней остановкой
                    batch_size=int(params['batch_size']),  # Убедимся, что это int
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Загружаем лучшую модель
                model = load_model(ensemble_dir / f"model_{i+1}.h5")
                
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
                logger.info(f"Модель {i+1} обучена: val_accuracy={model_meta['val_accuracy']:.4f}, epochs={model_meta['epochs_trained']}")
            
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {i+1}: {str(e)}")
                # Продолжаем со следующей моделью
                continue
    
    def predict(self, X):
        """
        Выполняет прогнозирование ансамблем моделей.
        
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
        
        # Получаем предсказания от каждой модели
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
            
        # Усредняем прогнозы
        ensemble_predictions = np.mean(predictions, axis=0)
        
        # Преобразуем в бинарные метки
        binary_predictions = (ensemble_predictions > 0.5).astype(int).flatten()
        
        return binary_predictions, ensemble_predictions
    
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
        Создает модель с указанной конфигурацией слоев.
        
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
        
        # Создаем последовательную модель
        model = Sequential()
        
        # Входной слой
        model.add(Input(shape=(self.input_shape,)))
        
        # Каждый скрытый слой
        for i, units in enumerate(layer_units):
            # Добавляем Dense слой
            model.add(Dense(
                int(units),  # Убедимся, что это int
                kernel_regularizer=regularizer
            ))
            
            # Активация
            if params['activation'] == 'leaky_relu':
                model.add(LeakyReLU(alpha=0.1))
            else:
                model.add(Activation(params['activation']))
            
            # Нормализация и регуляризация
            if params.get('use_batch_norm', True):
                model.add(BatchNormalization())
                
            dropout_rate = float(params.get('dropout_rate', 0.3))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))
        
        return model

def run_optimal_classifier(problem_number, run_hyperopt=True, n_trials=50, ensemble_size=5, analyze_noise=True,
                          min_noise=0.0, max_noise=0.5, noise_step=0.1):
    """
    Запускает полный процесс оптимизации, обучения и анализа классификатора.
    
    Args:
        problem_number: Номер задачи MONK (1, 2, или 3)
        run_hyperopt: Выполнять ли оптимизацию гиперпараметров
        n_trials: Количество попыток для оптимизации гиперпараметров
        ensemble_size: Размер ансамбля моделей
        analyze_noise: Выполнять ли анализ устойчивости к шуму
        min_noise: Минимальный уровень шума (0.0 - 1.0)
        max_noise: Максимальный уровень шума (0.0 - 1.0)
        noise_step: Шаг изменения уровня шума (0.0 - 1.0)
        
    Returns:
        EnsembleClassifier: Обученный ансамбль моделей
    """
    logger.info(f"=== Запуск оптимального классификатора для MONK-{problem_number} ===")
    start_time = time.time()
    
    # 1. Загрузка и подготовка данных
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
    
    # 2. Оптимизация гиперпараметров (или загрузка ранее найденных)
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
                best_params = {
                    'n_layers': 3,
                    'units_first': 128 if problem_number == 2 else 64,
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
    
    # 3. Построение и обучение ансамбля моделей
    logger.info(f"Создание ансамбля из {ensemble_size} моделей")
    ensemble = EnsembleClassifier(problem_number, base_params=best_params, num_models=ensemble_size)
    ensemble.build_ensemble(X_train, y_train, X_val, y_val)
    
    # 4. Оценка на тестовой выборке
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
    
    # 5. Анализ устойчивости к шуму
    if analyze_noise:
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
    
    analyze_noise = input("Выполнить анализ устойчивости к шуму? (y/n, по умолчанию: y): ").lower() != 'n'
    
    # Новые параметры для настройки шума
    min_noise = 0.0
    max_noise = 0.5
    noise_step = 0.1
    
    if analyze_noise:
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
    
    # Запуск классификатора с новыми параметрами
    print(f"\nЗапуск классификатора для MONK-{problem_number}...")
    try:
        ensemble = run_optimal_classifier(
            problem_number, 
            run_hyperopt=run_hyperopt, 
            n_trials=n_trials, 
            ensemble_size=ensemble_size, 
            analyze_noise=analyze_noise,
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