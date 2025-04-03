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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
import requests
from pathlib import Path
from io import StringIO

# Подавляем предупреждения для более чистого вывода
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("heart_disease_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HeartDiseaseClassifier")

# Установка seed для воспроизводимости
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Создание директорий для моделей и результатов
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

class DataLoader:
    """Класс для загрузки и предобработки данных о заболеваниях сердца."""
    
    @staticmethod
    def load_heart_disease_data(data_filepath: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Загружает данные о заболеваниях сердца и выполняет предобработку признаков.
        
        Args:
            data_filepath: Путь к данным (если None, данные будут загружены с UCI)
            
        Returns:
            X_train, y_train, X_test, y_test: Обработанные признаки и метки для обучения и тестирования
        """
        logger.info("Загрузка данных о заболеваниях сердца")
        
        try:
            # Если файл не указан или не существует, загружаем с UCI
            if data_filepath is None or not os.path.exists(data_filepath):
                logger.info("Загрузка данных с UCI Repository...")
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
                response = requests.get(url)
                data = pd.read_csv(StringIO(response.text), header=None)
                
                # Сохраняем локально для будущего использования
                local_path = DATA_DIR / "heart_cleveland.csv"
                data.to_csv(local_path, index=False)
                logger.info(f"Данные сохранены локально: {local_path}")
            else:
                logger.info(f"Загрузка данных из локального файла: {data_filepath}")
                data = pd.read_csv(data_filepath, header=None)
            
            # Назначаем имена столбцов
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            data.columns = column_names
            
            # Обработка отсутствующих значений
            # В датасете Cleveland отсутствующие значения обозначены как "?"
            data = data.replace('?', np.nan)
            
            # Преобразование столбцов в числовой формат
            for col in data.columns:
                if data[col].dtype == object:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Проверка наличия NA и их заполнение
            if data.isna().sum().sum() > 0:
                logger.info(f"Найдены отсутствующие значения: {data.isna().sum().sum()}")
                # Заполняем отсутствующие значения наиболее частыми значениями для категориальных
                # и медианами для числовых признаков
                for col in data.columns:
                    if col in ['ca', 'thal']:  # Категориальные признаки
                        data[col] = data[col].fillna(data[col].mode()[0])
                    else:  # Числовые признаки
                        data[col] = data[col].fillna(data[col].median())
            
            # Преобразование целевой переменной (>0 означает наличие заболевания)
            data['target'] = (data['target'] > 0).astype(int)
            
            # Инженерия признаков
            data = DataLoader.enhance_features(data)
            
            # Выделяем признаки и целевую переменную
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Логируем информацию о данных
            logger.info(f"Загружено {len(data)} записей, {X.shape[1]} признаков")
            logger.info(f"Распределение классов: {y.value_counts().to_dict()}")
            
            # Масштабирование числовых признаков
            numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            scaler = StandardScaler()
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
            )
            
            # Преобразование в numpy массивы с правильными типами данных
            X_train_array = X_train.values.astype(np.float32)
            y_train_array = y_train.values.astype(np.float32)
            X_test_array = X_test.values.astype(np.float32)
            y_test_array = y_test.values.astype(np.float32)
            
            logger.info(f"Данные успешно подготовлены: {len(X_train)} обучающих и {len(X_test)} тестовых примеров")
            
            # Сохраняем обработанные данные
            processed_data = {
                'X_train': X_train_array,
                'y_train': y_train_array,
                'X_test': X_test_array,
                'y_test': y_test_array,
                'feature_names': X.columns.tolist(),
                'scaler': scaler
            }
            
            with open(DATA_DIR / "processed_heart_data.pkl", "wb") as f:
                pickle.dump(processed_data, f)
            
            return X_train_array, y_train_array, X_test_array, y_test_array
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
    
    @staticmethod
    def enhance_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Расширяет набор признаков для улучшения обучения моделей.
        
        Args:
            data: DataFrame с исходными признаками
            
        Returns:
            DataFrame с расширенным набором признаков
        """
        df = data.copy()
        
        # 1. Возрастные группы
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], 
                               labels=[0, 1, 2, 3]).astype(int)
        
        # 2. Индекс массы тела (BMI) - не можем рассчитать напрямую, но создадим заменяющие признаки
        
        # 3. Отношение холестерина к тахикардии 
        df['chol_thalach_ratio'] = df['chol'] / df['thalach']
        
        # 4. Взаимодействие между полом и типом боли в груди
        df['sex_cp'] = df['sex'] * df['cp']
        
        # 5. Взаимодействие между возрастом и максимальной частотой сердечных сокращений
        df['age_thalach'] = df['age'] * df['thalach']
        
        # 6. Взаимодействие между стенокардией и депрессией ST
        df['exang_oldpeak'] = df['exang'] * df['oldpeak']
        
        # 7. Категории артериального давления
        df['bp_category'] = pd.cut(df['trestbps'], 
                                 bins=[0, 120, 140, 160, 200], 
                                 labels=[0, 1, 2, 3]).astype(int)
        
        # 8. Категории холестерина
        df['chol_category'] = pd.cut(df['chol'], 
                                   bins=[0, 200, 240, 300, 600], 
                                   labels=[0, 1, 2, 3]).astype(int)
        
        # 9. Клинический индекс риска - комбинация нескольких факторов
        # (упрощенная версия)
        risk_factors = (df['age'] > 50).astype(int) + df['sex'] + df['fbs'] + df['exang']
        df['risk_index'] = risk_factors
        
        # 10. Полиномиальные признаки для ключевых переменных
        df['age_squared'] = df['age'] ** 2
        df['thalach_squared'] = df['thalach'] ** 2
        df['oldpeak_squared'] = df['oldpeak'] ** 2
        
        # 11. Отношение возраста к максимальной частоте сердечных сокращений
        df['age_thalach_ratio'] = df['age'] / df['thalach']
        
        # 12. Взаимодействие между количеством крупных сосудов и талассемией
        df['ca_thal'] = df['ca'] * df['thal']
        
        # Можно добавить и другие признаки, основанные на медицинских знаниях

        # Логгируем информацию о новых признаках
        new_features = [col for col in df.columns if col not in data.columns]
        logger.info(f"Добавлено {len(new_features)} новых признаков: {new_features}")
        
        return df

class HyperOptimizer:
    """Оптимизация гиперпараметров с использованием Optuna."""
    
    @staticmethod
    def optimize_hyperparameters(X_train, y_train, X_val, y_val, dataset_name='heart', n_trials=100):
        """
        Выполняет оптимизацию гиперпараметров для нейронной сети.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
            dataset_name: Название датасета для логов
            n_trials: Количество попыток оптимизации
            
        Returns:
            Лучшие найденные гиперпараметры
        """
        logger.info(f"Запуск оптимизации гиперпараметров для {dataset_name} с {n_trials} попытками")
        
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
            study_name=f'{dataset_name}_optimization'
        )
        
        # Запускаем оптимизацию
        study.optimize(objective, n_trials=n_trials)
        
        # Логируем результаты
        logger.info(f"Лучшие гиперпараметры для {dataset_name}: {study.best_params}")
        logger.info(f"Достигнутая точность: {study.best_value:.4f}")
        
        # Сохраняем результаты исследования
        with open(RESULTS_DIR / f"hyperopt_results_{dataset_name}.pkl", "wb") as f:
            pickle.dump(study, f)
        
        # Визуализация результатов
        try:
            # Импортируем нужные модули здесь, чтобы они не были обязательными
            import plotly
            
            # Сохраняем график важности параметров
            param_importances = optuna.visualization.plot_param_importances(study)
            fig_importance = param_importances.update_layout(
                title=f"Важность параметров для {dataset_name}", 
                width=1000, 
                height=600
            )
            fig_importance.write_image(RESULTS_DIR / f"param_importance_{dataset_name}.png")
            
            # Сохраняем оптимизационную историю
            optimization_history = optuna.visualization.plot_optimization_history(study)
            fig_history = optimization_history.update_layout(
                title=f"История оптимизации для {dataset_name}", 
                width=1000, 
                height=600
            )
            fig_history.write_image(RESULTS_DIR / f"optimization_history_{dataset_name}.png")
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
    
    def __init__(self, dataset_name, base_params=None, num_models=5):
        """
        Инициализация классификатора.
        
        Args:
            dataset_name: Название датасета
            base_params: Базовые гиперпараметры для вариации
            num_models: Количество моделей в ансамбле
        """
        self.dataset_name = dataset_name
        self.base_params = base_params
        self.num_models = num_models
        self.models = []
        self.input_shape = None
        
        logger.info(f"Инициализирован EnsembleClassifier для {dataset_name} с {num_models} моделями")
    
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
        ensemble_dir = MODELS_DIR / f"{self.dataset_name}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        
        # УЛУЧШЕНИЕ: Более разнообразные архитектуры для моделей
        activations = ['relu', 'elu', 'selu', 'tanh', 'sigmoid']
        
        # Значительно разнообразим конфигурации слоев
        layer_configs = [
            [int(self.base_params['units_first'] * 4), int(self.base_params['units_first'] * 2), int(self.base_params['units_first'])],  # Широкая сеть
            [int(self.base_params['units_first']), int(self.base_params['units_first'] * 2), int(self.base_params['units_first'])],  # Расширяющаяся в середине
            [int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.5)],  # Узкая и глубокая
            [int(self.base_params['units_first'] * 2), int(self.base_params['units_first'] * 0.5)],  # Резкое сужение
            [int(self.base_params['units_first']), int(self.base_params['units_first'] * 0.75), int(self.base_params['units_first'] * 0.5), int(self.base_params['units_first'] * 0.25)]  # Плавное сужение
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
        Выполняет прогнозирование ансамблем моделей с адаптивной стратегией голосования.
        
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
        ensemble_dir = MODELS_DIR / f"{self.dataset_name}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = ensemble_dir / f"model_{i+1}.h5"
            model.save(model_path)
        
        # Сохраняем метаинформацию ансамбля
        meta = {
            'dataset_name': self.dataset_name,
            'num_models': int(self.num_models),
            'base_params': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in self.base_params.items()} if self.base_params else {},
            'input_shape': int(self.input_shape) if self.input_shape is not None else None
        }
        
        with open(ensemble_dir / "ensemble_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Ансамбль из {self.num_models} моделей сохранен в {ensemble_dir}")
    
    @classmethod
    def load(cls, dataset_name):
        """
        Загружает ансамбль моделей с диска.
        
        Args:
            dataset_name: Название датасета
            
        Returns:
            Экземпляр EnsembleClassifier с загруженными моделями
        """
        ensemble_dir = MODELS_DIR / f"{dataset_name}_ensemble"
        
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
            dataset_name=meta['dataset_name'],
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
        
        logger.info(f"Загружен ансамбль из {len(ensemble.models)} моделей для {dataset_name}")
        
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

class ComparativeAnalyzer:
    """Класс для сравнительного анализа различных алгоритмов в условиях шума."""
    
    def __init__(self, ensemble_model, dataset_name):
        """
        Инициализация анализатора.
        
        Args:
            ensemble_model: Обученный ансамбль моделей
            dataset_name: Название датасета
        """
        self.ensemble_model = ensemble_model
        self.dataset_name = dataset_name
        self.models = {}
        self.model_names = []
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Добавляем ансамбль в словарь моделей
        self.models["Ensemble NN"] = ensemble_model
        self.model_names.append("Ensemble NN")
        
        # Логирование
        logger.info(f"Инициализирован ComparativeAnalyzer для {dataset_name}")
    
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
        
        # Настройки для разных моделей
        rf_estimators = 200
        gb_estimators = 200
        mlp_hidden_layer_sizes = (100, 50, 25)
        
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
        with open(self.results_dir / f"comparative_analysis_{self.dataset_name}.json", 'w') as f:
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
            plt.title(f'Сравнительный анализ устойчивости к шуму {noise_type.capitalize()} для {self.dataset_name}', fontsize=16)
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
            plt.savefig(self.results_dir / f"comparative_{noise_type}_{self.dataset_name}.png", dpi=300)
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
        plt.title(f'Устойчивость ансамбля к различным типам шума для {self.dataset_name}', fontsize=16)
        plt.xlabel('Уровень шума', fontsize=14)
        plt.ylabel('Точность', fontsize=14)
        plt.ylim(0, 1.05)
        plt.xlim(-0.02, max(noise_levels) + 0.02)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Сохраняем график
        plt.tight_layout()
        plt.savefig(self.results_dir / f"ensemble_noise_comparison_{self.dataset_name}.png", dpi=300)
        plt.close()

class SuperEnsemble:
    """Расширенный ансамбль с улучшенной производительностью и устойчивостью."""
    
    def __init__(self, dataset_name, base_params=None, nn_ensemble_size=5):
        """
        Инициализирует суперансамбль.
        
        Args:
            dataset_name: Название датасета
            base_params: Базовые параметры для нейронных сетей
            nn_ensemble_size: Размер ансамбля нейронных сетей
        """
        self.dataset_name = dataset_name
        self.base_params = base_params
        self.nn_ensemble_size = nn_ensemble_size
        self.nn_ensemble = None
        self.additional_models = {}  # Словарь для дополнительных алгоритмов
        self.meta_model = None
        self.optimal_weights = None
        
        logger.info(f"Инициализирован SuperEnsemble для {dataset_name}")
    
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Обучает суперансамбль.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
            
        Returns:
            self: Обученный экземпляр SuperEnsemble
        """
        logger.info("Начало обучения SuperEnsemble")
        
        # Шаг 1: Обучаем ансамбль нейронных сетей
        logger.info(f"Обучение ансамбля из {self.nn_ensemble_size} нейронных сетей")
        self.nn_ensemble = EnsembleClassifier(
            self.dataset_name, 
            base_params=self.base_params, 
            num_models=self.nn_ensemble_size
        )
        self.nn_ensemble.build_ensemble(X_train, y_train, X_val, y_val)
        
        # Шаг 2: Добавляем базовые алгоритмы машинного обучения
        self._add_basic_models(X_train, y_train)
        
        # Шаг 3: Пробуем добавить продвинутые алгоритмы (если доступны)
        self._try_add_advanced_models(X_train, y_train)
        
        # Шаг 4: Обучаем мета-модель (стекинг)
        self._train_meta_model(X_train, y_train, X_val, y_val)
        
        logger.info("SuperEnsemble обучен успешно")
        return self
    
    def _add_basic_models(self, X_train, y_train):
        """Добавляет базовые алгоритмы машинного обучения."""
        logger.info("Добавление базовых алгоритмов")
        
        # Определяем модели для добавления
        models = {
            'rf_100': RandomForestClassifier(n_estimators=100, random_state=42),
            'rf_200': RandomForestClassifier(n_estimators=200, random_state=43),
            'gb_100': GradientBoostingClassifier(n_estimators=100, random_state=44),
            'gb_200': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=45),
            'svm_rbf': SVC(kernel='rbf', probability=True, random_state=46),
            'svm_poly': SVC(kernel='poly', degree=3, probability=True, random_state=47),
            'knn_5': KNeighborsClassifier(n_neighbors=5),
            'knn_7': KNeighborsClassifier(n_neighbors=7),
            'lr': LogisticRegression(C=1.0, max_iter=1000, random_state=48)
        }
        
        # Обучаем и добавляем каждую модель
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                self.additional_models[name] = model
                logger.info(f"Обучена и добавлена модель: {name}")
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {name}: {str(e)}")
    
    def _try_add_advanced_models(self, X_train, y_train):
        """Пробует добавить продвинутые алгоритмы, если доступны."""
        try:
            # Пробуем импортировать продвинутые библиотеки
            logger.info("Попытка добавления продвинутых алгоритмов")
            import_success = False
            
            # XGBoost
            try:
                import xgboost as xgb
                xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=49)
                xgb_model.fit(X_train, y_train)
                self.additional_models['xgb'] = xgb_model
                logger.info("Добавлен XGBoost")
                import_success = True
            except ImportError:
                logger.info("XGBoost недоступен")
            
            # LightGBM
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=50)
                lgb_model.fit(X_train, y_train)
                self.additional_models['lgbm'] = lgb_model
                logger.info("Добавлен LightGBM")
                import_success = True
            except ImportError:
                logger.info("LightGBM недоступен")
            
            # CatBoost
            try:
                from catboost import CatBoostClassifier
                cb_model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, random_seed=51, verbose=False)
                cb_model.fit(X_train, y_train)
                self.additional_models['catboost'] = cb_model
                logger.info("Добавлен CatBoost")
                import_success = True
            except ImportError:
                logger.info("CatBoost недоступен")
            
            if not import_success:
                logger.warning("Не удалось добавить ни одного продвинутого алгоритма")
                
        except Exception as e:
            logger.error(f"Ошибка при добавлении продвинутых алгоритмов: {str(e)}")
    
    def _train_meta_model(self, X_train, y_train, X_val, y_val):
        """Обучает мета-модель для стекинга."""
        logger.info("Обучение мета-модели для стекинга")
        
        try:
            # Получаем предсказания от всех моделей на валидационной выборке
            meta_features = self._get_meta_features(X_val)
            
            # Обучаем мета-модель
            meta_model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=3, 
                random_state=52
            )
            meta_model.fit(meta_features, y_val)
            self.meta_model = meta_model

            # Пробуем использовать генетическую оптимизацию, если не получится - используем простую
            try:
                genetic_weights = self._optimize_genetic_weights(meta_features, y_val)
                if genetic_weights is None:
                    self._optimize_simple_weights(meta_features, y_val)
            except Exception as e:
                logger.warning(f"Ошибка при генетической оптимизации: {str(e)}")
                self._optimize_simple_weights(meta_features, y_val)
            
            logger.info("Мета-модель обучена успешно")
        except Exception as e:
            logger.error(f"Ошибка при обучении мета-модели: {str(e)}")
            logger.info("Будет использоваться взвешенное голосование")
    
    def _get_meta_features(self, X):
        """Получает предсказания от всех моделей для стекинга."""
        all_predictions = []
        
        # Предсказания от нейронных сетей
        for model in self.nn_ensemble.models:
            try:
                preds = model.predict(X, verbose=0).flatten()
                all_predictions.append(preds)
            except Exception as e:
                logger.warning(f"Ошибка при получении предсказаний от нейронной сети: {str(e)}")
                # В случае ошибки добавляем нулевые предсказания
                all_predictions.append(np.zeros(X.shape[0]))
        
        # Предсказания от дополнительных моделей
        for name, model in self.additional_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X)
                all_predictions.append(preds)
            except Exception as e:
                logger.warning(f"Ошибка при получении предсказаний от модели {name}: {str(e)}")
                all_predictions.append(np.zeros(X.shape[0]))
        
        # Преобразуем в numpy массив и транспонируем для получения формы [n_samples, n_models]
        return np.column_stack(all_predictions)
    
    def _optimize_simple_weights(self, meta_features, y_val):
        """Простая оптимизация весов на основе точности моделей."""
        n_models = meta_features.shape[1]
        accuracies = []
        
        # Вычисляем точность каждой модели
        for i in range(n_models):
            binary_preds = (meta_features[:, i] > 0.5).astype(int)
            acc = accuracy_score(y_val, binary_preds)
            accuracies.append(acc)
        
        # Преобразуем точности в веса
        weights = np.array(accuracies)
        weights = weights / np.sum(weights)  # Нормализация
        
        self.optimal_weights = weights
        logger.info(f"Оптимизированы веса для {n_models} моделей")
    
    def predict(self, X):
        """
        Делает предсказание с помощью суперансамбля.
        
        Args:
            X: Входные признаки
            
        Returns:
            y_pred: Бинарные предсказания
            y_prob: Вероятностные предсказания
        """
        # Получаем предсказания от всех моделей
        meta_features = self._get_meta_features(X)
        
        # Если мета-модель доступна, используем её
        if self.meta_model is not None:
            try:
                y_prob = self.meta_model.predict_proba(meta_features)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
                return y_pred, y_prob.reshape(-1, 1)
            except Exception as e:
                logger.warning(f"Ошибка при использовании мета-модели: {str(e)}")
                logger.info("Переключение на взвешенное голосование")
        
        # Взвешенное голосование
        if self.optimal_weights is not None:
            y_prob = np.dot(meta_features, self.optimal_weights)
        else:
            # Равные веса, если оптимальные недоступны
            y_prob = np.mean(meta_features, axis=1)
        
        y_pred = (y_prob > 0.5).astype(int)
        return y_pred, y_prob.reshape(-1, 1)
    
    def evaluate(self, X, y_true):
        """
        Оценивает производительность суперансамбля.
        
        Args:
            X: Признаки для оценки
            y_true: Истинные метки
            
        Returns:
            Словарь с метриками производительности
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
        
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
            
        # Формируем словарь с результатами
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        logger.info(f"Оценка SuperEnsemble: accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}")
        return results
    
    def _optimize_genetic_weights(self, meta_features, y_val, population_size=30, generations=50, 
                             elite_ratio=0.2, mutation_prob=0.2, crossover_types='uniform'):
        """
        Оптимизация весов моделей ансамбля с помощью генетического алгоритма.
        
        Args:
            meta_features: Предсказания от всех моделей (shape: [n_samples, n_models])
            y_val: Истинные метки для валидационных данных
            population_size: Размер популяции для генетического алгоритма
            generations: Количество поколений для эволюции
            elite_ratio: Доля лучших особей, которые переходят в следующее поколение без изменений
            mutation_prob: Вероятность мутации каждого гена
            crossover_types: Тип кроссовера ('single', 'uniform' или 'both')
            
        Returns:
            np.ndarray: Оптимизированные веса для каждой модели или None в случае ошибки
        """
        logger.info(f"Запуск генетической оптимизации весов (популяция: {population_size}, поколения: {generations})")
        
        try:
            import random
            
            # Проверяем, что данные валидны
            if meta_features.shape[0] != len(y_val):
                logger.error(f"Несоответствие размеров данных: meta_features={meta_features.shape}, y_val={len(y_val)}")
                return None
                
            n_models = meta_features.shape[1]
            
            if n_models == 0:
                logger.error("Нет моделей для оптимизации весов")
                return None
                
            # Преобразуем целевые метки в numpy массив, если они еще не являются им
            y_val = np.asarray(y_val)
            
            # Функция фитнеса - точность взвешенного ансамбля
            def fitness(weights):
                # Проверка на нулевую сумму весов
                sum_weights = np.sum(weights)
                if sum_weights <= 0:
                    return 0.0
                    
                # Нормализация весов
                weights_normalized = weights / sum_weights
                
                # Взвешенные предсказания
                weighted_preds = np.dot(meta_features, weights_normalized)
                
                # Преобразуем в бинарные метки
                binary_preds = (weighted_preds > 0.5).astype(int)
                
                # Вычисляем и возвращаем точность
                return accuracy_score(y_val, binary_preds)
            
            # Инициализация популяции с разнообразными наборами весов
            population = []
            
            # Стратегия 1: Равные веса для всех моделей
            population.append(np.ones(n_models) / n_models)
            
            # Стратегия 2: Случайные веса
            for _ in range(population_size // 3):
                weights = np.random.uniform(0, 1, n_models)
                population.append(weights)
            
            # Стратегия 3: Доминирование отдельных моделей
            for _ in range(population_size // 3):
                weights = np.random.uniform(0, 0.1, n_models)
                dominant_idx = np.random.randint(0, n_models)
                weights[dominant_idx] = np.random.uniform(0.5, 1.0)
                population.append(weights)
            
            # Стратегия 4: Разреженные веса (часть моделей не используется)
            while len(population) < population_size:
                weights = np.zeros(n_models)
                active_models = np.random.choice(n_models, np.random.randint(1, n_models + 1), replace=False)
                weights[active_models] = np.random.uniform(0, 1, len(active_models))
                population.append(weights)
            
            # Генетический алгоритм
            best_fitness = 0
            best_weights = None
            stagnation_counter = 0  # Счетчик поколений без улучшения
            
            for gen in range(generations):
                # Оценка фитнеса
                fitness_scores = [fitness(weights) for weights in population]
                
                # Проверка лучшего результата
                current_best = max(fitness_scores)
                if current_best > best_fitness:
                    best_fitness = current_best
                    best_idx = fitness_scores.index(current_best)
                    best_weights = population[best_idx].copy()
                    logger.info(f"Поколение {gen + 1}/{generations}: Новый лучший результат = {best_fitness:.4f}")
                    stagnation_counter = 0  # Сбрасываем счетчик
                else:
                    stagnation_counter += 1
                
                # Ранний останов при стагнации более 10 поколений и хорошей точности
                if stagnation_counter > 10 and best_fitness > 0.95:
                    logger.info(f"Ранний останов на поколении {gen + 1} (стагнация, достигнута точность {best_fitness:.4f})")
                    break
                    
                # Выбор лучших особей (элитизм)
                elite_size = max(1, int(population_size * elite_ratio))
                elite_indices = np.argsort(fitness_scores)[-elite_size:]
                elite = [population[i] for i in elite_indices]
                
                # Создание нового поколения
                new_population = elite.copy()
                
                # Отбор родителей на основе фитнеса (турнирный отбор)
                def select_parent():
                    # Выбираем k особей случайно и возвращаем лучшую
                    k = 3  # Размер турнира
                    tournament_indices = np.random.choice(len(population), k, replace=False)
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                    return population[winner_idx]
                
                # Реализация различных типов кроссовера
                def single_point_crossover(parent1, parent2):
                    crossover_point = random.randint(1, n_models - 1)
                    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    return child
                    
                def uniform_crossover(parent1, parent2):
                    mask = np.random.randint(0, 2, n_models).astype(bool)
                    child = np.copy(parent1)
                    child[mask] = parent2[mask]
                    return child
                
                # Кроссовер и мутация
                while len(new_population) < population_size:
                    # Выбор родителей с помощью турнирного отбора
                    parent1 = select_parent()
                    parent2 = select_parent()
                    
                    # Кроссовер
                    if crossover_types == 'single':
                        child = single_point_crossover(parent1, parent2)
                    elif crossover_types == 'uniform':
                        child = uniform_crossover(parent1, parent2)
                    else:  # 'both' - чередуем типы кроссовера
                        if np.random.random() < 0.5:
                            child = single_point_crossover(parent1, parent2)
                        else:
                            child = uniform_crossover(parent1, parent2)
                    
                    # Мутация
                    for i in range(n_models):
                        if random.random() < mutation_prob:
                            # Различные стратегии мутации
                            mutation_type = random.choice(['scale', 'reset', 'small_nudge'])
                            
                            if mutation_type == 'scale':
                                # Умножение на случайный коэффициент
                                child[i] *= random.uniform(0.5, 1.5)
                            elif mutation_type == 'reset':
                                # Полный сброс значения
                                child[i] = random.uniform(0, 1)
                            else:  # 'small_nudge'
                                # Небольшое смещение
                                child[i] += random.uniform(-0.1, 0.1)
                                child[i] = max(0, child[i])  # Гарантируем неотрицательность
                    
                    # Добавляем потомка в новую популяцию
                    new_population.append(child)
                
                # Обновляем популяцию
                population = new_population
                
                # Если мы достигли идеальной точности, останавливаемся
                if best_fitness == 1.0:
                    logger.info("Достигнута идеальная точность 1.0, останавливаем оптимизацию")
                    break
                    
                # Адаптивная настройка вероятности мутации
                # Увеличиваем вероятность мутации при стагнации
                if stagnation_counter > 5:
                    mutation_prob = min(0.4, mutation_prob * 1.2)
                else:
                    mutation_prob = max(0.1, mutation_prob * 0.9)
            
            # Нормализация лучших весов
            if best_weights is not None:
                # Обработка краевого случая с нулевой суммой весов
                weight_sum = np.sum(best_weights)
                if weight_sum <= 0:
                    logger.warning("Сумма весов равна или меньше нуля, использование равных весов")
                    best_weights = np.ones(n_models) / n_models
                else:
                    best_weights = best_weights / weight_sum
                    
                self.optimal_weights = best_weights
                logger.info(f"Оптимизация завершена. Лучшая точность: {best_fitness:.4f}")
                
                # Анализ и вывод информации о найденных весах
                
                # Находим значимые модели (с весом > 1%)
                significant_indices = np.where(best_weights > 0.01)[0]
                significant_count = len(significant_indices)
                logger.info(f"Найдено {significant_count} значимых моделей (вес > 1%)")
                
                # Вывод весов для топ-5 моделей
                top_indices = np.argsort(best_weights)[-5:][::-1]  # Топ-5 в порядке убывания
                
                # Сначала считаем индексы для нейронных сетей
                nn_count = len(self.nn_ensemble.models)
                other_models_names = list(self.additional_models.keys())
                
                weights_info = []
                for idx in top_indices:
                    if idx < nn_count:
                        name = f"NeuralNet_{idx+1}"
                    else:
                        other_idx = idx - nn_count
                        if other_idx < len(other_models_names):
                            name = other_models_names[other_idx]
                        else:
                            name = f"Model_{idx}"
                    weights_info.append(f"{name}: {best_weights[idx]:.4f}")
                
                logger.info(f"Топ-5 моделей по весам: {', '.join(weights_info)}")
                
                # Проверяем, что результаты имеют смысл
                total_weight = sum(best_weights[idx] for idx in top_indices)
                logger.info(f"Топ-5 моделей содержат {total_weight:.2%} общего веса")
                
                # Проверяем, доминирует ли одна модель
                if best_weights[top_indices[0]] > 0.5:
                    logger.info(f"Обнаружена доминирующая модель: {weights_info[0]}")
                
                return best_weights
            else:
                logger.warning("Не удалось найти оптимальные веса")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при генетической оптимизации: {str(e)}")
            logger.exception("Подробности ошибки:")
            return None
    
def analyze_ensemble_size(dataset_name, max_ensemble_size=15, noise_levels=None):
    """
    Запускает анализ зависимости точности от размера ансамбля.
    
    Args:
        dataset_name: Название датасета
        max_ensemble_size: Максимальный размер ансамбля для исследования
        noise_levels: Список уровней шума
    
    Returns:
        Результаты анализа
    """
    logger.info(f"=== Запуск анализа размера ансамбля для {dataset_name} ===")
    start_time = time.time()
    
    # Загрузка данных
    X_train, y_train, X_test, y_test = DataLoader.load_heart_disease_data()
    
    # Разделение на тренировочную и валидационную выборки
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Установка базовых гиперпараметров
    base_params = {
        'n_layers': 3,
        'units_first': 64,
        'activation': 'relu',
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
    
    # Результаты для разных размеров ансамбля
    results = {
        'ensemble_sizes': list(range(1, max_ensemble_size + 1)),
        'dataset_name': dataset_name,
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
            ensemble = EnsembleClassifier(dataset_name, base_params=base_params, num_models=size)
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
        
        plt.title(f'Зависимость точности от размера ансамбля ({dataset_name})', fontsize=14)
        plt.xlabel('Размер ансамбля', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.grid(alpha=0.3)
        plt.xticks(sizes)
        
        plt.savefig(RESULTS_DIR / f"ensemble_size_impact_{dataset_name}.png", dpi=300)
        plt.close()
        
        # Для разных уровней шума
        plt.figure(figsize=(12, 8))
        
        for noise_level in noise_levels:
            noise_key = f"noise_{int(noise_level*100)}"
            if noise_key in results['accuracies']:
                sizes = [item['size'] for item in results['accuracies'][noise_key]]
                accuracies = [item['accuracy'] for item in results['accuracies'][noise_key]]
                
                plt.plot(sizes, accuracies, 'o-', label=f'Шум {int(noise_level*100)}%')
        
        plt.title(f'Влияние размера ансамбля на устойчивость к шуму ({dataset_name})', fontsize=14)
        plt.xlabel('Размер ансамбля', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.grid(alpha=0.3)
        plt.xticks(sizes)
        plt.legend()
        
        plt.savefig(RESULTS_DIR / f"ensemble_size_noise_impact_{dataset_name}.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Ошибка при визуализации результатов: {str(e)}")
    
    # Сохраняем результаты в JSON
    try:
        with open(RESULTS_DIR / f"ensemble_size_analysis_{dataset_name}.json", 'w') as f:
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

def analyze_superensemble_robustness(super_ensemble, X_test, y_test, dataset_name, 
                                    noise_types=['gaussian', 'uniform', 'impulse'], 
                                    noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Анализирует устойчивость суперансамбля к шуму по сравнению с компонентами.
    
    Args:
        super_ensemble: Обученный SuperEnsemble
        X_test: Тестовые данные
        y_test: Истинные метки
        dataset_name: Название датасета
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
        
        plt.title(f'Устойчивость к шуму {noise_type.capitalize()} для {dataset_name}', fontsize=14)
        plt.xlabel('Уровень шума', fontsize=12)
        plt.ylabel('Точность', fontsize=12)
        plt.ylim(0.5, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"superensemble_{noise_type}_robustness_{dataset_name}.png", dpi=300)
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
    
    plt.suptitle(f'Сравнение устойчивости к шуму для {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(RESULTS_DIR / f"superensemble_comparative_robustness_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Сохраняем результаты
    with open(RESULTS_DIR / f"superensemble_robustness_{dataset_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_heart_disease_classifier(run_hyperopt=True, n_trials=50, ensemble_size=5, 
                              analyze_noise=True, run_comparative=True,
                              min_noise=0.0, max_noise=0.5, noise_step=0.1,
                              ensemble_type='standard'):
    """
    Запускает полный процесс оптимизации, обучения и анализа классификатора сердечных заболеваний.
    
    Args:
        run_hyperopt: Выполнять ли оптимизацию гиперпараметров
        n_trials: Количество попыток для оптимизации гиперпараметров
        ensemble_size: Размер ансамбля моделей
        analyze_noise: Выполнять ли анализ устойчивости к шуму
        run_comparative: Выполнять ли сравнительный анализ с другими алгоритмами
        min_noise: Минимальный уровень шума (0.0 - 1.0)
        max_noise: Максимальный уровень шума (0.0 - 1.0)
        noise_step: Шаг изменения уровня шума (0.0 - 1.0)
        ensemble_type: Тип ансамбля ('standard', 'snapshot', 'stacked')
        
    Returns:
        EnsembleClassifier: Обученный ансамбль моделей
    """
    dataset_name = "heart"
    logger.info(f"=== Запуск классификатора сердечных заболеваний ===")
    start_time = time.time()
    
    # Загрузка и подготовка данных
    X_train, y_train, X_test, y_test = DataLoader.load_heart_disease_data()
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
            dataset_name, n_trials=n_trials
        )
    else:
        # Попытка загрузить ранее сохраненные результаты оптимизации
        try:
            result_path = RESULTS_DIR / f"hyperopt_results_{dataset_name}.pkl"
            if result_path.exists():
                with open(result_path, "rb") as f:
                    study = pickle.load(f)
                best_params = study.best_params
                logger.info(f"Загружены ранее найденные гиперпараметры: {best_params}")
            else:
                logger.warning("Файл с результатами оптимизации не найден, используем значения по умолчанию")
                # Значения по умолчанию для сердечных данных
                best_params = {
                    'n_layers': 3,
                    'units_first': 64,
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'dropout_rate': 0.3,
                    'use_residual': True,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'optimizer': 'adam',
                    'use_regularization': True,
                    'l1_factor': 1e-5,
                    'l2_factor': 1e-4
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
                'use_residual': True,
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'use_regularization': True,
                'l1_factor': 1e-5,
                'l2_factor': 1e-4
            }
    
    # Построение и обучение ансамбля моделей в зависимости от выбранного типа
    logger.info(f"Создание ансамбля из {ensemble_size} моделей, тип: {ensemble_type}")
    ensemble = EnsembleClassifier(dataset_name, base_params=best_params, num_models=ensemble_size)
    
    # Обучение ансамбля в зависимости от выбранного типа
    if ensemble_type == 'standard':
        ensemble.build_ensemble(X_train, y_train, X_val, y_val)
    elif ensemble_type == 'stacked':
        # Используем существующий метод build_stacked_ensemble
        ensemble.build_stacked_ensemble(X_train, y_train, X_val, y_val, 
                                     initial_ensemble_size=min(ensemble_size * 2, 20),
                                     final_ensemble_size=ensemble_size)
    elif ensemble_type == 'snapshot':
        # Предполагается, что метод build_snapshot_ensemble определен в EnsembleClassifier
        ensemble.build_snapshot_ensemble(X_train, y_train, X_val, y_val, n_snapshots=ensemble_size)
    
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
    
    # Добавим информацию о метриках классификации
    plt.subplot(2, 2, 3)
    report_text = ""
    if 'classification_report' in results:
        report = results['classification_report']
        for class_name, metrics in report.items():
            if class_name in ['0', '1'] or class_name in [0, 1]:
                report_text += f"Класс {class_name}:\n"
                report_text += f"   Precision: {metrics['precision']:.4f}\n"
                report_text += f"   Recall: {metrics['recall']:.4f}\n"
                report_text += f"   F1-score: {metrics['f1-score']:.4f}\n"
                report_text += f"   Support: {metrics['support']}\n\n"
        report_text += f"Точность: {results['accuracy']:.4f}\n"
        if 'roc_auc' in results and not np.isnan(results['roc_auc']):
            report_text += f"ROC AUC: {results['roc_auc']:.4f}\n"
    
    plt.text(0.1, 0.5, report_text, fontsize=10, va='center')
    plt.axis('off')
    plt.title('Метрики классификации')
    
    # Сохраняем визуализацию
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"results_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Сохраняем ансамбль моделей
    ensemble.save()
    
    # Сравнительный анализ с другими алгоритмами
    if run_comparative:
        logger.info("Сравнительный анализ с другими алгоритмами")
        comparative = ComparativeAnalyzer(ensemble, dataset_name)
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
                NoiseAnalyzer.visualize_noise_resistance(results, ensemble.dataset_name)
                
                # Сохраняем результаты
                with open(RESULTS_DIR / f"noise_analysis_{ensemble.dataset_name}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                    
                return results
            
            @staticmethod
            def visualize_noise_resistance(results, dataset_name):
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
                ax.set_title(f'Устойчивость к шуму для {dataset_name}', fontsize=16)
                ax.set_xlabel('Уровень шума', fontsize=14)
                ax.set_ylabel('Точность', fontsize=14)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(-0.02, max(results['noise_levels']) + 0.02)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12)
                
                # Сохраняем график
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / f"noise_resistance_{dataset_name}.png", dpi=300)
                plt.close()
        
        # Создаем массив уровней шума на основе параметров
        noise_levels = np.arange(min_noise, max_noise + noise_step/2, noise_step).tolist()

        # Анализируем устойчивость к шуму
        noise_analysis = NoiseAnalyzer.analyze_noise_resistance(
            ensemble, X_test, y_test,
            noise_types=['gaussian', 'uniform', 'impulse', 'missing'],
            noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
            n_experiments=3
        )
    
    # Выводим сводную информацию
    elapsed_time = time.time() - start_time
    logger.info(f"=== Результаты для {dataset_name} ===")
    logger.info(f"Точность: {results['accuracy']:.4f}")
    logger.info(f"ROC AUC: {results.get('roc_auc', 'N/A')}")
    
    # Получаем детальную информацию из отчета
    if 'classification_report' in results:
        for cls, metrics in results['classification_report'].items():
            if cls in ['0', '1']:
                logger.info(f"Класс {cls}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")
    
    logger.info(f"Время выполнения: {elapsed_time:.2f} секунд")
    
    return ensemble

def run_super_ensemble_heart(run_hyperopt=False, n_trials=50, nn_ensemble_size=5,
                          analyze_noise=True, run_comparative=True,
                          min_noise=0.0, max_noise=0.5, noise_step=0.1):
    """
    Запускает расширенный суперансамбль для задачи классификации сердечных заболеваний.
    
    Args:
        run_hyperopt: Выполнять ли оптимизацию гиперпараметров
        n_trials: Количество попыток для оптимизации гиперпараметров
        nn_ensemble_size: Размер ансамбля нейронных сетей
        analyze_noise: Выполнять ли анализ устойчивости к шуму
        run_comparative: Выполнять ли сравнительный анализ с другими алгоритмами
        min_noise: Минимальный уровень шума (0.0 - 1.0)
        max_noise: Максимальный уровень шума (0.0 - 1.0)
        noise_step: Шаг изменения уровня шума (0.0 - 1.0)
        
    Returns:
        SuperEnsemble: Обученный суперансамбль
    """
    dataset_name = "heart"
    logger.info(f"=== Запуск SuperEnsemble для {dataset_name} ===")
    start_time = time.time()
    
    # Загрузка и подготовка данных
    X_train, y_train, X_test, y_test = DataLoader.load_heart_disease_data()
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
            dataset_name, n_trials=n_trials
        )
    else:
        # Попытка загрузить ранее сохраненные результаты оптимизации
        try:
            result_path = RESULTS_DIR / f"hyperopt_results_{dataset_name}.pkl"
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
                    'use_residual': True,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'optimizer': 'adam',
                    'use_regularization': True,
                    'l1_factor': 1e-5,
                    'l2_factor': 1e-4
                }
        except Exception as e:
            logger.error(f"Ошибка при загрузке гиперпараметров: {str(e)}")
            best_params = {
                'n_layers': 3,
                'units_first': 64,
                'activation': 'relu',
                'use_batch_norm': True,
                'dropout_rate': 0.3,
                'use_residual': True,
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'use_regularization': True,
                'l1_factor': 1e-5,
                'l2_factor': 1e-4
            }
    
    # Создание и обучение SuperEnsemble
    super_ensemble = SuperEnsemble(
        dataset_name=dataset_name,
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
    plt.savefig(RESULTS_DIR / f"superensemble_metrics_{dataset_name}.png", dpi=300)
    plt.close()
    
    # Анализ устойчивости суперансамбля к шуму
    if analyze_noise:
        robustness_results = analyze_superensemble_robustness(
            super_ensemble, X_test, y_test, dataset_name,
            noise_types=['gaussian', 'uniform', 'impulse', 'missing'],
            noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )

        # Сохраняем результаты анализа устойчивости
        with open(RESULTS_DIR / f"superensemble_robustness_{dataset_name}.json", 'w') as f:
            json.dump(robustness_results, f, indent=2)

    # Сравнительный анализ с другими алгоритмами
    if run_comparative:
        logger.info("Сравнительный анализ суперансамбля с другими алгоритмами")
        comparative = ComparativeAnalyzer(super_ensemble, dataset_name)
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
    
    # Выводим сводную информацию
    elapsed_time = time.time() - start_time
    logger.info(f"=== Результаты SuperEnsemble для {dataset_name} ===")
    logger.info(f"Точность: {results['accuracy']:.4f}")
    logger.info(f"ROC AUC: {results.get('roc_auc', 'N/A')}")
    
    # Получаем детальную информацию из отчета
    if 'classification_report' in results:
        for cls, metrics in results['classification_report'].items():
            if cls in ['0', '1']:
                logger.info(f"Класс {cls}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")
    
    logger.info(f"Время выполнения: {elapsed_time:.2f} секунд")
    
    return super_ensemble

if __name__ == "__main__":
    print("=== Классификатор заболеваний сердца с использованием ансамбля нейронных сетей ===")
    
    # Запрос параметров запуска
    run_hyperopt = input("Выполнить оптимизацию гиперпараметров? (y/n, по умолчанию: n): ").lower() == 'y'
    
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
    
    # Если выбран базовый ансамбль, запрашиваем тип построения
    if ensemble_type == "1":
        build_type = input("\nВыберите метод построения ансамбля:\n1. Стандартный\n2. Стекинг (с отбором моделей)\n3. Снапшоты (разные этапы обучения)\nВыбор (1/2/3, по умолчанию: 1): ") or "1"
        if build_type == "1":
            ensemble_build_type = "standard"
        elif build_type == "2":
            ensemble_build_type = "stacked"
        else:
            ensemble_build_type = "snapshot"
    
    # Запуск выбранного типа ансамбля
    try:
        if ensemble_type == "1":
            # Базовый ансамбль
            print(f"\nЗапуск ансамбля нейронных сетей для классификации заболеваний сердца...")
            ensemble = run_heart_disease_classifier(
                run_hyperopt=run_hyperopt, 
                n_trials=n_trials, 
                ensemble_size=ensemble_size,
                analyze_noise=analyze_noise,
                run_comparative=run_comparative,
                min_noise=min_noise,
                max_noise=max_noise,
                noise_step=noise_step,
                ensemble_type=ensemble_build_type
            )
        else:
            # SuperEnsemble
            print(f"\nЗапуск SuperEnsemble для классификации заболеваний сердца...")
            super_ensemble = run_super_ensemble_heart(
                run_hyperopt=run_hyperopt,
                n_trials=n_trials,
                nn_ensemble_size=ensemble_size,
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