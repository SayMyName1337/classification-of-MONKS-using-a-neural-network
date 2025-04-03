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
    
    def _optimize_genetic_weights(self, meta_features, y_val, population_size=30, generations=50):
        """Оптимизация весов с помощью генетического алгоритма."""
        logger.info("Запуск генетической оптимизации весов")
        
        try:
            import random
            
            n_models = meta_features.shape[1]
            
            # Функция фитнеса - точность взвешенного ансамбля
            def fitness(weights):
                weights_normalized = weights / np.sum(weights)  # Нормализация весов
                weighted_preds = np.dot(meta_features, weights_normalized)
                binary_preds = (weighted_preds > 0.5).astype(int)
                return accuracy_score(y_val, binary_preds)
            
            # Инициализация популяции
            population = []
            for _ in range(population_size):
                weights = np.random.uniform(0, 1, n_models)
                population.append(weights)
            
            # Генетический алгоритм
            best_fitness = 0
            best_weights = None
            
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
                
                # Выбор лучших особей
                elite_size = max(1, int(population_size * 0.2))
                elite_indices = np.argsort(fitness_scores)[-elite_size:]
                elite = [population[i] for i in elite_indices]
                
                # Создание нового поколения
                new_population = elite.copy()
                
                # Кроссовер и мутация
                while len(new_population) < population_size:
                    # Выбор родителей
                    parent1, parent2 = random.sample(elite, 2)
                    
                    # Кроссовер
                    crossover_point = random.randint(1, n_models-1)
                    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    
                    # Мутация (с небольшой вероятностью)
                    if random.random() < 0.2:
                        mutation_point = random.randint(0, n_models-1)
                        child[mutation_point] *= random.uniform(0.8, 1.2)
                    
                    new_population.append(child)
                
                population = new_population
            
            # Нормализация лучших весов
            if best_weights is not None:
                best_weights = best_weights / np.sum(best_weights)
                self.optimal_weights = best_weights
                logger.info(f"Оптимизация завершена. Лучшая точность: {best_fitness:.4f}")
                
                # Вывод весов для ключевых моделей
                top_indices = np.argsort(best_weights)[-5:]  # Топ-5 моделей
                
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
                
                return best_weights
            else:
                logger.warning("Не удалось найти оптимальные веса")
                return None