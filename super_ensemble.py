import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Импортируем ваш существующий класс ансамбля
from Monks_v2 import EnsembleClassifier, DataLoader

logger = logging.getLogger("SuperEnsemble")

class SuperEnsemble:
    """Расширенный ансамбль с улучшенной производительностью и устойчивостью."""
    
    def __init__(self, problem_number, base_params=None, nn_ensemble_size=5):
        """
        Инициализирует суперансамбль.
        
        Args:
            problem_number: Номер задачи MONK
            base_params: Базовые параметры для нейронных сетей
            nn_ensemble_size: Размер ансамбля нейронных сетей
        """
        self.problem_number = problem_number
        self.base_params = base_params
        self.nn_ensemble_size = nn_ensemble_size
        self.nn_ensemble = None
        self.additional_models = {}  # Словарь для дополнительных алгоритмов
        self.meta_model = None
        self.optimal_weights = None
        
        logger.info(f"Инициализирован SuperEnsemble для MONK-{problem_number}")
    
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
            self.problem_number, 
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
            
            # Оптимизируем веса (простой вариант)
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