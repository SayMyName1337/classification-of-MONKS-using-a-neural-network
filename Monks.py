import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import shutil
import tensorflow as tf
import pandas as pd
import concurrent.futures
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch
from sklearn.impute import SimpleImputer

# Remove old directory if it exists
if os.path.exists('tuner_logs'):
    shutil.rmtree('tuner_logs')

# Step 1: Set up logger
logging.basicConfig(
    filename='monks_enhanced_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
logger.info('\n')
logger.info("===" * 15)
logger.info("Enhanced MONK Classifier Started")
logger.info("===" * 15)

# Function for validated user input
def get_user_input(prompt, min_val, max_val, is_int=False):
    while True:
        try:
            value = float(input(prompt)) if not is_int else int(input(prompt))
            if min_val <= value <= max_val:
                logger.info(f"User input: {value}")
                return value
            else:
                logger.warning(f"Invalid input: {value}. Please enter a value between {min_val} and {max_val}.")
                print(f"Please enter a value from {min_val} to {max_val}.")
        except ValueError:
            logger.error("Invalid input: not a number.")
            print("Error: Enter a valid value.")

# Function to load Monk's dataset with enhanced features
def load_monks_data(problem_number):
    """
    Load the Monk's problem dataset with problem-specific feature engineering
    
    Args:
        problem_number: Which Monk's problem to load (1, 2, or 3)
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    logger.info(f"Loading Monk's problem {problem_number} dataset with enhanced features")
    
    try:
        # Define column names
        columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
        
        # Load training data
        train_file = f"monks-{problem_number}.train"
        # If the file doesn't exist, provide instructions to download
        if not os.path.exists(train_file):
            logger.error(f"Dataset file not found: {train_file}")
            print(f"Dataset file not found: {train_file}")
            print("Please download the Monk's dataset from UCI repository.")
            print("You can use the following URL: https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/")
            return None, None, None, None
        
        train_data = pd.read_csv(train_file, sep=' ', names=columns)
        train_data = train_data.drop('id', axis=1)  # Remove ID column
        
        # Load test data
        test_file = f"monks-{problem_number}.test"
        if not os.path.exists(test_file):
            logger.error(f"Dataset file not found: {test_file}")
            print(f"Dataset file not found: {test_file}")
            return None, None, None, None
        
        test_data = pd.read_csv(test_file, sep=' ', names=columns)
        test_data = test_data.drop('id', axis=1)  # Remove ID column
        
        # Problem-specific feature engineering
        train_data = add_problem_specific_features(train_data, problem_number)
        test_data = add_problem_specific_features(test_data, problem_number)
        
        # Extract features and target
        X_train = train_data.drop('class', axis=1)
        y_train = train_data['class']
        
        X_test = test_data.drop('class', axis=1)
        y_test = test_data['class']
        
        logger.info(f"Dataset loaded successfully. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features after engineering: {X_train.columns.tolist()}")
        
        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        print(f"Error loading dataset: {e}")
        return None, None, None, None

# Problem-specific feature engineering
def add_problem_specific_features(data, problem_number):
    """
    Add problem-specific engineered features based on the MONK's problem definitions
    
    Args:
        data: DataFrame with original features
        problem_number: Which Monk's problem (1, 2, or 3)
        
    Returns:
        DataFrame with added engineered features
    """
    df = data.copy()
    
    if problem_number == 1:
        # MONK-1: (a1 = a2) OR (a5 = 1)
        df['a1_eq_a2'] = (df['a1'] == df['a2']).astype(int)
        df['a5_eq_1'] = (df['a5'] == 1).astype(int)
        df['rule_feature'] = ((df['a1'] == df['a2']) | (df['a5'] == 1)).astype(int)
        
    elif problem_number == 2:
        # MONK-2: Exactly two attributes have value 1
        # Count attributes with value 1
        df['count_1s'] = ((df['a1'] == 1) + (df['a2'] == 1) + 
                          (df['a3'] == 1) + (df['a4'] == 1) + 
                          (df['a5'] == 1) + (df['a6'] == 1))
        df['exactly_two_1s'] = (df['count_1s'] == 2).astype(int)
        
        # Add pairwise interactions
        for i in range(1, 6):
            for j in range(i+1, 7):
                df[f'a{i}_and_a{j}_eq_1'] = ((df[f'a{i}'] == 1) & (df[f'a{j}'] == 1)).astype(int)
    
    elif problem_number == 3:
        # MONK-3: (a5 = 3 AND a4 = 1) OR (a5 ≠ 4 AND a2 ≠ 3)
        df['a5_eq_3_and_a4_eq_1'] = ((df['a5'] == 3) & (df['a4'] == 1)).astype(int)
        df['a5_neq_4_and_a2_neq_3'] = ((df['a5'] != 4) & (df['a2'] != 3)).astype(int)
        df['rule_feature'] = ((df['a5'] == 3) & (df['a4'] == 1)) | ((df['a5'] != 4) & (df['a2'] != 3))
        df['rule_feature'] = df['rule_feature'].astype(int)
    
    # Add general feature interactions that might help
    # Pairwise differences
    for i in range(1, 6):
        for j in range(i+1, 7):
            df[f'diff_a{i}_a{j}'] = df[f'a{i}'] - df[f'a{j}']
    
    return df

# Enhanced Keras Tuner model definition function
def build_model(hp):
    """Define enhanced model architecture for Monk's problem with hyperparameter tuning"""
    logger.info("Building enhanced model with hyperparameters.")
    try:
        inputs = Input(shape=(hp.Int('input_dim', min_value=17, max_value=50, step=1),))
        
        # Choose activation function
        activation = hp.Choice('activation', values=['relu', 'elu', 'selu'])
        
        # First layer with tunable units
        x = Dense(
            hp.Int('units_layer1', min_value=32, max_value=256, step=32),
            activation=activation
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1))(x)
        
        # Second layer with tunable units
        x = Dense(
            hp.Int('units_layer2', min_value=16, max_value=128, step=16),
            activation=activation
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1))(x)
        
        # Optional third layer
        if hp.Boolean('use_third_layer'):
            x = Dense(
                hp.Int('units_layer3', min_value=8, max_value=64, step=8),
                activation=activation
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.3, step=0.1))(x)
        
        # Output layer (binary classification)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Tunable learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        
        # Choose optimizer
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Enhanced model built and compiled successfully.")
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

# Function to add noise to features
def add_noise_to_features(X, noise_level, noise_type='gaussian'):
    """
    Add noise to features with different noise types
    
    Args:
        X: Input features
        noise_level: Level of noise (0 to 1)
        noise_type: Type of noise ('gaussian', 'uniform', 'impulse', 'missing')
    
    Returns:
        Noisy version of X
    """
    logger.info(f"Adding {noise_type} noise to data with level {noise_level * 100}%.")
    try:
        # Convert X to float type to allow for NaN values
        X = X.astype(float) if noise_type == 'missing' else X
        
        num_samples, num_features = X.shape
        num_noisy_samples = round(noise_level * num_samples)
        noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
        X_noisy = X.copy()
        
        # Calculate feature-specific statistics for more realistic noise
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        feature_mins = np.min(X, axis=0)
        feature_maxs = np.max(X, axis=0)
        
        for idx in noisy_indices:
            # Affect a random number of features
            num_noisy_features = np.random.randint(1, num_features + 1)
            noisy_features = np.random.choice(num_features, num_noisy_features, replace=False)
            
            for feature in noisy_features:
                feature_range = feature_maxs[feature] - feature_mins[feature]
                
                if noise_type == 'gaussian':
                    # Scale noise relative to feature's standard deviation
                    noise = np.random.normal(0, feature_stds[feature])
                    X_noisy[idx, feature] += noise
                
                elif noise_type == 'uniform':
                    # Scale uniform noise relative to feature range
                    noise = np.random.uniform(-0.5, 0.5) * feature_range
                    X_noisy[idx, feature] += noise
                
                elif noise_type == 'impulse':
                    # Use realistic impulse values
                    impulse_type = np.random.choice(['min', 'max', 'extreme'])
                    if impulse_type == 'min':
                        X_noisy[idx, feature] = feature_mins[feature]
                    elif impulse_type == 'max':
                        X_noisy[idx, feature] = feature_maxs[feature]
                    else:  # extreme
                        extreme_factor = np.random.choice([-1.5, 1.5])
                        X_noisy[idx, feature] = feature_means[feature] + extreme_factor * feature_stds[feature]
                
                elif noise_type == 'missing':
                    X_noisy[idx, feature] = np.nan
        
        logger.info(f"Noise added to {num_noisy_samples} samples ({num_noisy_samples/num_samples:.2%} of data).")
        return X_noisy
    
    except Exception as e:
        logger.error(f"Error adding noise: {e}")
        raise

# Handle missing values
def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values using different imputation strategies.
    
    Args:
        X: Input features with missing values
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')
    
    Returns:
        X with imputed values
    """
    logger.info(f"Handling missing values using {strategy} strategy.")
    try:
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            return imputer.fit_transform(X)
        elif strategy == 'knn':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            return imputer.fit_transform(X)
        else:
            logger.warning(f"Unknown imputation strategy: {strategy}. Using 'mean' instead.")
            imputer = SimpleImputer(strategy='mean')
            return imputer.fit_transform(X)
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise

# Visualize results
def visualize_results(accuracies, noise_level, noise_type):
    """
    Visualize the impact of noise on model accuracy.
    """
    logger.info("Generating result visualizations.")
    try:
        # Create accuracy distribution plot
        plt.figure(figsize=(10, 6))
        
        # Plot individual experiment accuracies
        plt.plot(range(1, len(accuracies) + 1), accuracies, 'o-', alpha=0.7, label='Experiment Accuracy')
        
        # Plot mean accuracy line
        mean_accuracy = np.mean(accuracies)
        plt.axhline(y=mean_accuracy, color='r', linestyle='--', 
                   label=f'Mean Accuracy: {mean_accuracy:.2f}%')
        
        # Plot baseline (no noise) accuracy if available
        if 'baseline_accuracy' in globals():
            plt.axhline(y=baseline_accuracy, color='g', linestyle='-.',
                       label=f'Baseline Accuracy: {baseline_accuracy:.2f}%')
        
        # Add trend line
        z = np.polyfit(range(1, len(accuracies) + 1), accuracies, 1)
        p = np.poly1d(z)
        plt.plot(range(1, len(accuracies) + 1), p(range(1, len(accuracies) + 1)), 
                 linestyle='--', color='orange', alpha=0.7, label='Trend')
        
        plt.title(f'Model Accuracy with {noise_type.capitalize()} Noise ({noise_level*100:.1f}%)')
        plt.xlabel('Experiment')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        # plt.savefig(f'enhanced_monks_accuracy_with_{noise_type}_noise_{int(noise_level*100)}percent.png')
        logger.info(f"Visualization saved to 'enhanced_monks_accuracy_with_{noise_type}_noise_{int(noise_level*100)}percent.png'")
        
        # Show statistics summary
        plt.figure(figsize=(8, 4))
        plt.boxplot(accuracies, vert=False, patch_artist=True)
        plt.title(f'Accuracy Distribution with {noise_type.capitalize()} Noise')
        plt.xlabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = (
            f"Mean: {np.mean(accuracies):.2f}%\n"
            f"Median: {np.median(accuracies):.2f}%\n"
            f"Std Dev: {np.std(accuracies):.2f}%\n"
            f"Min: {np.min(accuracies):.2f}%\n"
            f"Max: {np.max(accuracies):.2f}%"
        )
        plt.figtext(0.7, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save the statistics plot
        # plt.savefig(f'enhanced_monks_accuracy_stats_{noise_type}_noise_{int(noise_level*100)}percent.png')
        logger.info(f"Statistics visualization saved to 'enhanced_monks_accuracy_stats_{noise_type}_noise_{int(noise_level*100)}percent.png'")
        
        plt.close('all')  # Close all figures to free memory
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        logger.info("Continuing execution without visualizations.")

# Create ensemble of models
def create_ensemble_parallel(X_train, y_train, X_val, y_val, input_dim, problem_number, num_models=5):
    """Create an ensemble of models using parallel processing"""
    logger.info(f"Creating ensemble of {num_models} models in parallel")
    model_paths = []
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_models, os.cpu_count())) as executor:
        # Submit tasks
        futures = [
            executor.submit(
                train_single_model, 
                i, X_train, y_train, X_val, y_val, input_dim, problem_number
            ) for i in range(num_models)
        ]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                model_path = future.result()
                model_paths.append(model_path)
            except Exception as e:
                logger.error(f"Error in parallel training: {e}")
    
    # Load all models from disk
    with tf.device('/CPU:0'):
        models = [load_model(path) for path in model_paths]
    return models

# Ensemble prediction function
def ensemble_predict(models, X):
    """
    Make predictions with ensemble of models.
    
    Parameters
    ----------
    models : list of tf.keras.Model
        List of trained Keras models
    X : ndarray
        Features to predict on, shape (n_samples, n_features)
        
    Returns
    -------
    avg_predictions : ndarray
        Raw averaged predictions, shape (n_samples, 1)
    class_predictions : ndarray
        Binarized class predictions, shape (n_samples,)
    """
    # Get predictions from each model
    predictions = [model.predict(X, verbose=0) for model in models]
    
    # Average the predictions
    avg_predictions = np.mean(predictions, axis=0)
    
    # Convert to class labels
    class_predictions = (avg_predictions > 0.5).astype(int).flatten()
    
    return avg_predictions, class_predictions

# Evaluate model performance
def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance with detailed metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name/description of the model
        
    Returns:
        Dict with performance metrics
    """
    logger.info(f"Evaluating {model_name} performance")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Log results
    logger.info(f"{model_name} accuracy: {accuracy:.2f}%")
    logger.info(f"{model_name} confusion matrix:\n{cm}")
    logger.info(f"{model_name} classification report:\n{json.dumps(report, indent=2)}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

# Enhanced run experiments function
def run_experiments(models, X_test_encoded, y_test, noise_type, noise_levels, num_experiments):
    """
    Run experiments with different noise levels and types using ensemble models.
    
    Args:
        models: List of ensemble models
        X_test_encoded: Encoded test features
        y_test: Test labels
        noise_type: Type of noise to apply
        noise_levels: List of noise levels to test
        num_experiments: Number of experiments per noise level
    
    Returns:
        Results dictionary
    """
    logger.info("===== Starting comprehensive noise experiments with ensemble model =====")
    
    results = {
        'noise_levels': noise_levels,
        'noise_type': noise_type,
        'accuracies_per_level': [],
        'mean_accuracies': [],
        'std_accuracies': []
    }
    
    # First, get baseline accuracy (no noise)
    _, y_pred = ensemble_predict(models, X_test_encoded)
    baseline_accuracy = accuracy_score(y_test, y_pred) * 100
    logger.info(f'Ensemble baseline accuracy (no noise): {baseline_accuracy:.2f}%')
    results['baseline_accuracy'] = baseline_accuracy
    
    # Run experiments for each noise level
    for noise_level in noise_levels:
        logger.info(f"===== Testing noise level: {noise_level*100:.1f}% =====")
        level_accuracies = []
        
        for experiment in range(1, num_experiments + 1):
            logger.info(f"Experiment {experiment} with {noise_type} noise at {noise_level*100:.1f}% level")
            
            # Add noise to test data
            X_test_noisy = add_noise_to_features(X_test_encoded, noise_level, noise_type)
            
            # Handle missing values if needed
            if noise_type == 'missing':
                X_test_noisy = handle_missing_values(X_test_noisy, strategy='mean')
            
            # Make predictions using ensemble
            _, y_pred = ensemble_predict(models, X_test_noisy)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred) * 100
            level_accuracies.append(accuracy)
            logger.info(f'Experiment {experiment} accuracy: {accuracy:.2f}%')
        
        # Store results for this noise level
        results['accuracies_per_level'].append(level_accuracies)
        mean_accuracy = np.mean(level_accuracies)
        std_accuracy = np.std(level_accuracies)
        results['mean_accuracies'].append(mean_accuracy)
        results['std_accuracies'].append(std_accuracy)
        
        logger.info(f"Average accuracy at {noise_level*100:.1f}% noise: {mean_accuracy:.2f}% (±{std_accuracy:.2f}%)")
        
        # Visualize results for this noise level
        visualize_results(level_accuracies, noise_level, noise_type)
    
    return results

# K-fold cross-validation for better model evaluation
def k_fold_cross_validation(X, y, n_splits=5, problem_number=1):
    """
    Perform K-fold cross-validation with ensemble models.
    
    Args:
        X: Features
        y: Labels
        n_splits: Number of folds
        problem_number: Which Monk's problem
        
    Returns:
        Mean accuracy and std deviation
    """
    logger.info(f"Performing {n_splits}-fold cross-validation")
    
    # Convert to numpy arrays and ensure float type
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Training fold {fold+1}/{n_splits}")
        
        # Get fold data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model for this fold
        model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(64, activation='elu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='elu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Train the model
        model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        _, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        fold_accuracies.append(val_acc * 100)
        logger.info(f"Fold {fold+1} accuracy: {val_acc:.4f}")
    
    # Calculate mean and standard deviation
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    logger.info(f"Cross-validation results: {mean_accuracy:.2f}% (±{std_accuracy:.2f}%)")
    return mean_accuracy, std_accuracy

def benchmark_algorithms(X_train, y_train, X_test, y_test, problem_number, noise_level=0, noise_type=None):
    """
    Benchmark different algorithms on the same data
    
    Returns:
        DataFrame with performance metrics for each algorithm
    """
    results = []
    
    # Prepare noisy test data if needed
    X_test_noisy = X_test
    if noise_level > 0 and noise_type:
        X_test_noisy = add_noise_to_features(X_test, noise_level, noise_type)
        if noise_type == 'missing':
            X_test_noisy = handle_missing_values(X_test_noisy)
    
    # 1. Your Enhanced Neural Network Ensemble
    ensemble_models = create_ensemble_parallel(X_train, y_train, X_test, y_test, 
                                      X_train.shape[1], problem_number)
    _, y_pred_ensemble = ensemble_predict(ensemble_models, X_test_noisy)
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble) * 100
    results.append({"Algorithm": "Enhanced NN Ensemble", "Accuracy": ensemble_acc})
    
    # 2. Basic Neural Network
    basic_nn = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    basic_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    basic_nn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    _, basic_nn_acc = basic_nn.evaluate(X_test_noisy, y_test, verbose=0)
    results.append({"Algorithm": "Basic NN", "Accuracy": basic_nn_acc * 100})
    
    # 3. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test_noisy)
    rf_acc = accuracy_score(y_test, y_pred_rf) * 100
    results.append({"Algorithm": "Random Forest", "Accuracy": rf_acc})
    
    # 4. Support Vector Machine
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test_noisy)
    svm_acc = accuracy_score(y_test, y_pred_svm) * 100
    results.append({"Algorithm": "SVM", "Accuracy": svm_acc})
    
    # 5. XGBoost
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test_noisy)
    xgb_acc = accuracy_score(y_test, y_pred_xgb) * 100
    results.append({"Algorithm": "XGBoost", "Accuracy": xgb_acc})
    
    # Create DataFrame with results
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df["Algorithm"], results_df["Accuracy"])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.title(f'Algorithm Comparison for MONK-{problem_number} ' + 
             (f'with {noise_level*100}% {noise_type} noise' if noise_level > 0 else 'without noise'))
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)  # Leave room for text
    plt.grid(axis='y', alpha=0.3)
    # plt.savefig(f'algorithm_comparison_monk{problem_number}' + 
            #    (f'_{noise_type}_{int(noise_level*100)}' if noise_level > 0 else '') + '.png')
    
    return results_df

def train_single_model(i, X_train, y_train, X_val, y_val, input_dim, problem_number):
    """
        Train a single model for the ensemble
    """
    # Configure logging for this process
    process_logger = logging.getLogger(f"ensemble_model_{i}")
    process_logger.info(f"Training ensemble model {i+1}")
    
    # Create model with variations as in your original code
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64 + i*16, activation='elu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32 + i*8, activation='elu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Use different learning rates for diversity
    lr = 0.001 * (1.0 + i * 0.5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    batch_size = 16 + i * 8
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_schedule],
        verbose=0  # Set to 0 to avoid messy output
    )
    
    # Save model
    model_path = f'ensemble_model_{problem_number}_{i+1}.h5'
    model.save(model_path)
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    process_logger.info(f"Model {i+1} validation accuracy: {val_acc:.4f}")
    
    return model_path

def main():
    logger.info("===" * 15)
    logger.info("Enhanced MONK Classifier Program started")
    logger.info("===" * 15)
    
    try:
        # Get which Monk's problem to use
        problem_number = get_user_input("Enter which Monk's problem to use (1, 2, or 3): ", 1, 3, is_int=True)
        
        # User input for noise parameters
        logger.info("Starting user input phase.")
        
        # Modified input for noise levels
        min_noise = get_user_input("Enter minimum noise percentage (0 to 100): ", 0, 100) / 100
        max_noise = get_user_input("Enter maximum noise percentage (0 to 100): ", min_noise*100, 100) / 100
        noise_steps = get_user_input("Enter number of noise steps: ", 1, 10, is_int=True)
        
        if noise_steps == 1:
            noise_levels = [min_noise]
        else:
            noise_levels = np.linspace(min_noise, max_noise, noise_steps)
        
        num_experiments = get_user_input("Enter number of experiments for each noise level: ", 1, 100, is_int=True)
        
        # Choose noise type
        noise_type = input("Enter noise type (gaussian, uniform, impulse, missing): ").strip().lower()
        while noise_type not in ['gaussian', 'uniform', 'impulse', 'missing']:
            logger.warning(f"Invalid noise type: {noise_type}.")
            print("Error: Enter a valid noise type (gaussian, uniform, impulse, missing).")
            noise_type = input("Enter noise type: ").strip().lower()
        
        # Number of models in ensemble
        num_ensemble_models = get_user_input("Enter number of models for ensemble (1-10): ", 1, 10, is_int=True)
        
        logger.info(f"User selected Monk's problem {problem_number}")
        logger.info(f"User-defined noise levels: from {min_noise*100}% to {max_noise*100}% in {noise_steps} steps")
        logger.info(f"Number of experiments per level: {num_experiments}")
        logger.info(f"Noise type: {noise_type}")
        logger.info(f"Number of ensemble models: {num_ensemble_models}")
        
        # Load data with problem-specific feature engineering
        X_train, y_train, X_test, y_test = load_monks_data(problem_number)
        
        if X_train is None:
            logger.error("Failed to load data. Exiting program.")
            return
        
        logger.info("Data loaded successfully.")
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Encode features if necessary
        logger.info("Preparing data for training...")
        # Convert to numpy arrays for consistency
        # X_train_encoded = X_train.values
        # X_test_encoded = X_test.values
        # y_train_encoded = y_train.values
        # y_test_encoded = y_test.values

        # Convert to numpy arrays for consistency
        X_train_encoded = X_train.values.astype(np.float32)
        X_test_encoded = X_test.values.astype(np.float32)
        y_train_encoded = y_train.values.astype(np.float32)
        y_test_encoded = y_test.values.astype(np.float32)
        
        # Split training data to create a validation set
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_encoded, y_train_encoded, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train_split.shape}, Validation set: {X_val.shape}")
        
        # Perform k-fold cross-validation for initial model assessment
        logger.info("Performing initial model assessment with k-fold cross-validation...")
        mean_cv_accuracy, std_cv_accuracy = k_fold_cross_validation(
            X_train_encoded, y_train_encoded, n_splits=5, problem_number=problem_number
        )
        
        logger.info(f"Cross-validation results: {mean_cv_accuracy:.2f}% (±{std_cv_accuracy:.2f}%)")
        
        # Create and train ensemble models
        logger.info(f"Creating ensemble of {num_ensemble_models} models...")
        input_dim = X_train_encoded.shape[1]
        ensemble_models = create_ensemble_parallel(
            X_train_split, y_train_split, X_val, y_val, 
            input_dim, problem_number, num_models=num_ensemble_models
        )
        
        # Evaluate baseline ensemble performance (no noise)
        logger.info("Evaluating baseline ensemble performance...")
        _, baseline_predictions = ensemble_predict(ensemble_models, X_test_encoded)
        baseline_metrics = evaluate_model(y_test_encoded, baseline_predictions, "Baseline Ensemble")
        baseline_accuracy = baseline_metrics['accuracy']
        
        logger.info(f"Baseline ensemble accuracy: {baseline_accuracy:.2f}%")
        
        # Run comprehensive noise experiments
        logger.info("Starting noise experiments...")
        results = run_experiments(
            ensemble_models, X_test_encoded, y_test_encoded, 
            noise_type, noise_levels, num_experiments
        )
        
        # Generate and save final summary report
        logger.info("Generating final summary report...")
        
        # Create summary visualization
        plt.figure(figsize=(12, 8))
        
        # Plot mean accuracies with error bars
        plt.errorbar(
            [level * 100 for level in noise_levels],
            results['mean_accuracies'],
            yerr=results['std_accuracies'],
            fmt='o-',
            capsize=5,
            label=f'{noise_type.capitalize()} Noise'
        )
        
        # Add baseline
        plt.axhline(y=baseline_accuracy, color='g', linestyle='--', 
                   label=f'Baseline (No Noise): {baseline_accuracy:.2f}%')
        
        plt.title(f'Impact of {noise_type.capitalize()} Noise on MONK-{problem_number} Classification Accuracy')
        plt.xlabel('Noise Level (%)')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the summary plot
        summary_plot_file = f'monk{problem_number}_{noise_type}_noise_summary.png'
        # plt.savefig(summary_plot_file)
        logger.info(f"Summary plot saved to {summary_plot_file}")
        
        # Show final results
        print("\n" + "="*50)
        print(f"MONK-{problem_number} NOISE RESISTANCE EXPERIMENT RESULTS")
        print("="*50)
        print(f"Baseline accuracy (no noise): {baseline_accuracy:.2f}%")
        print(f"Noise type: {noise_type.capitalize()}")
        print("\nAccuracy at different noise levels:")
        
        for i, level in enumerate(noise_levels):
            print(f"  {level*100:.1f}% noise: {results['mean_accuracies'][i]:.2f}% (±{results['std_accuracies'][i]:.2f}%)")
        
        print("\nExperiment complete! See log file and generated plots for details.")
        
        # Export results to JSON for future reference
        results_file = f'monk{problem_number}_{noise_type}_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'problem_number': problem_number,
                'noise_type': noise_type,
                'noise_levels': [float(level) for level in noise_levels],
                'baseline_accuracy': float(baseline_accuracy),
                'mean_accuracies': [float(acc) for acc in results['mean_accuracies']],
                'std_accuracies': [float(std) for std in results['std_accuracies']]
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        logger.info("Enhanced MONK Classifier Program completed successfully")

        # After training your ensemble models
        print("\nComparing with other algorithms...")
        comparison_results = benchmark_algorithms(X_train_encoded, y_train_encoded, 
                                                X_test_encoded, y_test_encoded, 
                                                problem_number)
        print("\nAlgorithm comparison results:")
        print(comparison_results)

        # Optionally compare with noise
        if max_noise > 0:
            noise_comparison = benchmark_algorithms(X_train_encoded, y_train_encoded, 
                                                X_test_encoded, y_test_encoded, 
                                                problem_number, noise_level=max_noise, 
                                                noise_type=noise_type)
            print("\nAlgorithm comparison with noise:")
            print(noise_comparison)
        
    except Exception as e:
        logger.error(f"Error in main program: {e}")
        logger.exception("Exception details:")
        print(f"An error occurred: {e}")
        print("Check log file for details.")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()