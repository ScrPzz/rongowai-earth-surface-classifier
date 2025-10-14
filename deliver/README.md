# Technical Documentation - GNSS-R DDM Classification Pipeline

## Architecture Overview

Modular pipeline for binary classification of Delay Doppler Maps (DDM) from GNSS-R satellite data. The architecture supports two approaches: processing on raw data (200D) and processing on compressed data via autoencoder (20D). The system performs ETL from NetCDF files, feature engineering, model training (XGBoost/TabNet), probabilistic calibration, and stratified K-fold validation. The dataflow proceeds from preprocessing/quality filtering → normalization/compression → feature extraction → model training → calibration → comprehensive testing.

### Global Technical Specifications

**Data Formats:**
- Input: NetCDF4 (.nc) with raw_counts variables (n_time × n_samples × 5 × 40)
- Intermediate: Parquet with ZSTD compression
- Models: joblib/pickle for serialization
- Scalers: joblib for MinMaxScaler/StandardScaler

**Dataset Dimensions:**
- Training: 10M balanced samples (5M class 0, 5M class 1)
- Test: 2M balanced samples (1M class 0, 1M class 1)
- Raw features: 50+ dimensions post-extraction
- Compressed: 20 dimensions post-encoder

**Quality Filtering Criteria:**
- SNR (Signal-to-Noise Ratio): > 0 dB (optional)
- Copol gain: ≥ 5 dB
- Xpol gain: ≥ 5 dB
- Specular point distance: 2000-10000 m
- Exclusion: NaN values, zero-sum DDM

**Target Variable:**
- Class 0: Non-land surface (ocean, water)
- Class 1: Land surface (sp_surface_type 1-7)
- Balanced binary classification

---

## Quick Start

### Recommended Workflow
```
1. ETL Phase:
   - Execute [ETL]raw_counts_dataset_generator.ipynb for feature extraction
   - OR execute [ETL]geok_compressed_dataset_generator_encoders.ipynb for compression 

2. Training Phase:
   - For raw approach: execute [model_training]raw_counts_xgboost.ipynb or raw_counts_tabnet.ipynb
   - For compressed approach: execute [model_training]geok_xgboost.ipynb or geok_tabnet_w_features.ipynb

3. Calibration Phase (optional):
   - Execute xboost_calibration.ipynb to calibrate probabilities

4. Validation:
   - Use ModelTester/DeepTest for comprehensive evaluation
```

### Example Raw Counts Pipeline Usage
```python
# 1. ETL
from NetCDFPreprocessor import NetCDFPreprocessor
preprocessor = NetCDFPreprocessor(root_dir="path/to/netcdf")
fit_data, labels, sp_centers = preprocessor.process_all_files_random_picked()

# 2. Feature Extraction
from DDMFeatureExtractorV2 import DDMFeatureExtractorV2
extractor = DDMFeatureExtractorV2()
features = extractor.extract_features_parallel(fit_data)

# 3. Training
from XGBoostPipeline import XGBoostPipeline
pipeline = XGBoostPipeline(features_df=features, labels_df=labels)
pipeline.hyperparameter_tuning()
pipeline.train_final_model()
pipeline.save_model_and_scaler()
```

### Example Compressed Pipeline Usage
```python
# 1. Compression
from DDMProcessor import DDMProcessor
processor = DDMProcessor(input_folder="path/to/netcdf",
                         output_folder="path/to/output")
processor.load_encoder("encoder.pth")
compressed_dict = processor.process_all_files()

# 2. Training
from TabNetBinaryClassifierOptuna import TabNetBinaryClassifierOptuna
classifier = TabNetBinaryClassifierOptuna(device='cuda')
classifier.prepare_data(compressed_data, labels)
classifier.optimize_hyperparameters(n_trials=100)
classifier.train_final_model()
```

---

## Notebook Index

### ETL Pipeline
1. **[ETL]raw_counts_dataset_generator** - Feature extraction from raw DDM counts
2. **[ETL]geok_compressed_dataset_generator_encoders** - DDM compression via autoencoder

### Model Training
3. **[model_training]raw_counts_xgboost** - XGBoost on raw features
4. **[model_training]raw_counts_xgboost copy** - XGBoost raw features variant
5. **[model_training]raw_counts_tabnet** - TabNet on raw features
6. **[model_training]geok_xgboost** - XGBoost on compressed + features
7. **[model_training]geok_xgboost_dim20** - XGBoost on compressed (pure 20D)
8. **[model_training]geok_tabnet_w_features** - TabNet on compressed + features
9. **[model_training]geok_tabnet_dim_20** - TabNet on compressed (pure 20D)

### Post-Processing & Utilities
10. **xboost_calibration** - XGBoost probability calibration (Platt/Isotonic)
11. **peak_detector_demonstator** - Peak detection algorithm validation

---

## 1. [ETL]raw_counts_dataset_generator

### Execution Mechanics
Batch NetCDF loading → quality filtering (SNR>0, gain≥5, distance 2-10km) → DDM matrix extraction → peak region detection → feature extraction (50+ geometric/statistical features) → balanced train/test split → Parquet output.

### Main Classes

#### NetCDFPreprocessor
NetCDF processor with integrity validation and multi-parameter quality filtering.

**Attributes:**
- `root_dir`: NetCDF files root directory
- `netcdf_file_list`: list of files to process
- `preprocessing_method`: SNR filtering strategy (filtered/unfiltered)
- `return_sp_centers`: flag for specular point coordinate return

**Methods:**
- `check_integrity(f)`: NetCDF structure validation (raw_counts, sp_alt, sp_inc_angle, gains, surface_type, etc.)
- `preprocess_snr_filtered(f)`: DDM extraction with SNR>0, copol/xpol≥5, distance 2000-10000m filtering
- `preprocess_snr_unfiltered(f)`: DDM extraction without SNR constraints
- `process_all_files_random_picked()`: batch processing with random sampling

#### PeakRegionDetector
DDM peak region detector via statistical/adaptive/clustering methods.

**Attributes:**
- `N`: number of pixels in peak region (default 10)
- `method`: detection algorithm (statistical/adaptive/clustering)

**Methods:**
- `find_peak_region(ddm)`: identifies N-pixel connected region around DDM peak
- `extract_comprehensive_features()`: extraction of 20+ geometric-statistical features (area, perimeter, compactness, distances, moments)
- `get_max_distance_in_region()`: calculation of maximum intra-region pixel distance
- `visualize_result()`: peak region plotting on DDM

#### DDMFeatureExtractorV2
Comprehensive feature extractor with quadrant-based + peak-based approach.

**Attributes:**
- `data_format`: input format (5x40 / 40x5)
- `feature_names`: list of extracted feature names
- `peak_detector`: PeakRegionDetector instance

**Methods:**
- `extract_features_parallel()`: parallelization of feature extraction on large datasets
- `gini()`: Gini coefficient for distribution inequality measurement
- `extract_row_features()`: statistical feature extraction per DDM sample

### Dataflow
NetCDF → DDM matrices (n_samples × 5 × 40) → peak detection → feature vector (50+ dim) → labels (binary surface type) → balanced split → Parquet (separate features + labels).

---

## 2. [ETL]geok_compressed_dataset_generator_encoders

### Execution Mechanics
NetCDF loading → quality filtering → MinMaxScaler normalization [0,1] → encoding via PyTorch autoencoder (200D → 20D) → balanced split → Parquet output + persistent scaler.

### Main Classes

#### DDMProcessor
End-to-end pipeline for DDM compression via neural encoder.

**Attributes:**
- `input_folder`: source NetCDF folder path
- `output_folder`: compressed data output path
- `device`: PyTorch device (cuda/cpu)
- `scaler`: MinMaxScaler for normalization
- `encoder`: PyTorch encoder model

**Methods:**
- `set_encoder(encoder_model)`: encoder model setting for compression
- `load_encoder(model_path)`: loading pre-trained encoder from disk
- `preprocess_snr_filtered(f)`: DDM filtering for signal quality (SNR, gain, distance)
- `normalize_data(ddm_data)`: scaling [0,1] with MinMaxScaler
- `compress_data(tensor_data)`: encoding through neural network in batches (DataLoader batch_size=32)

#### Encoder
Autoencoder neural network for DDM compression 200→100→20 dimensions.

**Attributes:**
- `net`: Sequential(Linear(200,100) → ReLU → Linear(100,20) → ReLU)

**Methods:**
- `forward(x)`: encoder forward pass

#### Decoder
Decoder neural network for reconstruction 20→100→200 dimensions.

**Attributes:**
- `net`: Sequential(Linear(20,100) → ReLU → Linear(100,200))

**Methods:**
- `forward(x)`: decoder forward pass

### Dataflow
NetCDF → raw counts (200D flattened) → MinMaxScaler → Tensor → Encoder → compressed (20D) → balanced split (10M train, 2M test) → Parquet + scaler.pkl.

---

## 3. xboost_calibration

### Execution Mechanics
Loading pre-trained XGBoost + scaled features → train/val/test split → CalibratedClassifierCV calibration (sigmoid/isotonic) → metrics comparison (Brier score, log loss, ROC-AUC) → calibration curve visualization → calibrated model saving.

### Main Classes
No custom classes defined. Use sklearn CalibratedClassifierCV with 'sigmoid' (Platt Scaling) and 'isotonic' (Isotonic Regression) methods.

### Dataflow
Compressed features + labels → scaling → split 64% train / 16% val / 20% test → fit CalibratedClassifierCV (CV=5) → calibrated probability prediction → metrics + plots → calibrated_model.pkl.

---

## 4. [model_training]raw_counts_tabnet

### Execution Mechanics
Loading raw features + Parquet labels → balanced split → Optuna hyperparameter search (100 trials) → TabNet training with best params → threshold optimization (F1/precision/recall) → stratified K-fold evaluation → output model + scaler + optimization results.

### Main Classes

#### DataLoader
Parquet loading/caching manager with balanced sampling.

**Attributes:**
- `features_path`: features Parquet path
- `labels_path`: labels Parquet path
- `_features_df_cache`: features DataFrame cache
- `_labels_df_cache`: labels DataFrame cache

**Methods:**
- `load_full_data()`: full dataset loading with optional caching
- `get_max_balanced_sample_info()`: class distribution analysis for balanced sampling
- `clear_cache()`: cache memory release

#### TabNetBinaryClassifier
TabNet wrapper with GPU support, threshold optimization, MLflow tracking.

**Attributes:**
- `device`: computational device (cuda/cpu)
- `tabnet_params`: TabNet hyperparameters (n_d, n_a, n_steps, gamma, lambda_sparse, mask_type)
- `model`: TabNetClassifier instance
- `scaler`: feature scaler
- `track_experiment`: MLflow tracking flag

**Methods:**
- `prepare_data()`: split and feature scaling for training
- `optimize_threshold()`: optimal classification threshold search for target metric (F1/precision/recall)
- `plot_threshold_optimization()`: visualization of threshold impact on metrics
- `get_gpu_memory_info()`: GPU memory monitoring
- `clear_gpu_memory()`: GPU cache release

#### TabNetBinaryClassifierOptuna
TabNet extension with Optuna hyperparameter search.

**Attributes:**
- `best_params`: optimal parameters from tuning
- `study`: Optuna study object
- inherits TabNetBinaryClassifier attributes

**Methods:**
- `objective()`: Optuna objective function for optimization
- `create_balanced_subset()`: stratified balanced subset creation
- `split_train_val_test()`: 3-way splitting with stratification
- `prepare_data()`: data preparation for optimization

#### ModelTester
Testing framework with stratified K-fold evaluation.

**Attributes:**
- `model_path`: saved model path
- `test_data`: test features
- `test_labels`: test labels
- `results`: evaluation results dictionary

**Methods:**
- `load_model()`: TabNet loading from disk
- `create_stratified_splits()`: stratified test fold generation
- `calculate_comprehensive_metrics()`: calculation of accuracy, precision, recall, F1, AUC
- `run_comprehensive_evaluation()`: complete evaluation pipeline execution
- `generate_detailed_report()`: formatted performance report

### Dataflow
Raw features Parquet + labels → balanced split → Optuna search (validation AUC) → train final TabNet → threshold optimization → K-fold test → model + scaler + results.csv.

---

## 5. [model_training]geok_xgboost

### Execution Mechanics
Loading compressed features Parquet → additional statistical feature extraction → RandomizedSearchCV hyperparameter tuning → final XGBoost training → stratified K-fold deep test → feature importance analysis → output model + scaler + CV results.

### Main Classes

#### DDMFeatureExtractor
Statistical feature extractor from compressed DDM representations.

**Methods:**
- `extract_ddm_features(fit_data)`: statistical feature calculation from encoded data
- `gini(array)`: Gini coefficient
- `combined_features_to_dataframe()`: merge encoded + statistical features

#### XGBoostPipeline
XGBoost training pipeline with RandomizedSearchCV and MLflow tracking.

**Attributes:**
- `features_df`: training features DataFrame
- `labels_df`: training labels
- `model`: XGBClassifier
- `best_params`: optimal parameters from CV
- `scaler`: features StandardScaler
- `experiment_name`: MLflow experiment identifier

**Methods:**
- `prepare_data()`: train/test split with scaling
- `hyperparameter_tuning()`: RandomizedSearchCV for optimal parameters
- `train_final_model()`: model training with best hyperparameters
- `save_model_and_scaler()`: model + scaler persistence
- `predict_new_data()`: inference on new samples
- `plot_all_visualizations()`: feature importance, confusion matrix, ROC curves

#### DeepTest
Advanced XGBoost testing framework with stratified evaluation.

**Attributes:**
- `model_path`: saved XGBoost model path
- `scaler_path`: saved scaler path
- `feature_names`: list of feature names
- `results`: evaluation metrics storage

**Methods:**
- `load_model()`: XGBoost loading from joblib/pickle
- `create_test_splits()`: stratified K-fold test splits
- `calculate_metrics()`: comprehensive metrics calculation
- `run_comprehensive_test()`: complete evaluation pipeline
- `get_feature_importance()`: feature importance ranking extraction

### Dataflow
Compressed Parquet (20D) → extract statistical features → combined features (20D + stats) → RandomizedSearchCV → train final model → K-fold deep test → feature importance → model.joblib + scaler.joblib + results.csv.

---

## 6. [model_training]geok_xgboost_dim20

### Execution Mechanics
Identical to geok_xgboost but without additional feature extraction. Direct training on 20D compressed data.

### Main Classes
- DataLoader
- XGBoostPipeline
- DeepTest

### Dataflow
20D encoded Parquet → RandomizedSearchCV → train XGBoost → K-fold test → model + scaler + results.

---

## 7. [model_training]geok_tabnet_w_features

### Execution Mechanics
Combination of geok_xgboost approach (feature extraction) + raw_counts_tabnet (TabNet training). Compressed features + statistical → Optuna tuning → TabNet training → threshold optimization → deep test.

### Main Classes
- DDMFeatureExtractor
- DataLoader
- TabNetBinaryClassifier
- TabNetBinaryClassifierOptuna
- ModelTester

### Dataflow
Compressed Parquet → extract features → combined features → Optuna search → train TabNet → threshold opt → K-fold test → model + scaler + enhanced_features.

---

## 8. [model_training]geok_tabnet_dim_20

### Execution Mechanics
Direct TabNet training on 20D encoded data without feature extraction.

### Main Classes
- DataLoader
- TabNetBinaryClassifier
- TabNetBinaryClassifierOptuna
- ModelTester

### Dataflow
20D Parquet → Optuna TabNet tuning → train → threshold opt → test → model + scaler.

---

## 9. [model_training]raw_counts_xgboost

### Execution Mechanics
XGBoost training on features extracted from raw DDM counts (no compression).

### Main Classes
- DataLoader
- XGBoostPipeline
- DeepTest

### Dataflow
Raw features Parquet → RandomizedSearchCV → train XGBoost → K-fold test → model + scaler + feature_importance.

---

## 10. peak_detector_demonstator

### Execution Mechanics
Demonstration notebook for visual validation of peak detection algorithm on sample DDM.

### Main Classes
- NetCDFPreprocessor
- PeakRegionDetector

### Dataflow
NetCDF sample → extract DDM → apply peak detection (multiple methods) → visualize detected regions → display features.

---

## Complete System Architecture

### ETL Phase
**Path 1 - Raw Counts:**
```
NetCDF → NetCDFPreprocessor (quality filtering) → PeakRegionDetector → DDMFeatureExtractorV2 → Raw Features (50+D) → Parquet
```

**Path 2 - Compressed:**
```
NetCDF → DDMProcessor (quality filtering) → MinMaxScaler → Encoder (200→20D) → Compressed Features → Parquet
```

### Model Training Phase
**Raw Counts Models:**
```
Raw Features → XGBoost/TabNet → Trained Model
```

**Compressed Models (pure 20D):**
```
20D Encoded → XGBoost/TabNet → Trained Model
```

**Compressed + Features Models:**
```
20D Encoded → DDMFeatureExtractor → Combined Features → XGBoost/TabNet → Trained Model
```

### Post-Training Phase
```
Trained Model → Calibration (Platt/Isotonic) → Calibrated Model
All Models → ModelTester/DeepTest (stratified K-fold) → Comprehensive Metrics
```

### Design Pattern
Modular pipeline with ETL/Training/Testing separation. Multi-stage quality filtering ensures data integrity. Dual approach (raw vs compressed) enables compression/performance trade-off analysis. Integrated hyperparameter optimization (Optuna/RandomizedSearchCV). Post-training probabilistic calibration improves prediction reliability. Stratified K-fold validation ensures metric robustness.

---

## Modeling Approaches Comparison Table

| Approach | Input Dim | Feature Extraction | Algorithm | Hyperparameter Search | Output Files |
|-----------|-----------|-------------------|-----------|----------------------|--------------|
| **Raw XGBoost** | 50+ | PeakRegionDetector + DDMFeatureExtractorV2 | XGBoost | RandomizedSearchCV | model.joblib, scaler.joblib, cv_results.csv |
| **Raw TabNet** | 50+ | PeakRegionDetector + DDMFeatureExtractorV2 | TabNet | Optuna (100 trials) | model.zip, scaler.joblib, optuna_results.csv |
| **Compressed XGBoost** | 20+stats | Encoder + DDMFeatureExtractor | XGBoost | RandomizedSearchCV | model.joblib, scaler.joblib, cv_results.csv |
| **Compressed XGBoost 20D** | 20 | Encoder Only | XGBoost | RandomizedSearchCV | model.joblib, scaler.joblib, cv_results.csv |
| **Compressed TabNet** | 20+stats | Encoder + DDMFeatureExtractor | TabNet | Optuna (100 trials) | model.zip, scaler.joblib, optuna_results.csv |
| **Compressed TabNet 20D** | 20 | Encoder Only | TabNet | Optuna (100 trials) | model.zip, scaler.joblib, optuna_results.csv |

**Notes:**
- "20D" approaches test encoder performance without feature engineering
- "+stats" approaches combine encoded representation with statistical features
- RandomizedSearchCV: 5-fold CV, typically 50-100 iterations
- Optuna: TPE sampler, 100 trials, early stopping on validation AUC

---

## Extracted Features - Technical Detail

### Peak Region Features (PeakRegionDetector)
**Geometric:**
- `peak_area`: number of pixels in peak region (N-pixel connected component)
- `peak_perimeter`: peak region perimeter
- `peak_compactness`: area/perimeter² ratio (circularity measure)
- `peak_eccentricity`: equivalent ellipse eccentricity
- `peak_orientation`: equivalent ellipse orientation [degrees]
- `max_distance_in_peak`: maximum Euclidean distance intra-region

**Spatial Statistics:**
- `mean_distance_to_centroid`: mean pixel distance from region centroid
- `std_distance_to_centroid`: standard deviation of distances from centroid
- `peak_centroid_row/col`: region centroid coordinates

**Intensity:**
- `peak_intensity_mean/std/min/max`: pixel intensity statistics in region
- `peak_intensity_range`: max - min intensity in region
- `peak_to_background_ratio`: peak/background intensity ratio

### Statistical Features (DDMFeatureExtractor)
**Global Distribution:**
- `gini_coefficient`: intensity distribution inequality measure (0=uniform, 1=maximum concentration)
- `mean/std/median/max`: complete DDM distribution descriptive statistics
- `skewness/kurtosis`: distribution shape (asymmetry/tails)
- `q25/q75/iqr`: quartiles and interquartile range

**Quadrant Analysis:**
- `quadrant_X_mean/std`: statistics for each DDM quadrant (4 quadrants)
- `quadrant_energy_ratio`: energy distribution between quadrants

**Spatial Frequency:**
- Features from DCT (Discrete Cosine Transform) of DDM (optional)
- Delay-doppler pattern analysis

---

## Hyperparameters - Search Space

### XGBoost RandomizedSearchCV
```python
param_distributions = {
    'n_estimators': [100, 200, 300, 500, 800, 1000],
    'max_depth': [3, 5, 7, 9, 12, 15],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.01, 0.1, 1, 10]
}
```
**Strategies:**
- CV folds: 5
- Scoring: ROC-AUC
- n_iter: 50-100
- n_jobs: -1 (parallel)

### TabNet Optuna
```python
search_space = {
    'n_d': [8, 16, 24, 32, 64],  # decision layer width
    'n_a': [8, 16, 24, 32, 64],  # attention layer width
    'n_steps': [3, 4, 5, 6, 7],  # number of sequential attention steps
    'gamma': [1.0, 1.3, 1.5, 2.0],  # relaxation factor
    'lambda_sparse': [0.0001, 0.001, 0.01],  # sparsity regularization
    'momentum': [0.01, 0.02, 0.05],  # batch normalization momentum
    'mask_type': ['sparsemax', 'entmax']  # attention mask function
}
```
**Strategies:**
- Optimizer: Optuna TPE
- n_trials: 100
- Early stopping: patience=20 (validation loss)
- Objective: validation AUC

---

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: fraction of correct predictions
- **Precision**: TP / (TP + FP) - positive prediction reliability
- **Recall**: TP / (TP + FN) - positive class coverage
- **F1-Score**: harmonic mean of precision/recall
- **ROC-AUC**: area under ROC curve (threshold-independent)
- **PR-AUC**: area under precision-recall curve

### Calibration Metrics
- **Brier Score**: mean squared error between predicted probabilities and labels (0-1 scale, lower=better)
- **Log Loss**: cross-entropy loss (lower=better)
- **ECE (Expected Calibration Error)**: mean difference between confidence and accuracy per bin
- **Calibration Curve**: reliability diagram (predicted prob vs true fraction)

### Validation Strategy
- **Stratified K-Fold**: preserves class distribution in each fold
- **K**: typically 5-10 folds
- **Metrics aggregation**: mean ± std across folds

---

### Computational Requirements
- **GPU**: CUDA-compatible GPU recommended for TabNet (5-10x faster training)
- **RAM**: minimum 16GB, recommended 32GB for complete datasets
- **Storage**: ~50GB for raw + preprocessed datasets + models
---

## Recommended Directory Structure

```
project_root/
│
├── data/
│   ├── raw/
│   │   └── RONGOWAI_L1_SDR_V1.0/        # NetCDF files
│   ├── processed/
│   │   ├── raw_counts_fit_features.parquet
│   │   ├── raw_counts_fit_labels.parquet
│   │   ├── compressed_fit_features.parquet
│   │   └── compressed_fit_labels.parquet
│   └── balanced/
│       ├── balanced_df_old_encoder_10M.parquet
│       └── test_df_old_encoder_2M.parquet
│
├── models/
│   ├── encoders/
│   │   ├── encoder_enh.pth
│   │   └── encoder_all_surface2.pth
│   ├── xgboost/
│   │   ├── xgboost_compressed_with_features.joblib
│   │   ├── xgboost_raw_counts.joblib
│   │   └── scaler.joblib
│   ├── tabnet/
│   │   ├── tabnet_compressed.zip
│   │   ├── tabnet_raw.zip
│   │   └── scaler.joblib
│   └── calibrated/
│       └── xgb_calibrated_model.pkl
│
├── results/
│   ├── cv_results/
│   │   ├── xgboost_cv_results.csv
│   │   └── optuna_study_results.csv
│   ├── feature_importance/
│   │   └── feature_importance_comparison.csv
│   └── plots/
│       ├── calibration_comparison_full.png
│       ├── roc_curves.png
│       └── confusion_matrices.png
│
├── notebooks/
│   ├── [ETL]raw_counts_dataset_generator.ipynb
│   ├── [ETL]geok_compressed_dataset_generator_encoders.ipynb
│   ├── [model_training]*.ipynb
│   ├── xboost_calibration.ipynb
│   └── peak_detector_demonstator.ipynb
│
├── logs/
│   └── mlflow/                          # MLflow tracking artifacts
│
└── README.md
```

---
