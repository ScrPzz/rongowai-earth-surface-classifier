# TabNet ONNX Conversion

This directory contains tools and notebooks for converting trained TabNet models to ONNX format for optimized inference in production environments.

---

## Overview

Convert PyTorch TabNet models and StandardScaler preprocessors to ONNX format for:
- Faster inference performance
- Cross-platform deployment
- Production optimization
- Framework-agnostic serving

---

## Files Structure

```
tabnet_onnx/
├── model.zip                          # Original TabNet model (PyTorch)
├── scaler.joblib                      # StandardScaler for preprocessing
├── test_data.parquet                  # Test features for validation
├── test_labels.parquet                # Test labels for validation
├── 01_convert_tabnet_to_onnx.ipynb    # Conversion notebook
├── 02_test_tabnet_onnx.ipynb          # Testing notebook with real data
├── 03_tabnet_inference.ipynb          # Inference examples
├── README.md                          # This file
└── onnx_models/                       # Generated ONNX models
    ├── model.onnx                     # TabNet model in ONNX format
    ├── scaler.onnx                    # StandardScaler in ONNX format
    ├── onnx_metadata.json             # Model metadata
    ├── confusion_matrices.png         # Test visualizations
    ├── performance_analysis.png       # Performance plots
    └── predictions.csv                # Example predictions
```

---

## Quick Start

### Prerequisites

```bash
pip install jupyter numpy pandas torch onnx onnxruntime skl2onnx scikit-learn matplotlib seaborn joblib
```

### Running the Notebooks

Execute notebooks in sequence:

1. **01_convert_tabnet_to_onnx.ipynb** - Convert models to ONNX
2. **02_test_tabnet_onnx.ipynb** - Validate conversion with real data
3. **03_tabnet_inference.ipynb** - Production inference examples

```bash
jupyter notebook
```

Or execute from command line:
```bash
jupyter nbconvert --to notebook --execute 01_convert_tabnet_to_onnx.ipynb
```

---

## Notebook Descriptions

### Notebook 01: Conversion

**Purpose:** Convert TabNet PyTorch model and StandardScaler to ONNX format

**Process:**
1. Extract TabNet model from model.zip archive
2. Load PyTorch model and StandardScaler
3. Convert both to ONNX using torch.onnx.export and skl2onnx
4. Validate ONNX models
5. Save models and metadata

**Outputs:**
- `onnx_models/model.onnx` - TabNet classifier
- `onnx_models/scaler.onnx` - StandardScaler preprocessor
- `onnx_models/onnx_metadata.json` - Model configuration

**Runtime:** Approximately 10-20 seconds

---

### Notebook 02: Testing

**Purpose:** Comprehensive validation using real GNSS-R test data

**Tests Performed:**
1. Prediction accuracy comparison (PyTorch vs ONNX)
2. Performance metrics evaluation
3. Confusion matrix analysis
4. Batch size performance benchmarking
5. Edge case validation

**Data Sources:**
- `test_data.parquet` - Real test features
- `test_labels.parquet` - Ground truth labels

**Outputs:**
- `onnx_models/confusion_matrices.png`
- `onnx_models/performance_analysis.png`
- Console output with detailed metrics

**Runtime:** 30-60 seconds for 10K samples

---

### Notebook 03: Inference Examples

**Purpose:** Production-ready inference patterns

**Examples Included:**
1. Single sample prediction
2. Batch processing (100 samples)
3. Real test data inference
4. Performance benchmarking
5. CSV export of predictions

**Use Cases:**
- Learning ONNX model usage
- Production deployment reference
- Integration testing

**Runtime:** 10-20 seconds

---

## Model Information

**Task:** Binary classification (Ocean vs Land)

**Features:** 78 engineered features from GNSS-R DDM data

**Input:**
- Shape: `[batch_size, 78]`
- Type: `float32`
- Preprocessing: StandardScaler (z-score normalization)

**Output:**
- Shape: `[batch_size, 2]`
- Type: `float32`
- Classes: 0 (Ocean), 1 (Land)
- Values: Class probabilities

---

## Usage Example

### Python Code

```python
import numpy as np
import onnxruntime as rt

# Load ONNX models
scaler_sess = rt.InferenceSession("onnx_models/scaler.onnx")
model_sess = rt.InferenceSession("onnx_models/model.onnx")

# Prepare input (78 features)
X = np.random.randn(1, 78).astype(np.float32)

# Step 1: Scale input
scaler_input = scaler_sess.get_inputs()[0].name
scaler_output = scaler_sess.get_outputs()[0].name
X_scaled = scaler_sess.run([scaler_output], {scaler_input: X})[0]

# Step 2: Predict
model_input = model_sess.get_inputs()[0].name
probabilities = model_sess.run(None, {model_input: X_scaled})[0]

# Get prediction
prediction = 1 if probabilities[0][1] > 0.5 else 0
print(f"Prediction: {prediction} (Ocean=0, Land=1)")
print(f"Probabilities: {probabilities[0]}")
```

### Using Helper Class

```python
from pathlib import Path
from tabnet_inference import TabNetONNXPredictor

# Initialize
predictor = TabNetONNXPredictor()

# Single prediction
result = predictor.predict_single(X[0])
print(result)
# {'prediction': 1, 'prediction_label': 'Land',
#  'confidence': 0.7234, 'probabilities': {'Ocean': 0.2766, 'Land': 0.7234}}

# Batch prediction
predictions, probabilities = predictor.predict(X_batch)
```

---

## Performance

Expected inference performance (approximate):

| Batch Size | Original PyTorch | ONNX       | Speedup |
|------------|------------------|------------|---------|
| 1          | 5ms              | 0.5ms      | 10x     |
| 10         | 12ms             | 3ms        | 4x      |
| 100        | 45ms             | 15ms       | 3x      |
| 1,000      | 180ms            | 120ms      | 1.5x    |

**Note:** Actual performance depends on hardware and model complexity.

**Recommendations:**
- Use ONNX for batch sizes 1-1000
- Optimal throughput at batch size 100-500
- CPU inference sufficient for most use cases

---

## Expected Results

### Conversion (Notebook 01)

```
CONVERSION COMPLETE
Output directory: onnx_models/
  scaler.onnx (1.2 KB)
  model.onnx (15.8 MB)
  onnx_metadata.json
```

### Testing (Notebook 02)

**Prediction Accuracy:**
- Predictions match: 100% (or >99.9%)
- Max probability difference: < 1e-5
- Mean probability difference: < 1e-6

**Performance Metrics:**
```
Metric      Original    ONNX        Difference
Accuracy    0.8750      0.8750      0.00e+00
Precision   0.8680      0.8680      0.00e+00
Recall      0.8820      0.8820      0.00e+00
F1          0.8750      0.8750      0.00e+00
ROC-AUC     0.9430      0.9430      0.00e+00
```

---

## Troubleshooting

### Issue: Model file not found

**Error:**
```
FileNotFoundError: Model not found: model.zip
```

**Solution:**
- Ensure `model.zip` is in the tabnet_onnx directory
- Check that model was saved correctly
- Verify file permissions

### Issue: Feature count mismatch

**Error:**
```
ValueError: Expected 78 features, got 80
```

**Solution:**
- Verify test data has correct number of features
- Check that scaler and model were trained on same data
- Ensure preprocessing matches training pipeline

### Issue: Out of memory during testing

**Solution:**
- Reduce test sample size in notebook 02:
  ```python
  n_samples = min(5000, len(X_test_array))  # Use 5K instead of 10K
  ```
- Process data in smaller batches
- Close other applications

### Issue: ONNX predictions differ significantly

**Check:**
1. Input is float32 dtype
2. Data is scaled with ONNX scaler first
3. No NaN or Inf values in input
4. Feature order matches training
5. PyTorch model is in eval() mode

---

## Technical Details

### TabNet Architecture

TabNet uses sequential attention mechanism for feature selection. When converting to ONNX:
- Attention masks are preserved
- Feature importance can be extracted
- Dynamic feature selection maintained

### ONNX Export

Conversion uses `torch.onnx.export` with:
- Opset version: 12 (widely supported)
- Dynamic batch size support
- Constant folding enabled
- Input/output names specified

### Scaler Conversion

StandardScaler converted via `skl2onnx`:
- Mean and standard deviation preserved
- Operations: subtract mean, divide by std
- Numerical precision: float32

---

## Input Data Requirements

### Test Data Format

**test_data.parquet:**
- Type: Parquet file
- Shape: `[n_samples, 78]`
- Dtype: float32 or float64 (auto-converted)
- Features: GNSS-R engineered features

**test_labels.parquet:**
- Type: Parquet file
- Shape: `[n_samples]` or `[n_samples, 1]`
- Dtype: int (0 or 1)
- Values: 0 (Ocean), 1 (Land)

### Feature List

Model expects these 78 features (order matters):
- Global statistics (mean, std, min, max, median, range, etc.)
- Peak detection features
- Spectral features (FFT)
- Quadrant-based features
- Window-based features
- Differential features
- Autocorrelation features

Full feature list available in `onnx_metadata.json` after conversion.

---

## Deployment

### Production Checklist

1. **Validate conversion**
   ```bash
   jupyter nbconvert --execute 02_test_tabnet_onnx.ipynb
   # Ensure prediction accuracy > 99.9%
   ```

2. **Test with real data**
   - Run inference on representative samples
   - Verify output distributions
   - Check edge cases

3. **Performance testing**
   - Benchmark expected load
   - Test concurrent requests
   - Monitor memory usage

4. **Integration**
   - Load ONNX sessions once at startup
   - Reuse sessions for all predictions
   - Implement error handling

### Deployment Options

**Python Backend:**
```python
from fastapi import FastAPI
from tabnet_inference import TabNetONNXPredictor

app = FastAPI()
predictor = TabNetONNXPredictor()  # Load once

@app.post("/predict")
def predict(features: list):
    result = predictor.predict_single(np.array(features))
    return result
```

**Other Languages:**
- C++: ONNX Runtime C++ API
- Java: ONNX Runtime Java API
- JavaScript: ONNX.js
- C#: ONNX Runtime .NET API

---

## Notes

1. **Preprocessing Order:** Always apply scaler before model inference

2. **Data Type:** Use float32 for best performance and compatibility

3. **Batch Size:** Optimal range is 10-500 samples per batch

4. **Session Reuse:** Load ONNX sessions once and reuse for all predictions

5. **Model Updates:** Re-run conversion notebook after retraining model

6. **GPU Support:** Install `onnxruntime-gpu` for GPU inference:
   ```bash
   pip install onnxruntime-gpu
   ```

---

## References

- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [TabNet Paper](https://arxiv.org/abs/1908.07442)
- [skl2onnx Guide](http://onnx.ai/sklearn-onnx/)

---

## Support

For issues or questions:
- Check troubleshooting section above
- Review notebook outputs for error messages
- Verify input data format and features
- Ensure all dependencies are installed

---

**Generated:** 2025-09-30
**Version:** 1.0
**Status:** Ready for use