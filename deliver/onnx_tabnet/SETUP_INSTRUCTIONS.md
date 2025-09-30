# TabNet ONNX Setup Instructions

Quick guide to prepare and run the TabNet ONNX conversion notebooks.

---

## Step 1: Prepare Input Files

Place the following files in the `tabnet_onnx/` directory:

### Required Files

1. **model.zip** - Your trained TabNet PyTorch model
2. **scaler.joblib** - StandardScaler used for preprocessing
3. **test_data.parquet** - Test features for validation
4. **test_labels.parquet** - Test labels for validation

### File Specifications

**model.zip structure:**
```
model.zip
└── model.pt (or model.pth, checkpoint.pt, etc.)
```

**scaler.joblib:**
- Sklearn StandardScaler fitted on training data
- Must have same number of features as model input (typically 78)

**test_data.parquet:**
- Shape: [n_samples, 78]
- Dtype: float32 or float64
- Features: Same order as training

**test_labels.parquet:**
- Shape: [n_samples] or [n_samples, 1]
- Dtype: int
- Values: 0 (Ocean) or 1 (Land)

---

## Step 2: Link or Copy Test Data

If test data exists in `../processed_data/`, create links:

```bash
cd /home/atogni/Desktop/deliver/tabnet_onnx

# Option 1: Create symbolic links
ln -s ../processed_data/compressed_test_features.parquet test_data.parquet
ln -s ../processed_data/compressed_test_labels.parquet test_labels.parquet

# Option 2: Copy files
cp ../processed_data/compressed_test_features.parquet test_data.parquet
cp ../processed_data/compressed_test_labels.parquet test_labels.parquet
```

---

## Step 3: Install Dependencies

```bash
pip install jupyter numpy pandas torch onnx onnxruntime skl2onnx scikit-learn matplotlib seaborn joblib
```

---

## Step 4: Run Notebooks

### Option A: Jupyter Notebook Interface

```bash
cd /home/atogni/Desktop/deliver/tabnet_onnx
jupyter notebook
```

Then open and run in order:
1. `01_convert_tabnet_to_onnx.ipynb`
2. `02_test_tabnet_onnx.ipynb`
3. `03_tabnet_inference.ipynb`

### Option B: Command Line Execution

```bash
cd /home/atogni/Desktop/deliver/tabnet_onnx

# Run conversion
jupyter nbconvert --to notebook --execute 01_convert_tabnet_to_onnx.ipynb

# Run testing
jupyter nbconvert --to notebook --execute 02_test_tabnet_onnx.ipynb

# Run examples
jupyter nbconvert --to notebook --execute 03_tabnet_inference.ipynb
```

---

## Step 5: Verify Outputs

After running, check that these files were created:

```
onnx_models/
├── model.onnx                  # TabNet model in ONNX format
├── scaler.onnx                 # StandardScaler in ONNX format
├── onnx_metadata.json          # Model metadata
├── confusion_matrices.png      # Test visualizations
├── performance_analysis.png    # Performance plots
└── predictions.csv             # Example predictions
```

---

## Troubleshooting

### Model not found error

If you see `FileNotFoundError: Model not found: model.zip`:

1. Check that `model.zip` exists in `tabnet_onnx/` directory
2. Verify the zip file contains `model.pt` or similar
3. Ensure file has read permissions

### Scaler feature mismatch

If you see `Feature mismatch: model=78, scaler=80`:

1. Verify scaler was fitted on same features as model
2. Check that test data has same number of features
3. Ensure feature order matches training

### Test data not found

If you see `FileNotFoundError: Test data not found`:

1. Check that test_data.parquet and test_labels.parquet exist
2. Verify links are correct if using symbolic links
3. Try copying files instead of linking

### Out of memory

If you encounter memory errors:

1. Reduce test sample size in notebook 02:
   ```python
   n_samples = min(5000, len(X_test_array))
   ```
2. Close other applications
3. Process in smaller batches

---

## Quick Checklist

Before running notebooks:

- [ ] model.zip is in tabnet_onnx/ directory
- [ ] scaler.joblib is in tabnet_onnx/ directory
- [ ] test_data.parquet is available (linked or copied)
- [ ] test_labels.parquet is available (linked or copied)
- [ ] All dependencies are installed
- [ ] Jupyter is installed and working

After running notebook 01:

- [ ] onnx_models/model.onnx exists
- [ ] onnx_models/scaler.onnx exists
- [ ] onnx_models/onnx_metadata.json exists
- [ ] No errors in notebook output

After running notebook 02:

- [ ] Prediction accuracy > 99%
- [ ] Performance metrics match original model
- [ ] Confusion matrices generated
- [ ] Performance plots generated

---

## Next Steps

After successful conversion and testing:

1. Review test results in notebook 02 output
2. Examine performance metrics and plots
3. Run inference examples in notebook 03
4. Integrate ONNX models into your application
5. Deploy to production environment

---

## Support

For detailed information, see:
- `README.md` - Complete documentation
- `MOCKUP_FILES.txt` - File format specifications
- Notebook outputs - Detailed error messages

---

**Last Updated:** 2025-09-30