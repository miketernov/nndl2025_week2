# Titanic TensorFlow.js – Fix Notes

This project is based on the Titanic survival prediction app.  
Below are the fixes and modifications applied to make the app run correctly.

---

## 1. Data Inspection
- **File:** `app.js`  
- **Functions affected:** `parseCSV`, `loadData`  
- **Fix:** Replaced the naive CSV parser with a robust one (handles quotes, commas, BOM).  
- **Also:** Added normalization of numeric/categorical fields (e.g., `Survived`, `Pclass`, `Sex`, `Embarked`).  
- **Reason:** Without this, charts in *Data Inspection* were empty.

---

## 2. Evaluation Metrics
- **File:** `app.js`  
- **Function:** `plotROC`  
- **Fix:** Sorted ROC points and ensured AUC is always positive.  
- **Reason:** AUC was showing as negative (e.g., `-0.90`).  

---

## 3. Prediction
- **File:** `app.js`  
- **Functions:** `predict`, table rendering  
- **Fix:** Flattened TensorFlow predictions (`[[0.8],[0.3],...]`) into numbers.  
- **Reason:** Fixed the error `row[key].toFixed is not a function`.

---

## 4. Export
- **File:** `app.js`  
- **Function:** `exportResults`  
- **Fix:** Flattened predictions before CSV export, formatted probabilities, and added model export with `model.save`.  
- **Reason:** Submission/probabilities CSVs now generate correctly, and the model can be downloaded.

---

## Summary
- Fixed CSV parsing and normalization (charts now display).  
- Fixed ROC/AUC calculation (no more negative values).  
- Fixed prediction formatting error.  
- Fixed CSV export and added model download.  

---

✅ With these changes, **all stages (Data Inspection → Evaluation → Prediction → Export)** work correctly.
