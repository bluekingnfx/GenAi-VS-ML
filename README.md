# Lead Conversion Prediction: GenAI (LoRA) vs. Traditional ML (Stacked Generalization)

This project explores and compares two distinct approaches to predict lead conversion:
1.  **Generative AI (GenAI)**: Fine-tuning a Large Language Model (LLM) using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
2.  **Traditional Machine Learning (ML)**: Building a robust ensemble model using Stacked Generalization.

---

## 1. GenAI Approach (LoRA Fine-tuning)
Located in `GENAI_LORA_Model.ipynb`.

This notebook demonstrates how to adapt a general-purpose LLM for a specific classification task (Lead Conversion).

### Methodology
- **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Technique**: **LoRA (Low-Rank Adaptation)** for efficient training.
- **Quantization**: 4-bit quantization (`bitsandbytes`) to reduce memory footprint.
- **Data Preparation**: Formatted the dataset into instruction-response pairs:
    - *Instruction*: "Predict if the lead will convert based on the following features..."
    - *Response*: "Converted" or "Not Converted".
- **Hyperparameters**:
    - Rank (r): 8
    - Alpha: 16
    - Target Modules: `q_proj`, `v_proj`
    - Batch Size: 2 (with 8 gradient accumulation steps)
    - Epochs: 3

### Performance
- **Accuracy**: ~80.28%
- **Metrics**: High Precision and Recall, showcasing the model's ability to understand feature context through natural language.

---

## 2. Traditional ML Approach (Stacked Generalization)
Located in `Stacked generalization.ipynb`.

This notebook focuses on a more classical feature-driven pipeline, culminating in a powerful ensemble.

### Methodology
- **Data Preprocessing**:
    - Handled missing values (imputation).
    - Categorical Encoding: used `category_encoders.TargetEncoder` and grouping for high-cardinality features (e.g., Lead Source, Country).
    - Feature Scalling: `StandardScaler`.
- **Base Models**:
    - Logistic Regression (Baseline)
    - Random Forest Classifier
    - XGBoost Classifier
    - Support Vector Classifier (SVC)
- **Ensemble Technique**: **Stacked Generalization**.
    - Used "Out-of-Fold" (OOF) predictions from base models as inputs for a meta-model.
    - **Meta-model**: A Dense Neural Network built with Keras.

### Performance
- **Accuracy**: ~81.21%
- **Key Finding**: The stacked ensemble outperformed standalone models by effectively combining the strengths of tree-based, linear, and kernel-based learners.

---

## 3. Comparison Summary

| Metric | GenAI (LoRA) | Stacked ML Ensemble |
| :--- | :--- | :--- |
| **Accuracy** | 80.28% | **81.21%** |
| **Complexity** | High (Deep Learning/NLP) | Moderate (Feature Engineering) |
| **Interpretability** | Moderate (Prompt-based) | High (Feature Importance available) |
| **Training Resources** | GPU Required (VRAM) | CPU/GPU (Faster on small sets) |

While the Traditional ML approach yielded slightly higher accuracy in this specific case, the GenAI approach demonstrates significant potential for handling unstructured data or scenarios where reasoning over features is beneficial.

---

## Installation & Requirements

To run these notebooks, ensure you have Python installed along with the following primary libraries:

```bash
pip install torch transformers peft datasets bitsandbytes accelerate
pip install pandas numpy scikit-learn xgboost tensorflow category_encoders matplotlib seaborn
```

*Note: The GenAI notebook requires a CUDA-enabled GPU for efficient quantization and fine-tuning.*
