## 🩺 Pneumonia Detection from Chest X-rays (EfficientNetB0)
This project implements a deep learning pipeline for the automated binary classification of Pneumonia from Chest X-ray (CXR) images. It uses EfficientNetB0 via Transfer Learning and includes an interactive Streamlit web application with live Grad-CAM model explainability.

### 🔗 Live demo: https://YOUR-APP-NAME.streamlit.app
<!-- replace the URL above after deploying (see "Deploy a permanent link" below) -->

*Research / education project — not a medical device and not a diagnosis.*

> [!NOTE]
> **Project versions.** **v1** is the first experiment (`original_experiment/` — frozen EfficientNetB0 head, trained on 5,218 images); its results are listed under *Results*. **v2** is the version in this repository's code and in the live demo: the training pipeline was re-implemented with a patient-level validation split, fine-tuning of the top EfficientNetB0 layers and identical preprocessing in training, evaluation and the app. Both are evaluated on the same 624-image Kaggle test split.

## Technical Highlights
- **Transfer Learning (EfficientNetB0):** ImageNet-pretrained EfficientNetB0 with a small head (GlobalAveragePooling → Dense 256 → Dropout 0.3 → sigmoid).
- **Two-phase training (v2):** the head is trained first on a frozen backbone; then the top 30 layers are fine-tuned (BatchNorm frozen) at a low learning rate. *(v1 used the frozen backbone only.)*
- **Class Weighting:** balanced class weights handle the NORMAL/PNEUMONIA imbalance.
- **Patient-level validation split (v2):** the Kaggle `val/` folder has only 16 images, so `train/` + `val/` are pooled and re-split ≈85/15 **by patient** (id parsed from the file name) — no patient appears in both training and validation. The Kaggle `test/` split is only used for the final evaluation.
- **Data augmentation:** rotation, zoom, translation and contrast changes on the training set only (v1 also used horizontal flips; v2 omits them because flipping mirrors chest anatomy).
- **Grad-CAM Explainability:** the Streamlit app overlays a heatmap of the regions that pushed the model towards the *predicted* class.

## Results
Both versions are evaluated on the Kaggle test split (624 images: 234 NORMAL, 390 PNEUMONIA), decision threshold 0.5.

| Metric | v1 — first experiment | v2 — current model |
|---|---|---|
| Accuracy | 0.8942 | 0.843 |
| Precision (PNEUMONIA) | 0.8876 | 0.807 |
| Recall / sensitivity (PNEUMONIA) | 0.9513 | **0.985** |
| Specificity (NORMAL recall) | ≈ 0.80 | 0.607 |
| ROC-AUC | 0.9644 | 0.966 |

- **v1** figures are the saved outputs of [`original_experiment/pneumonia_train_original.ipynb`](original_experiment/pneumonia_train_original.ipynb) (test loss 0.2408; NORMAL precision 0.91 / recall 0.80, PNEUMONIA precision 0.89 / recall 0.95).
- **v2** figures are in [`artifacts/metrics.json`](artifacts/metrics.json), with training curves, confusion matrix, ROC curve and Grad-CAM samples next to it in [`artifacts/`](artifacts/). v2 reaches a sensitivity of 98.5%, which matters in a screening setting where a missed case costs more than a false alarm; the decision threshold can be tuned to trade sensitivity against specificity.

## Data used
Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle), paediatric chest X-rays.

| Run | Training | Validation | Test |
|---|---|---|---|
| v1 — first experiment | **5,218** (1,343 NORMAL / 3,875 PNEUMONIA) | 224 (112 / 112) | 624 (234 / 390) |
| v2 — current model | pooled Kaggle train + val, split ≈85/15 by patient (exact counts in `artifacts/metrics.json` and on the app's *Results & About* page) | | 624 (234 / 390) |

`train.py` downloads the data automatically with `kagglehub`; manually the layout is:
```
chest_xray/
├── train/  (NORMAL/, PNEUMONIA/)
├── val/    (NORMAL/, PNEUMONIA/)
└── test/   (NORMAL/, PNEUMONIA/)
```

## Repository Structure
| Path | Purpose |
|---|---|
| `app.py` | Streamlit app (Home / Pneumonia Detection / Results & About) |
| `core.py` | preprocessing, prediction and Grad-CAM used by the app and training |
| `train.py` | download data → patient-level split → train → evaluate → save model + `artifacts/` |
| `pneumonia_detector_model.keras` | trained v2 model used by the app |
| `artifacts/` | v2 metrics and plots |
| `original_experiment/` | the v1 Colab notebooks (training + single-image test) with their saved outputs |
| `colab_run.ipynb` | Colab runner (install → train → serve through a temporary tunnel) |
| `.streamlit/config.toml` | app theme |
| `requirements.txt`, `requirements-train.txt` | app / training dependencies |
| `homepage.jpeg` | image shown on the app's home page |

## Technologies Used
Python · TensorFlow / Keras (EfficientNetB0) · NumPy · Pillow · Matplotlib · Seaborn (v1 notebook plots) · scikit-learn · Streamlit · kagglehub

## Streamlit
`app.py` lets you upload a chest X-ray and see the prediction, the P(PNEUMONIA) score and the Grad-CAM explanation.

**Prerequisites:** Python 3.10+; the trained model `pneumonia_detector_model.keras` next to `app.py` (or set `MODEL_PATH=/path/to/model`).

**Installation**
```bash
pip install -r requirements.txt
```
**Run**
```bash
streamlit run app.py
```
This opens the web application in your browser, where you can upload an X-ray image and view the prediction and the Grad-CAM explanation.

### Deploy a permanent link (Streamlit Community Cloud, free)
1. Push this repo (including the model file) to a public GitHub repository.
2. Open <https://share.streamlit.io> ▸ **New app** ▸ choose the repo, branch `main`, main file `app.py`.
3. **Advanced settings** ▸ Python 3.11 or 3.12 ▸ **Deploy**. You get a permanent `https://….streamlit.app` URL — paste it at the top of this README.

Free apps go to sleep after a period without visitors; the first visitor just clicks *“Yes, get this app back up”* (about a minute).

## Training
```bash
pip install -r requirements-train.txt
python train.py            # downloads the Kaggle data; add --help for options
```
This writes `pneumonia_detector_model.keras` and `artifacts/`. Training is stochastic, so a re-run gives slightly different numbers — quote the ones in your own `artifacts/metrics.json`. In Google Colab, `colab_run.ipynb` automates install → train → serve.

**Pin the TensorFlow version you trained with** (keeps the saved model loadable in the deployed app):
```python
import tensorflow as tf; print(tf.__version__)   # then set tensorflow-cpu==<that version> in requirements.txt
```

## Scope & Next Steps
- **External validation:** the data come from one paediatric centre; evaluating on additional datasets is the natural next step.
- **Threshold tuning:** choose the decision threshold on a validation set to balance sensitivity and specificity.
- **Patient-level cross-validation** for tighter performance estimates.
- **Advanced Grad-CAM:** smoother or higher-resolution maps (e.g. Grad-CAM++).
- **API Deployment:** wrap the model in a REST API (e.g. FastAPI) for integration into other systems such as Electronic Health Records.
- **Multi-Class Detection:** extend the model to classify viral vs. bacterial pneumonia.
