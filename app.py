import json
import os

import streamlit as st
from PIL import Image

import core

st.set_page_config(page_title="Chest X-ray Pneumonia Detection", page_icon="🩻", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DISCLAIMER = (
    "Research / education demo only — **not a medical device and not a diagnosis.** "
    "Trained on a small paediatric dataset; never use it for clinical decisions."
)

st.markdown(
    """
<style>
/* sticky section banner (stays at the top while the page scrolls) */
.element-container:has(.hero), [data-testid="stElementContainer"]:has(.hero){
  position:sticky; top:2.875rem; z-index:100; background:#ffffff; padding-bottom:.55rem;}
.hero{background:linear-gradient(135deg,#0b3c5d 0%,#0e7c86 100%);color:#fff;padding:1rem 1.5rem;border-radius:14px}
.hero-title{color:#fff;font-size:1.7rem;font-weight:700;line-height:1.25}
.hero-sub{color:#e6f4f6;font-size:.98rem;margin-top:.15rem}
.card{background:#fff;border:1px solid #e3eaee;border-radius:14px;padding:1rem 1.2rem;
      box-shadow:0 1px 3px rgba(16,40,60,.07);height:8.4rem}
.card-title{font-size:1.1rem;font-weight:700;color:#0b3c5d;margin-bottom:.35rem}
.card p{margin:0;color:#42525e;font-size:.95rem}
/* no hover "link" icons next to headings */
[data-testid="stHeaderActionElements"]{display:none !important}
@media (max-width:640px){.hero-title{font-size:1.25rem}.hero-sub{font-size:.85rem}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model(path):
    return core.load_model(path)


@st.cache_data(show_spinner=False)
def load_metrics(path, mtime):
    with open(path) as f:
        return json.load(f)


def get_metrics():
    p = os.path.join(HERE, "artifacts", "metrics.json")
    return load_metrics(p, os.path.getmtime(p)) if os.path.exists(p) else None


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


st.sidebar.title("🩻 Pneumonia AI")
page = st.sidebar.radio("Navigate", ["Home", "Pneumonia Detection", "Results & About"], label_visibility="collapsed")
st.sidebar.caption(DISCLAIMER)

# --------------------------------------------------------------------------- Home
if page == "Home":
    hero("Chest X-Ray Pneumonia Detection",
         "EfficientNetB0 classifier with Grad-CAM explanations — upload an X-ray and see what the model looked at.")
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown('<div class="card"><div class="card-title">1 · Upload</div><p>Add a PNG or JPG chest X-ray on '
                'the <b>Pneumonia Detection</b> page.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><div class="card-title">2 · Predict</div><p>The model returns the '
                'probability of pneumonia and a NORMAL / PNEUMONIA label.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><div class="card-title">3 · Explain</div><p>Grad-CAM highlights the image '
                'regions that pushed the model towards its answer.</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        m = get_metrics()
        recall_line = ""
        if m and "recall_pneumonia" in m["test"]:
            recall_line = (f" On the same 624-image Kaggle test split it reaches a pneumonia recall of "
                           f"**{m['test']['recall_pneumonia']:.3f}** (0.951 in the first experiment).")
        st.info(
            "**About this version.** This app runs the second version of the project. The first experiment "
            "(frozen EfficientNetB0 head, 5,218 training images) is kept in `original_experiment/` and its "
            "results are listed under *Results & About*. For this version the training pipeline was re-implemented to:\n\n"
            "- use a **patient-level validation split**, so no patient appears in both training and validation;\n"
            "- **fine-tune the top EfficientNetB0 layers** after training the classification head;\n"
            "- use **identical image preprocessing** in training, evaluation and this app.\n\n"
            "It is the model that produces the predictions on the next page." + recall_line
        )
    with right:
        img = os.path.join(HERE, "homepage.jpeg")
        if os.path.exists(img):
            st.image(img)

# ---------------------------------------------------------------- Detection page
elif page == "Pneumonia Detection":
    hero("Analyse a chest X-ray", "Upload an image to get the prediction and the Grad-CAM explanation.")

    model_path = core.resolve_model_path()
    if model_path is None:
        st.error(
            "Trained model not found. Run `python train.py` (or the Colab notebook) first — it creates "
            "`pneumonia_detector_model.keras` next to `app.py`. You can also set the `MODEL_PATH` "
            "environment variable to a model file."
        )
        st.stop()
    with st.spinner("Loading the model (first run takes a few seconds)..."):
        try:
            model = get_model(model_path)
        except Exception as e:  # noqa: BLE001
            st.error(f"Error loading the model: {e}")
            st.stop()
    layer_name = core.find_gradcam_layer(model)

    up_col, res_col = st.columns([1, 1.15], gap="large")
    with up_col:
        uploaded = st.file_uploader("Choose a PNG or JPG chest X-ray", type=["png", "jpg", "jpeg"])
        pil_image = None
        if uploaded:
            try:
                pil_image = Image.open(uploaded).convert("RGB")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the image: {e}")
                st.stop()
            st.image(pil_image, caption="Uploaded X-ray")
        else:
            st.caption("Best results with frontal chest X-rays similar to the training data (paediatric, PA view). "
                       "The image is processed in memory and not stored.")

    with res_col:
        if pil_image is None:
            st.subheader("Prediction result", anchor=False)
            st.caption("The prediction, the pneumonia probability and the Grad-CAM explanation appear here "
                       "once an image is uploaded.")
        else:
            with st.spinner("Analysing..."):
                batch = core.preprocess(pil_image)
                label, idx, p_pneu, conf = core.predict(model, batch)
            st.subheader("Prediction result", anchor=False)
            msg = f"Prediction: **{label}** — confidence {conf:.1%}"
            (st.error if idx == 1 else st.success)(msg)
            st.progress(p_pneu, text=f"P(PNEUMONIA) = {p_pneu:.3f}  ·  decision threshold 0.5")

    if pil_image is not None:
        st.subheader("Model explainability (Grad-CAM)", anchor=False)
        st.caption(
            f"Warm colours mark the regions that pushed the model towards **{label}**. "
            "Grad-CAM shows where the model looked; it is not a clinical localisation."
        )
        if layer_name is None:
            st.error("No convolutional layer found for Grad-CAM.")
        else:
            try:
                heat = core.make_gradcam_heatmap(batch, model, layer_name, target_pneumonia=bool(idx))
                overlay = core.overlay_heatmap(pil_image, heat)
                g1, g2 = st.columns(2)
                g1.image(pil_image, caption="Original")
                g2.image(overlay, caption="Grad-CAM overlay")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error generating Grad-CAM heatmap: {e}")

# ------------------------------------------------------------------ Results page
else:
    hero("Results & About", "Evaluation on the held-out Kaggle test split, the data used, and the method.")
    t_now, t_first, t_about = st.tabs(["Current model (v2)", "First experiment (v1)", "Data, method & next steps"])
    m = get_metrics()

    with t_now:
        if m is None:
            st.info("No evaluation results found yet — run `train.py` to generate `artifacts/metrics.json`.")
        else:
            t = m["test"]
            dt = m.get("data", {}).get("test", {})
            split = f" ({dt['NORMAL']} NORMAL / {dt['PNEUMONIA']} PNEUMONIA)" if dt else ""
            st.markdown(
                f"Evaluated on the **{t.get('n', 624)}-image Kaggle test split**{split}, which was never used "
                "for training or validation. Decision threshold 0.5."
            )
            shown = [("Accuracy", "accuracy"), ("Precision (pneumonia)", "precision_pneumonia"),
                     ("Recall (pneumonia)", "recall_pneumonia"), ("Specificity", "specificity"), ("ROC AUC", "roc_auc")]
            for col, (lab, key) in zip(st.columns(len(shown)), shown):
                if key in t:
                    col.metric(lab, f"{t[key]:.3f}")
            if "recall_pneumonia" in t and "roc_auc" in t:
                st.success(
                    f"**High sensitivity.** The model detects {t['recall_pneumonia']:.1%} of the pneumonia cases in the "
                    f"test split (ROC AUC {t['roc_auc']:.3f}). High sensitivity is the priority in a screening setting, "
                    "where a missed case costs more than a false alarm, and the decision threshold can be adjusted "
                    "to trade sensitivity against specificity."
                )
            a, b = st.columns(2)
            for col, name in ((a, "confusion_matrix.png"), (b, "roc_curve.png")):
                p = os.path.join(HERE, "artifacts", name)
                if os.path.exists(p):
                    col.image(p)
            for name, title in (("training_curves.png", "Training curves"),
                                ("gradcam_samples.png", "Grad-CAM samples on test images")):
                p = os.path.join(HERE, "artifacts", name)
                if os.path.exists(p):
                    with st.expander(title):
                        st.image(p)

    with t_first:
        st.markdown(
            "Results of the **first experiment** (original notebook, saved outputs in "
            "`original_experiment/pneumonia_train_original.ipynb`): frozen EfficientNetB0 backbone, trained on "
            "**5,218 images**, evaluated on the same 624-image Kaggle test split."
        )
        r = st.columns(5)
        r[0].metric("Accuracy", "0.894")
        r[1].metric("Precision (pneumonia)", "0.888")
        r[2].metric("Recall (pneumonia)", "0.951")
        r[3].metric("Specificity", "≈ 0.80")
        r[4].metric("ROC AUC", "0.964")
        st.caption("Exact printed values: accuracy 0.8942 · precision 0.8876 · recall 0.9513 · AUC 0.9644 · test loss 0.2408. "
                   "Per class: NORMAL precision 0.91 / recall 0.80, PNEUMONIA precision 0.89 / recall 0.95.")
        st.info("These are the results of the first experiment. The current model (previous tab) was trained "
                "with the re-implemented pipeline described on the Home page and is the one used by this app.")

    with t_about:
        st.subheader("Data", anchor=False)
        rows = [
            "| Run | Training | Validation | Test |",
            "|---|---|---|---|",
            "| First experiment (v1) | **5,218** (1,343 NORMAL / 3,875 PNEUMONIA) | 224 (112 / 112) | 624 (234 / 390) |",
        ]
        if m and "data" in m:
            d = m["data"]
            fmt = lambda s: f"{d[s]['NORMAL'] + d[s]['PNEUMONIA']:,} ({d[s]['NORMAL']:,} / {d[s]['PNEUMONIA']:,})"  # noqa: E731
            rows.append(f"| Current model (v2) | {fmt('train')} | {fmt('val')} | {fmt('test')} |")
        st.markdown("\n".join(rows))
        st.caption("Source: *Chest X-Ray Images (Pneumonia)* on Kaggle — paediatric chest X-rays (NORMAL / PNEUMONIA). "
                   "For v2 the Kaggle train and validation folders are pooled and re-split by patient; the test split is untouched.")

        st.subheader("Method", anchor=False)
        st.markdown(
            """
- **Model:** EfficientNetB0 (ImageNet weights) → GlobalAveragePooling → Dense(256) → Dropout(0.3) → sigmoid
- **Training (v2):** head trained on a frozen backbone, then the top 30 layers fine-tuned at a low learning rate
- **Imbalance:** balanced class weights · **Augmentation:** rotation, zoom, translation, contrast
- **Explainability:** Grad-CAM on EfficientNet's final feature map, computed for the predicted class
            """
        )
        st.subheader("Scope & next steps", anchor=False)
        st.markdown(
            """
- **External validation:** the data come from one paediatric centre; testing on additional datasets is the natural next step
- **Threshold tuning:** choose the decision threshold on a validation set to balance sensitivity and specificity
- **Patient-level cross-validation** for tighter performance estimates
- **Advanced Grad-CAM:** smoother, higher-resolution maps (e.g. Grad-CAM++)
- **API deployment:** wrap the model in a REST API (e.g. FastAPI) for integration into other systems such as Electronic Health Records
- **Multi-class detection:** distinguish viral from bacterial pneumonia
            """
        )
