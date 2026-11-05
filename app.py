"""
Student Performance Predictor — Educational Linear Regression Lab
-----------------------------------------------------------------
Topics covered (aligned with the Supervised Learning syllabus):
  • Multiple Linear Regression & Ridge regularization
  • Train / test split (no data leakage)
  • Feature scaling & optional polynomial features
  • Evaluation: R², MAE, RMSE, 5-fold CV
  • Residual analysis
  • Coefficient interpretation (explainability)
  • Interactive what-if prediction

Run locally:  python app.py
Hugging Face Spaces: set sdk: gradio, app_file: app.py
"""

from __future__ import annotations

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_engine import FEATURE_COLS, FEATURE_HELP, StudentPerformanceEngine

# Single engine instance (data loaded once)
engine = StudentPerformanceEngine()


# ---------------------------------------------------------------------------
# Helper formatters
# ---------------------------------------------------------------------------
def _metrics_md(result) -> str:
    return f"""
### Model: **{result.model_name}**
| Metric | Value |
|--------|-------|
| Train R² | {result.r2_train:.4f} |
| Test R² | {result.r2_test:.4f} |
| Test MAE | {result.mae_test:.2f} points |
| Test RMSE | {result.rmse_test:.2f} points |
| 5-Fold CV R² (mean ± std) | {result.cv_r2_mean:.4f} ± {result.cv_r2_std:.4f} |
| Samples (train / test) | {result.n_train} / {result.n_test} |

**How to read this**
- **R²** close to 1.0 → model explains most of the variance in exam scores.
- **MAE / RMSE** → typical prediction error in exam-score points (lower is better).
- **CV R²** → stability estimate; large std means the split is sensitive.
"""


def _coef_md(result) -> str:
    rows = []
    for name, coef in sorted(result.coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        base = name.split(" ")[0] if " " in name else name
        help_text = FEATURE_HELP.get(base, "Polynomial / interaction term")
        direction = "↑ increases" if coef > 0 else "↓ decreases"
        rows.append(f"| `{name}` | **{coef:+.4f}** | {direction} score · {help_text} |")
    table = "\n".join(rows)
    return f"""
### Coefficient Interpretation (Explainability)
| Feature | Coefficient | Meaning |
|---------|-------------|---------|
{table}

**Intercept:** {result.intercept:.2f}  
(Expected score when all standardized features are at their mean, if scaling is on.)
"""


def plot_residuals(result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(result.y_test, result.y_pred, alpha=0.65, edgecolors="k", linewidths=0.3)
    lims = [min(result.y_test.min(), result.y_pred.min()) - 2,
            max(result.y_test.max(), result.y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Exam Score")
    ax.set_ylabel("Predicted Exam Score")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[1]
    ax.scatter(result.y_pred, result.residuals, alpha=0.65, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Predicted Exam Score")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_coefficients(result):
    names = list(result.coefficients.keys())
    vals = list(result.coefficients.values())
    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(names))))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Feature Coefficients (sorted by |importance|)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def train_model(algorithm, alpha, test_size, use_scaling, poly_degree):
    result = engine.train(
        algorithm=algorithm,
        alpha=float(alpha),
        test_size=float(test_size),
        use_scaling=bool(use_scaling),
        poly_degree=int(poly_degree),
    )
    metrics = _metrics_md(result)
    coefs = _coef_md(result)
    fig_res = plot_residuals(result)
    fig_coef = plot_coefficients(result)
    status = f"✅ Trained **{result.model_name}** · Test R² = {result.r2_test:.3f}"
    return status, metrics, coefs, fig_res, fig_coef


def predict_score(study_hours, attendance, previous, sleep, assignments):
    if engine.model is None:
        return "⚠️ Train a model first (Training tab).", "", None

    values = {
        "study_hours": float(study_hours),
        "attendance_pct": float(attendance),
        "previous_score": float(previous),
        "sleep_hours": float(sleep),
        "assignments_completed": float(assignments),
    }
    out = engine.predict_one(values)
    score = out["predicted_score"]

    # Simple risk / performance band
    if score >= 85:
        band = "Excellent"
    elif score >= 70:
        band = "Good"
    elif score >= 55:
        band = "Average"
    else:
        band = "Needs improvement"

    md = f"""
### Predicted Exam Score: **{score}**
**Performance band:** {band}  
**Model used:** {out['model']}
"""
    # Contribution table when available
    contrib_md = ""
    if out.get("contributions"):
        lines = ["| Feature | Contribution (approx.) |", "|---------|------------------------|"]
        for k, v in sorted(out["contributions"].items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"| `{k}` | {v:+.2f} |")
        contrib_md = "\n".join(lines) + "\n\n*Contributions are on the model’s internal scale (after any scaling / poly transform).*"

    return md, contrib_md, None


def show_data_overview():
    summary = engine.load_summary()
    corr = summary["corr_with_target"]
    corr_rows = "\n".join(
        f"| `{k}` | {v:.3f} |" for k, v in sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    md = f"""
### Dataset Overview
- **Rows:** {summary['n_rows']}
- **Target:** `{summary['target']}` (exam score 0–100)
- **Features:** {', '.join(f'`{f}`' for f in summary['features'])}

### Correlation with Exam Score
| Feature | Pearson r |
|---------|-----------|
{corr_rows}

Higher absolute correlation → stronger linear relationship with the target.
"""
    head_df = pd.DataFrame(summary["head"])
    return md, head_df


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Student Performance Predictor",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
) as demo:
    gr.Markdown(
        """
# 📚 Student Performance Predictor
**Educational Multiple Linear Regression Lab**

Covers core syllabus topics: multiple linear regression, Ridge, train/test split,  
scaling, polynomial features, R² / MAE / RMSE / CV, residual plots, and coefficient interpretation.
"""
    )

    with gr.Tabs():
        # ---- Tab 1: Data ----
        with gr.Tab("1 · Data Overview"):
            gr.Markdown("Synthetic but realistic student dataset (400 rows). Generated with a known linear relationship + noise.")
            data_btn = gr.Button("Load dataset summary", variant="primary")
            data_md = gr.Markdown()
            data_table = gr.Dataframe(label="Sample rows")
            data_btn.click(show_data_overview, outputs=[data_md, data_table])

        # ---- Tab 2: Train ----
        with gr.Tab("2 · Train & Evaluate"):
            with gr.Row():
                algorithm = gr.Radio(["linear", "ridge"], value="linear", label="Algorithm")
                alpha = gr.Slider(0.01, 50.0, value=1.0, step=0.1, label="Ridge α (ignored for Linear)")
                test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Test size")
            with gr.Row():
                use_scaling = gr.Checkbox(value=True, label="StandardScaler (fit on train only)")
                poly_degree = gr.Slider(1, 2, value=1, step=1, label="Polynomial degree (1 = linear)")
            train_btn = gr.Button("Train model", variant="primary")
            status = gr.Markdown()
            with gr.Row():
                metrics_out = gr.Markdown()
                coef_out = gr.Markdown()
            with gr.Row():
                residual_plot = gr.Plot(label="Actual vs Predicted + Residuals")
                coef_plot = gr.Plot(label="Coefficient magnitudes")

            train_btn.click(
                train_model,
                inputs=[algorithm, alpha, test_size, use_scaling, poly_degree],
                outputs=[status, metrics_out, coef_out, residual_plot, coef_plot],
            )

        # ---- Tab 3: Predict ----
        with gr.Tab("3 · What-If Predictor"):
            gr.Markdown("Enter a student profile. The model must be trained first.")
            with gr.Row():
                study_hours = gr.Slider(0.5, 14.0, value=5.5, step=0.1, label="Study hours / day")
                attendance = gr.Slider(35, 100, value=80, step=1, label="Attendance %")
            with gr.Row():
                previous = gr.Slider(20, 95, value=65, step=1, label="Previous exam score")
                sleep = gr.Slider(3.0, 10.0, value=6.5, step=0.1, label="Sleep hours / night")
                assignments = gr.Slider(0, 10, value=7, step=1, label="Assignments completed (0–10)")
            predict_btn = gr.Button("Predict exam score", variant="primary")
            pred_md = gr.Markdown()
            contrib_md = gr.Markdown()
            predict_btn.click(
                predict_score,
                inputs=[study_hours, attendance, previous, sleep, assignments],
                outputs=[pred_md, contrib_md, gr.State()],
            )

        # ---- Tab 4: Learn ----
        with gr.Tab("4 · Concepts (Syllabus Map)"):
            gr.Markdown(
                """
### What this project teaches (mapped to the syllabus)

| Syllabus Topic | Where you see it |
|----------------|------------------|
| **Multiple Linear Regression** | Default algorithm; equation \(\hat{y} = \beta_0 + \sum \beta_j x_j\) |
| **Ridge Regression (L2)** | Algorithm = Ridge; α controls shrinkage of coefficients |
| **Train / Test Split** | Slider + metrics reported separately for train & test |
| **No Data Leakage** | `StandardScaler` is fit **only** on the training fold |
| **Feature Scaling** | Checkbox; coefficients become comparable across features |
| **Polynomial Features** | Degree = 2 adds squared & interaction terms (bias–variance demo) |
| **Evaluation Metrics** | R², MAE, RMSE + 5-fold Cross-Validation R² |
| **Residual Analysis** | Residual plot (should look random around 0) |
| **Explainability** | Coefficient table + plain-English interpretation |
| **What-If / Inference** | Interactive predictor with contribution breakdown |

### Quick mental model
1. Data is split → model never sees the test set while learning.
2. Scaling (optional) puts every feature on a similar numeric scale.
3. The model learns one weight per feature (or per poly term).
4. We judge the model on **held-out** data and with cross-validation.
5. Coefficients tell us *direction* and *relative strength* of each factor.

### Suggested experiments
- Train Linear vs Ridge and compare Test R² and coefficient sizes.
- Turn scaling off and watch coefficient magnitudes become hard to compare.
- Set polynomial degree = 2 and observe whether Test R² improves or overfits.
- Change a single slider in the Predictor and watch the predicted score move.
"""
            )

    gr.Markdown(
        "---\n*Built as a clean, explainable student project · Python + scikit-learn + Gradio*"
    )


if __name__ == "__main__":
    demo.launch()
