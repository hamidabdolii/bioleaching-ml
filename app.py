# -*- coding: utf-8 -*-
"""
Bioleaching Recovery Prediction Dashboard
==========================================
Streamlit app — trains HistGB on the full dataset at startup,
then lets users predict Recovery (%) for new conditions with
SHAP explanation + conformal confidence interval.

Run locally:
    pip install streamlit shap scikit-learn pandas openpyxl matplotlib
    streamlit run app.py

Deploy free:
    Push this file + requirements.txt to GitHub, then
    connect at https://streamlit.io/cloud
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import io, time

import streamlit as st
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Bioleaching Recovery Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:700; color:#1a6b3c; margin-bottom:0;}
    .sub-title  {font-size:1.0rem; color:#555; margin-top:0; margin-bottom:1.5rem;}
    .metric-box {background:#f0f7f4; border-radius:10px; padding:12px 18px;
                 border-left:4px solid #1a6b3c; margin-bottom:8px;}
    .warn-box   {background:#fff8e1; border-radius:8px; padding:10px 14px;
                 border-left:4px solid #f9a825;}
    .good-box   {background:#e8f5e9; border-radius:8px; padding:10px 14px;
                 border-left:4px solid #2e7d32;}
    .bad-box    {background:#fce4ec; border-radius:8px; padding:10px 14px;
                 border-left:4px solid #c62828;}
    [data-testid="stSidebar"] {background:#f4fbf7;}
</style>
""", unsafe_allow_html=True)

# ── constants (must match your dataset) ────────────────────────
TARGET       = "Recovery (%)"
LEAK_COLS    = ["Glycine concentration extracted (g/L)"]
RANDOM_STATE = 42
CONF_ALPHA   = 0.10     # 90 % conformal intervals

ELEMENTS = ["Al","Au","Ba","Cd","Ce","Co","Cu","Fe","La","Li",
            "Mn","Nd","Ni","Pd","Pr","Pt","Rh","Si","Sr","Ti","Zn"]

BIOAGENTS = [
    "A_ferrooxidans","A_niger","A_niger_MM1_and_SG1","A_thiooxidans",
    "A_thiooxidans_A_ferrooxidans","A_thiooxidans_L_ferriphilum",
    "Acidithiobacillus_sp","B_megaterium","B_megaterium_A_niger",
    "C_metallidurans","C_violaceum",
    "Mixed_culture_A_caldus_L_ferriphilum_Sulfobacillus_Ferroplasma",
    "P_brassicacearum","P_fluorescens",
]

TREATMENTS = [
    "indirect_leaching","one_step","spent_medium",
    "spent_medium_ultrasound_assisted_nitric_acid_pretreatment",
    "two_step","two_step_bioleaching_aqua_regia",
    "two_step_cell_free_spent_medium","two_step_pretreatment",
    "two_step_ultrasound_assisted_nitric_acid_pretreatment",
    "two_step_untreated","two_step_without_pretreatment",
]

MEDIUMS = [
    "9K_medium","Basel_317_S_power_1pct","Basic_medium","broth_medium",
    "Cell_free","LB_medium","LB_Miller_broth","mixed_culture",
    "modified_9K_medium","Norris_nutrient_medium","Potato_Dextrose_Agar",
    "Potato_dextrose_broth_PDB","spent_medium","sucrose_medium",
]

# element property lookup (from dataset)
ELEM_PROPS = {
    "Al":{"Z":13,"mass":26.982,"density":2.70,"EN":1.61,"mp":660,  "Ered":-1.66},
    "Au":{"Z":79,"mass":196.97,"density":19.3,"EN":2.54,"mp":1064, "Ered":1.50},
    "Ba":{"Z":56,"mass":137.33,"density":3.51,"EN":0.89,"mp":727,  "Ered":-2.91},
    "Cd":{"Z":48,"mass":112.41,"density":8.65,"EN":1.69,"mp":321,  "Ered":-0.40},
    "Ce":{"Z":58,"mass":140.12,"density":6.77,"EN":1.12,"mp":799,  "Ered":-2.34},
    "Co":{"Z":27,"mass":58.933,"density":8.90,"EN":1.88,"mp":1495, "Ered":-0.28},
    "Cu":{"Z":29,"mass":63.546,"density":8.96,"EN":1.90,"mp":1085, "Ered":0.34},
    "Fe":{"Z":26,"mass":55.845,"density":7.87,"EN":1.83,"mp":1538, "Ered":-0.44},
    "La":{"Z":57,"mass":138.91,"density":6.15,"EN":1.10,"mp":920,  "Ered":-2.38},
    "Li":{"Z":3, "mass":6.941, "density":0.53,"EN":0.98,"mp":181,  "Ered":-3.04},
    "Mn":{"Z":25,"mass":54.938,"density":7.21,"EN":1.55,"mp":1246, "Ered":-1.19},
    "Nd":{"Z":60,"mass":144.24,"density":7.01,"EN":1.14,"mp":1024, "Ered":-2.32},
    "Ni":{"Z":28,"mass":58.693,"density":8.91,"EN":1.91,"mp":1455, "Ered":-0.25},
    "Pd":{"Z":46,"mass":106.42,"density":12.0,"EN":2.20,"mp":1555, "Ered":0.95},
    "Pr":{"Z":59,"mass":140.91,"density":6.77,"EN":1.13,"mp":931,  "Ered":-2.35},
    "Pt":{"Z":78,"mass":195.08,"density":21.4,"EN":2.28,"mp":1772, "Ered":1.19},
    "Rh":{"Z":45,"mass":102.91,"density":12.4,"EN":2.28,"mp":1964, "Ered":0.76},
    "Si":{"Z":14,"mass":28.086,"density":2.33,"EN":1.90,"mp":1414, "Ered":-0.86},
    "Sr":{"Z":38,"mass":87.620,"density":2.64,"EN":0.95,"mp":777,  "Ered":-2.89},
    "Ti":{"Z":22,"mass":47.867,"density":4.51,"EN":1.54,"mp":1668, "Ered":-1.63},
    "Zn":{"Z":30,"mass":65.380,"density":7.13,"EN":1.65,"mp":420,  "Ered":-0.76},
}

# ── transformers (same as pipeline) ────────────────────────────
class TargetEncoderNamed(BaseEstimator, TransformerMixin):
    def __init__(self, col="element_symbol", smoothing=10, enabled=True):
        self.col=col; self.smoothing=smoothing; self.enabled=enabled
    def fit(self, X, y):
        X=pd.DataFrame(X)
        if (not self.enabled) or self.col not in X.columns:
            self.map_={}; self.grand_=float(np.mean(y)); return self
        self.grand_=float(np.mean(y))
        s=pd.DataFrame({"y":np.asarray(y),"g":X[self.col].values})
        a=s.groupby("g")["y"].agg(["mean","count"])
        a["e"]=(a["mean"]*a["count"]+self.grand_*self.smoothing)/(a["count"]+self.smoothing)
        self.map_=a["e"].to_dict(); return self
    def transform(self, X):
        X=pd.DataFrame(X).copy()
        if self.col in X.columns:
            if self.enabled:
                X[self.col+"_te"]=X[self.col].map(self.map_).fillna(self.grand_)
            X=X.drop(columns=[self.col])
        return X

class SelectorNamed(BaseEstimator, TransformerMixin):
    def __init__(self, var_thresh=0.01, sfm_threshold="mean", random_state=42):
        self.var_thresh=var_thresh; self.sfm_threshold=sfm_threshold
        self.random_state=random_state
    def fit(self, X, y):
        X=pd.DataFrame(X); self.in_=X.columns.tolist()
        self.vt_=VarianceThreshold(self.var_thresh); self.vt_.fit(X.values)
        vt=[n for n,k in zip(self.in_,self.vt_.get_support()) if k]
        if len(vt)<5: self.out_=vt; self.simple_=True; return self
        self.simple_=False
        et=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=3,
                               random_state=self.random_state,n_jobs=-1)
        et.fit(X[vt].values,y)
        self.sfm_=SelectFromModel(et,prefit=True,threshold=self.sfm_threshold)
        out=[n for n,k in zip(vt,self.sfm_.get_support()) if k]
        self.out_=out if out else vt; return self
    def transform(self, X): return pd.DataFrame(X)[self.out_].values
    def get_feature_names_out(self,*a): return np.array(self.out_)

# ── data loading & model training (cached) ─────────────────────
@st.cache_resource(show_spinner="Training model on your dataset...")
def load_and_train(data_path):
    try:
        try:
            df = pd.read_excel(data_path, sheet_name="cleaned_preserved")
        except Exception:
            df = pd.read_excel(data_path)
    except Exception as e:
        return None, None, None, None, None, None, str(e)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    y  = df[TARGET].astype(float)

    obj = df.select_dtypes(include=["object","category"]).columns.tolist()
    num = [c for c in df.columns
           if c not in [TARGET]+obj and pd.api.types.is_numeric_dtype(df[c])]
    feat = [c for c in num if c not in LEAK_COLS]

    X_raw = df[feat].copy()
    if "element_symbol" in df.columns:
        X_raw["element_symbol"] = df["element_symbol"].values

    # stratified split (20% calibration for conformal)
    ybin = pd.cut(y, 5, labels=False)
    itr, ical = train_test_split(np.arange(len(y)), test_size=0.20,
                                 random_state=RANDOM_STATE, stratify=ybin)
    Xtr_r, Xcal_r = X_raw.iloc[itr], X_raw.iloc[ical]
    ytr, ycal = y.iloc[itr], y.iloc[ical]

    # fit preprocessing on train
    te = TargetEncoderNamed(col="element_symbol", enabled=True)
    te.fit(Xtr_r, ytr)
    imp = SimpleImputer(strategy="median")
    sel = SelectorNamed(random_state=RANDOM_STATE)

    Xtr_e  = te.transform(Xtr_r)
    Xtr_i  = pd.DataFrame(imp.fit_transform(Xtr_e), columns=Xtr_e.columns)
    sel.fit(Xtr_i, ytr)
    NAMES  = sel.out_
    Xtr_s  = pd.DataFrame(sel.transform(Xtr_i), columns=NAMES)

    # fit model
    model = HistGradientBoostingRegressor(
        max_iter=657, learning_rate=0.0662, max_depth=8,
        min_samples_leaf=5, l2_regularization=8.746,
        max_bins=64, random_state=RANDOM_STATE)
    model.fit(Xtr_s, ytr)

    # conformal calibration
    Xcal_e = te.transform(Xcal_r)
    Xcal_i = pd.DataFrame(imp.transform(Xcal_e), columns=Xcal_e.columns)
    Xcal_s = pd.DataFrame(sel.transform(Xcal_i), columns=NAMES)
    cal_res = np.abs(ycal.values - np.clip(model.predict(Xcal_s), 0, 100))
    n = len(cal_res)
    q_level = np.ceil((n+1)*(1-CONF_ALPHA))/n
    conformal_q = float(np.quantile(cal_res, min(q_level, 1.0)))

    # train metrics
    train_pred = np.clip(model.predict(Xtr_s), 0, 100)
    cal_pred   = np.clip(model.predict(Xcal_s), 0, 100)
    train_mae  = mean_absolute_error(ytr, train_pred)
    cal_mae    = mean_absolute_error(ycal, cal_pred)
    cal_r2     = r2_score(ycal, cal_pred)

    prep = (te, imp, sel, NAMES)
    meta = {"train_mae":train_mae, "cal_mae":cal_mae,
            "cal_r2":cal_r2, "conformal_q":conformal_q,
            "n_features":len(NAMES), "n_train":len(ytr),
            "features":NAMES}
    return model, prep, df, y, meta, None, None

def preprocess_row(row_df, prep):
    te, imp, sel, NAMES = prep
    Xe = te.transform(row_df)
    Xi = pd.DataFrame(imp.transform(Xe), columns=Xe.columns)
    return pd.DataFrame(sel.transform(Xi), columns=NAMES)

def build_input_row(element, bioagent, treatment, medium,
                    inoculum, init_wt, particle_size, temperature,
                    ph, pulp_density_pct, pulp_density_gl,
                    time_h, shaking, df_template):
    """Build a one-row DataFrame matching the dataset columns."""
    row = {c: 0.0 for c in df_template.columns
           if c not in [TARGET, "element_symbol"] and
           pd.api.types.is_numeric_dtype(df_template[c])}
    # process features
    row.update({
        "Inoculum (v/v%)":         inoculum,
        "Initial Wt%":             init_wt,
        "Particle size (um)":      particle_size,
        "Temperature (C)":         temperature,
        "pH":                      ph,
        "Pulp density (%w/v)":     pulp_density_pct,
        "Pulp density (g/L or %)": pulp_density_gl,
        "Bioleaching time (h)":    time_h,
        "Shaking (RPM)":           shaking,
        "Atomic No":               ELEM_PROPS[element]["Z"],
    })
    # element one-hot
    col = f"Element_{element}"
    if col in row: row[col] = 1.0
    # element properties
    ep = ELEM_PROPS[element]
    for k,col_name in [("Z","element_Z"),("mass","element_atomic_mass"),
                       ("density","element_density_g_cm3"),
                       ("EN","element_electronegativity"),
                       ("mp","element_melting_point_C"),
                       ("Ered","element_std_reduction_potential_V")]:
        if col_name in row: row[col_name] = ep[k]
    # bioagent one-hot
    ba_col = f"Bioagent_{bioagent}"
    if ba_col in row: row[ba_col] = 1.0
    # treatment one-hot
    tr_col = f"Treatment_{treatment}"
    if tr_col in row: row[tr_col] = 1.0
    # medium one-hot
    med_col = f"Medium_{medium}"
    if med_col in row: row[med_col] = 1.0

    row_df = pd.DataFrame([row])
    row_df["element_symbol"] = element
    return row_df

# ── MAIN APP ───────────────────────────────────────────────────
st.markdown('<p class="main-title">⚗️ Bioleaching Recovery Predictor</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML-based prediction of metal recovery (%) '
            'with uncertainty quantification — HistGradientBoosting + '
            'split-conformal intervals</p>', unsafe_allow_html=True)

# sidebar: data path
with st.sidebar:
    st.header("⚙️ Configuration")
    data_path = st.text_input(
        "Excel dataset path",
        value=r"C:\000 NRI NRI NRI\BL\1- DATA AND CODE\Data set Biolich-AE (HA) - EnCoded- 2.xlsx",
        help="Full path to your encoded Excel file")
    st.caption("Model trains automatically on first load. "
               "Cached until path changes.")
    st.divider()
    st.markdown("**About**")
    st.caption("Author: Hamid Abdoli\n\n"
               "Model: HistGradientBoosting (v5 champion)\n\n"
               "Uncertainty: split-conformal (90% coverage)")

# load model
model, prep, df_ref, y_ref, meta, _, err = load_and_train(data_path)

if err or model is None:
    st.error(f"❌ Could not load dataset: {err}")
    st.info("Check the path in the sidebar. The file must be the encoded Excel file.")
    st.stop()

# model metrics banner
c1,c2,c3,c4 = st.columns(4)
c1.metric("Calibration MAE", f"{meta['cal_mae']:.2f} %")
c2.metric("Calibration R²",  f"{meta['cal_r2']:.3f}")
c3.metric("90% CI half-width", f"±{meta['conformal_q']:.1f} %")
c4.metric("Features used", str(meta['n_features']))

st.divider()

# ── INPUT PANEL ────────────────────────────────────────────────
st.subheader("🔬 Enter experimental conditions")

col_a, col_b, col_c = st.columns([1,1,1])

with col_a:
    st.markdown("**Element & Biology**")
    element  = st.selectbox("Target element", ELEMENTS, index=ELEMENTS.index("Cu"))
    bioagent = st.selectbox("Bioleaching agent", BIOAGENTS)
    treatment= st.selectbox("Treatment type", TREATMENTS,
                             index=TREATMENTS.index("one_step"))
    medium   = st.selectbox("Culture medium", MEDIUMS,
                             index=MEDIUMS.index("9K_medium"))

with col_b:
    st.markdown("**Process parameters**")
    ph            = st.slider("pH",        0.5, 11.0, 2.0, 0.1)
    temperature   = st.slider("Temperature (°C)", 20, 70, 30, 1)
    time_h        = st.slider("Bioleaching time (h)", 24, 2000, 480, 24)
    shaking       = st.slider("Shaking (RPM)", 0, 300, 160, 10)

with col_c:
    st.markdown("**Material parameters**")
    particle_size    = st.slider("Particle size (µm)", 1, 500, 62, 1)
    init_wt          = st.slider("Initial Wt%", 0.1, 50.0, 10.0, 0.1)
    pulp_density_pct = st.slider("Pulp density (%w/v)", 0.5, 30.0, 10.0, 0.5)
    pulp_density_gl  = st.slider("Pulp density (g/L)", 1.0, 200.0, 100.0, 1.0)
    inoculum         = st.slider("Inoculum (v/v%)", 1.0, 20.0, 10.0, 0.5)

# show element properties
with st.expander(f"📋 {element} element properties (auto-filled)"):
    ep = ELEM_PROPS[element]
    st.table(pd.DataFrame([{
        "Atomic No":ep["Z"], "Atomic mass":ep["mass"],
        "Density (g/cm³)":ep["density"],
        "Electronegativity":ep["EN"],
        "Melting point (°C)":ep["mp"],
        "Std. reduction potential (V)":ep["Ered"]}]))

# ── PREDICTION ────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🚀 Predict Recovery", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Computing prediction..."):
        try:
            row_df = build_input_row(
                element, bioagent, treatment, medium,
                inoculum, init_wt, particle_size, temperature,
                ph, pulp_density_pct, pulp_density_gl,
                time_h, shaking, df_ref)

            X_proc = preprocess_row(row_df, prep)
            pred   = float(np.clip(model.predict(X_proc)[0], 0, 100))
            lo     = float(np.clip(pred - meta["conformal_q"], 0, 100))
            hi     = float(np.clip(pred + meta["conformal_q"], 0, 100))

            # result display
            col_r1, col_r2 = st.columns([1,2])
            with col_r1:
                # colour by confidence
                if pred >= 70:    box, emoji = "good-box",  "🟢"
                elif pred >= 40:  box, emoji = "warn-box",  "🟡"
                else:             box, emoji = "bad-box",   "🔴"

                st.markdown(f"""
                <div class="{box}">
                  <h2 style="margin:0;">{emoji} {pred:.1f} %</h2>
                  <p style="margin:0;">Predicted Recovery</p>
                  <p style="margin:4px 0 0 0; font-size:0.9rem;">
                    90% interval: [{lo:.1f} % — {hi:.1f} %]
                  </p>
                </div>""", unsafe_allow_html=True)

                st.metric("Prediction",  f"{pred:.1f} %")
                st.metric("Lower bound", f"{lo:.1f} %")
                st.metric("Upper bound", f"{hi:.1f} %")
                st.caption(f"Conformal half-width: ±{meta['conformal_q']:.1f} %")

            with col_r2:
                # gauge-style bar
                fig, ax = plt.subplots(figsize=(6, 1.8))
                cmap = plt.cm.RdYlGn
                bar = ax.barh(["Recovery"], [pred],
                              color=cmap(pred/100), height=0.5)
                ax.barh(["Recovery"], [100], color="#eeeeee", height=0.5, zorder=0)
                ax.fill_betweenx([-0.25, 0.25], lo, hi,
                                 color="gray", alpha=0.3, label="90% CI")
                ax.axvline(pred, color="black", lw=1.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Recovery (%)")
                ax.set_title(f"{element} · pH {ph} · {time_h}h · {bioagent}")
                ax.legend(loc="lower right", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # SHAP explanation
            st.subheader("🔍 Feature contributions (SHAP)")
            try:
                import shap
                explainer = shap.Explainer(model, check_additivity=False)
                sv = explainer(X_proc, check_additivity=False)
                feat_names = list(meta["features"])
                vals  = sv.values[0]
                base  = float(sv.base_values[0]) if hasattr(sv,"base_values") else pred

                shap_df = pd.DataFrame({
                    "feature": feat_names,
                    "shap":    vals,
                    "abs":     np.abs(vals)
                }).sort_values("abs", ascending=False).head(15)

                fig2, ax2 = plt.subplots(figsize=(7, 0.35*len(shap_df)+1.5))
                colors = ["#2e7d32" if v > 0 else "#c62828" for v in shap_df["shap"]]
                ax2.barh(shap_df["feature"], shap_df["shap"], color=colors)
                ax2.axvline(0, color="black", lw=0.8)
                ax2.set_xlabel("SHAP value (impact on prediction)")
                ax2.set_title(f"Top feature contributions\n"
                              f"Base: {base:.1f}% → Prediction: {pred:.1f}%")
                ax2.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

                shap_df["direction"] = ["↑ increases" if v>0
                                        else "↓ decreases" for v in shap_df["shap"]]
                st.dataframe(shap_df[["feature","shap","direction"]].rename(
                    columns={"shap":"SHAP value"}
                ).style.format({"SHAP value":"{:.3f}"}), use_container_width=True)
            except ImportError:
                st.info("Install `shap` for feature explanations: `pip install shap`")
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            import traceback; st.code(traceback.format_exc())

# ── BATCH PREDICTION ──────────────────────────────────────────
st.divider()
st.subheader("📤 Batch prediction (upload CSV)")
st.caption("Upload a CSV with columns matching the encoded dataset. "
           "The app will predict Recovery for each row.")

uploaded = st.file_uploader("Choose CSV file", type=["csv","xlsx"])
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            batch_df = pd.read_csv(uploaded)
        else:
            batch_df = pd.read_excel(uploaded)
        st.write(f"Loaded {len(batch_df)} rows, {batch_df.shape[1]} columns.")

        if "element_symbol" not in batch_df.columns:
            st.warning("Column 'element_symbol' not found — target encoding will use grand mean.")
            batch_df["element_symbol"] = "Cu"

        X_b = preprocess_row(batch_df, prep)
        preds_b = np.clip(model.predict(X_b), 0, 100)
        batch_df["Predicted_Recovery_%"] = np.round(preds_b, 2)
        batch_df["CI_lower"] = np.clip(preds_b - meta["conformal_q"], 0, 100).round(2)
        batch_df["CI_upper"] = np.clip(preds_b + meta["conformal_q"], 0, 100).round(2)

        st.dataframe(batch_df[["Predicted_Recovery_%","CI_lower","CI_upper"]].head(20))

        csv_out = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions CSV", csv_out,
                           "batch_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# ── DATASET EXPLORER ──────────────────────────────────────────
st.divider()
with st.expander("📊 Dataset explorer"):
    st.write(f"**{len(df_ref)} records** | {df_ref.shape[1]} columns")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig3, ax3 = plt.subplots(figsize=(5,3))
        ax3.hist(y_ref, bins=30, color="#1a6b3c", edgecolor="white")
        ax3.set_xlabel("Recovery (%)"); ax3.set_ylabel("Count")
        ax3.set_title("Recovery distribution")
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with col_e2:
        if "element_symbol" in df_ref.columns:
            elem_means = (df_ref.groupby("element_symbol")[TARGET]
                          .mean().sort_values(ascending=False))
            fig4, ax4 = plt.subplots(figsize=(5,3))
            ax4.bar(elem_means.index, elem_means.values, color="#1a6b3c")
            ax4.set_xlabel("Element"); ax4.set_ylabel("Mean Recovery (%)")
            ax4.set_title("Mean recovery by element")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

    if st.checkbox("Show raw data sample"):
        st.dataframe(df_ref.head(10))
