import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Cancer Gene Expression Explorer", page_icon="🧬", layout="wide")

st.markdown(
    """
    <div style="padding:18px 24px;border-radius:12px;background:linear-gradient(90deg,#1F3864,#2E74B5);color:white;margin-bottom:12px">
      <div style="font-size:28px;font-weight:700">🧬 Cancer Gene Expression Explorer</div>
      <div style="font-size:15px;opacity:.9">DEGs, apoptosis genes, mutations and cancer vs healthy prediction from read counts</div>
      <div style="font-size:13px;opacity:.8;margin-top:6px">Hira Afzal (SP23-BSI-020) | Hamna Nasim (SP23-BSI-017) | Namra Munir Khokhar (SP23-BSI-040) | COMSATS University Islamabad</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PROTEIN_INFO = {
    "TP53": "p53, tumour suppressor; can trigger apoptosis",
    "BAX": "BAX, pro-apoptotic BCL-2 family protein",
    "BCL2": "BCL-2, anti-apoptotic protein",
    "CASP3": "Caspase-3, executes apoptosis",
    "MYC": "MYC, transcription factor driving cell growth",
    "EGFR": "Epidermal growth factor receptor, growth signalling",
    "GAPDH": "Housekeeping enzyme (glycolysis)",
    "ACTB": "Beta-actin, housekeeping structural protein",
    "VEGFA": "VEGF-A, promotes blood vessel growth",
    "CDKN1A": "p21, cell cycle inhibitor",
}
PRO_APOPTOSIS = {"TP53", "BAX", "BAK1", "CASP3", "CASP8", "CASP9", "BID", "BBC3", "FAS"}
ANTI_APOPTOSIS = {"BCL2", "BCL2L1", "MCL1", "XIAP", "BIRC5"}

# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Data")
source = st.sidebar.radio("Data source", ["Example (simulated)", "Upload my own"])

if source == "Upload my own":
    counts_file = st.sidebar.file_uploader("Read counts CSV (rows = samples, columns = genes + 'label')", type="csv")
    mut_file = st.sidebar.file_uploader("Mutations CSV (optional: sample, gene, mutation_type)", type="csv")
    if counts_file is None:
        st.info("Upload a counts CSV in the sidebar, or switch to the example data. See the **Help** tab for the format.")
        st.stop()
    data = pd.read_csv(counts_file, index_col=0)
    mutations = pd.read_csv(mut_file) if mut_file is not None else None
else:
    data = pd.read_csv("simulated_counts.csv", index_col=0)
    mutations = pd.read_csv("simulated_mutations.csv")
    st.sidebar.caption("Simulated practice data, not real patients.")
    st.sidebar.download_button("Download example counts CSV", data.to_csv().encode(), "example_counts.csv")

if "label" not in data.columns:
    st.error("The counts file needs a column named 'label'.")
    st.stop()

gene_cols = [c for c in data.columns if c != "label"]
data[gene_cols] = data[gene_cols].apply(pd.to_numeric, errors="coerce")
if data[gene_cols].isna().any().any():
    st.error("Gene columns must contain numbers only (read counts).")
    st.stop()

st.sidebar.header("2. Groups")
labels = sorted(data["label"].astype(str).unique())
if len(labels) < 2:
    st.error("The 'label' column needs at least two groups.")
    st.stop()
default_control = labels.index("healthy") if "healthy" in labels else 0
control = st.sidebar.selectbox("Control group", labels, index=default_control)
case_options = [l for l in labels if l != control]
case = st.sidebar.selectbox("Case group", case_options, index=case_options.index("cancer") if "cancer" in case_options else 0)

st.sidebar.header("3. Cut-offs")
padj_cut = st.sidebar.slider("Adjusted p-value below", 0.001, 0.1, 0.05, 0.001)
fc_cut = st.sidebar.slider("|log2 fold change| above", 0.5, 3.0, 1.0, 0.1)

default_apo = [g for g in gene_cols if g in PRO_APOPTOSIS | ANTI_APOPTOSIS]
apoptosis = st.sidebar.multiselect("Apoptosis genes", gene_cols, default=default_apo)

sub = data[data["label"].astype(str).isin([control, case])].copy()
sub["group"] = np.where(sub["label"].astype(str) == case, case, control)
n_case, n_control = (sub["group"] == case).sum(), (sub["group"] == control).sum()
if min(n_case, n_control) < 3:
    st.error("Each group needs at least 3 samples.")
    st.stop()

# ------------------------------------------------------------------ analysis
log_counts = np.log2(sub[gene_cols].T + 1)
case_cols = sub.index[sub["group"] == case]
ctrl_cols = sub.index[sub["group"] == control]


@st.cache_data
def run_deg(logc, case_idx, ctrl_idx, padj_cut, fc_cut):
    a, b = logc[list(case_idx)], logc[list(ctrl_idx)]
    res = pd.DataFrame(index=logc.index)
    res["log2FC"] = a.mean(axis=1) - b.mean(axis=1)
    res["p_value"] = stats.ttest_ind(a, b, axis=1).pvalue
    res["p_value"] = res["p_value"].fillna(1.0)
    res["padj"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["DEG"] = (res["padj"] < padj_cut) & (res["log2FC"].abs() > fc_cut)
    res["direction"] = np.where(~res["DEG"], "not significant", np.where(res["log2FC"] > 0, "up in " + case, "down in " + case))
    res["protein"] = [PROTEIN_INFO.get(g, "") for g in res.index]
    return res


result = run_deg(log_counts, tuple(case_cols), tuple(ctrl_cols), padj_cut, fc_cut)
deg_genes = result[result["DEG"]].sort_values("log2FC").index.tolist()

tabs = st.tabs(["📊 Overview", "🌋 DEGs", "🔥 Heatmap", "💀 Apoptosis", "🤖 Prediction", "🧪 Mutations", "ℹ️ Help"])

# ---- Overview
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", len(sub))
    c2.metric(case, int(n_case))
    c3.metric(control, int(n_control))
    c4.metric("Genes", len(gene_cols))
    st.subheader("Read counts per sample")
    totals = sub[gene_cols].sum(axis=1).rename("total reads (these genes)")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(len(totals)), totals.values, color=np.where(sub["group"] == case, "firebrick", "steelblue"))
    ax.set_xticks(range(len(totals)))
    ax.set_xticklabels(totals.index, rotation=90, fontsize=6)
    ax.set_ylabel("total counts")
    ax.set_title(f"Red = {case}, blue = {control}")
    st.pyplot(fig)
    st.subheader("Read counts table")
    st.dataframe(sub[gene_cols + ["group"]])

# ---- DEGs
with tabs[1]:
    up = int((result["DEG"] & (result["log2FC"] > 0)).sum())
    down = int((result["DEG"] & (result["log2FC"] < 0)).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("DEGs", int(result["DEG"].sum()))
    c2.metric(f"Up in {case}", up)
    c3.metric(f"Down in {case}", down)

    plot = result.copy()
    plot["nlp"] = -np.log10(plot["padj"] + 1e-300)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot["log2FC"], plot["nlp"], c=plot["DEG"].map({True: "red", False: "grey"}), s=60)
    for g in plot.index:
        ax.annotate(g + ("*" if g in apoptosis else ""), (plot.loc[g, "log2FC"], plot.loc[g, "nlp"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(fc_cut, ls="--", c="black", lw=0.8)
    ax.axvline(-fc_cut, ls="--", c="black", lw=0.8)
    ax.axhline(-np.log10(padj_cut), ls="--", c="black", lw=0.8)
    ax.set_xlabel(f"log2 fold change ({case} vs {control})")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Volcano plot (* = apoptosis genes)")
    st.pyplot(fig)
    st.dataframe(result.sort_values("padj").round(4))
    st.download_button("Download DEG table (CSV)", result.round(6).to_csv().encode(), "deg_results.csv")

# ---- Heatmap
with tabs[2]:
    if len(deg_genes) < 2:
        st.info("Need at least 2 DEGs for a heatmap. Try relaxing the cut-offs in the sidebar.")
    else:
        order = list(case_cols) + list(ctrl_cols)
        heat = log_counts.loc[deg_genes, order]
        z = heat.sub(heat.mean(axis=1), axis=0).div(heat.std(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(z, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        fig.colorbar(im, label="z-score (red = high, blue = low)")
        ax.set_yticks(range(len(z.index)))
        ax.set_yticklabels(z.index)
        ax.set_xticks([len(case_cols) / 2 - 0.5, len(case_cols) + len(ctrl_cols) / 2 - 0.5])
        ax.set_xticklabels([case, control])
        ax.axvline(len(case_cols) - 0.5, color="black", lw=1)
        ax.set_title("DEG heatmap (each column is a sample)")
        st.pyplot(fig)

# ---- Apoptosis
with tabs[3]:
    if not apoptosis:
        st.info("Choose apoptosis genes in the sidebar.")
    else:
        mc = log_counts.loc[apoptosis, case_cols].mean(axis=1)
        mk = log_counts.loc[apoptosis, ctrl_cols].mean(axis=1)
        x = np.arange(len(apoptosis))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.18, mc, 0.36, label=case, color="firebrick")
        ax.bar(x + 0.18, mk, 0.36, label=control, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(apoptosis)
        ax.set_ylabel("mean log2(count + 1)")
        ax.legend()
        ax.set_title("Apoptosis genes")
        st.pyplot(fig)
        table = result.loc[apoptosis, ["log2FC", "padj", "direction"]].copy()
        table["role"] = ["pro-apoptotic" if g in PRO_APOPTOSIS else "anti-apoptotic" if g in ANTI_APOPTOSIS else "unknown" for g in table.index]
        st.dataframe(table.round(4))
        anti_up = [g for g in table.index if table.loc[g, "role"] == "anti-apoptotic" and table.loc[g, "direction"].startswith("up")]
        pro_down = [g for g in table.index if table.loc[g, "role"] == "pro-apoptotic" and table.loc[g, "direction"].startswith("down")]
        pro_up = [g for g in table.index if table.loc[g, "role"] == "pro-apoptotic" and table.loc[g, "direction"].startswith("up")]
        if anti_up or pro_down:
            st.success(f"Apoptosis looks **suppressed** in {case}: anti-apoptotic up ({', '.join(anti_up) or 'none'}), pro-apoptotic down ({', '.join(pro_down) or 'none'}).")
        elif pro_up:
            st.warning(f"Apoptosis genes look **increased** in {case}: pro-apoptotic up ({', '.join(pro_up)}).")
        else:
            st.info("No clear apoptosis change with the current cut-offs.")
        st.caption("This is an expression-based hint, not proof. Real conclusions need real data and follow-up analysis.")

# ---- Prediction
with tabs[4]:
    X = np.log2(sub[gene_cols] + 1)
    y = (sub["group"] == case).astype(int)
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        st.warning("Not enough samples per group to split into training and test sets.")
        st.stop()

    def select_genes(Xs, ys):
        a, b = Xs[ys == 1], Xs[ys == 0]
        fc = a.mean() - b.mean()
        padj = multipletests(np.nan_to_num(stats.ttest_ind(a, b).pvalue, nan=1.0), method="fdr_bh")[1]
        sel = [g for g, f, p in zip(Xs.columns, fc, padj) if p < padj_cut and abs(f) > fc_cut]
        if not sel:
            sel = list(pd.Series(padj, index=Xs.columns).nsmallest(3).index)
        return sel

    def fit(Xs, ys):
        sel = select_genes(Xs, ys)
        mdl = make_pipeline(StandardScaler(), LogisticRegression())
        mdl.fit(Xs[sel], ys)
        return mdl, sel

    model_tr, sel_tr = fit(Xtr, ytr)
    pred = model_tr.predict(Xte[sel_tr])
    prob = model_tr.predict_proba(Xte[sel_tr])[:, 1]
    st.subheader("Model check (genes chosen on training samples only)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (test)", f"{accuracy_score(yte, pred):.2f}")
    c2.metric("ROC-AUC (test)", f"{roc_auc_score(yte, prob):.2f}" if yte.nunique() > 1 else "n/a")
    c3.metric("Test samples", len(yte))
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([control, case])
    ax.set_yticklabels([control, case])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.caption("Genes used: " + ", ".join(sel_tr) + ". Very high accuracy on simulated data is expected; real data is harder.")

    st.subheader("Predict new patients")
    final_model, sel_final = fit(X, y)
    mode = st.radio("Input", ["Pick a sample from the data", "Upload new patient CSV"], horizontal=True)
    if mode == "Pick a sample from the data":
        pick = st.selectbox("Sample", list(sub.index))
        new = sub.loc[[pick], gene_cols]
    else:
        up = st.file_uploader("New patient counts CSV (rows = patients, columns = genes)", type="csv", key="newpat")
        if up is None:
            st.stop()
        new = pd.read_csv(up, index_col=0)
        missing = [g for g in sel_final if g not in new.columns]
        if missing:
            st.error("Missing gene columns: " + ", ".join(missing))
            st.stop()
    p_new = final_model.predict_proba(np.log2(new[sel_final] + 1))[:, 1]
    out = pd.DataFrame({"prediction": np.where(p_new >= 0.5, case, control), f"probability of {case}": p_new.round(3)}, index=new.index)
    st.dataframe(out)
    st.warning("Demo tool for learning. Not for medical use.")

# ---- Mutations
with tabs[5]:
    if mutations is None:
        st.info("No mutation file uploaded. Upload a CSV with columns: sample, gene, mutation_type.")
    elif not {"sample", "gene"}.issubset(mutations.columns):
        st.error("Mutation file needs the columns 'sample' and 'gene'.")
    else:
        m = mutations[mutations["sample"].isin(sub.index)].copy()
        m["group"] = m["sample"].map(sub["group"])
        freq = m.groupby(["gene", "group"])["sample"].nunique().unstack(fill_value=0)
        for g in (case, control):
            if g not in freq.columns:
                freq[g] = 0
        pct = pd.DataFrame({case: freq[case] / n_case * 100, control: freq[control] / n_control * 100}).round(1)
        st.subheader("Samples with a mutation, by gene (%)")
        if len(pct):
            st.bar_chart(pct)
        st.dataframe(pct)
        st.subheader("Mutation list")
        st.dataframe(m)
        st.caption("Mutations come from the uploaded mutation file. They cannot be found from read counts alone.")

# ---- Help
with tabs[6]:
    st.markdown(
        """
**Counts CSV** (one row per sample):

| sample | TP53 | BAX | BCL2 | ... | label |
|---|---|---|---|---|---|
| C1 | 155 | 65 | 338 | ... | cancer |
| H1 | 192 | 139 | 131 | ... | healthy |

- First column: sample name. Other columns: gene read counts (numbers). Last column `label`: group of each sample (for example `healthy`, `benign`, `malignant`, `cancer`).
- Pick the **control** and **case** groups in the sidebar. For malignant vs benign, set control = benign and case = malignant.

**Mutations CSV** (optional): columns `sample`, `gene`, `mutation_type`.

**Where the counts come from:** Galaxy (featureCounts) or a GEO / TCGA count table.

**Not covered:** alignment and read counting are done in Galaxy. Protein levels need proteomics data. This app shows protein names for known genes only.
        """
    )
