import os
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Cancer Gene Expression Explorer", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {background: rgba(46,116,181,0.12); border: 1px solid rgba(46,116,181,0.35);
        padding: 12px 16px; border-radius: 12px;}
    button[data-baseweb="tab"] {font-size: 16px; padding: 10px 14px;}
    .hero {padding:20px 26px;border-radius:14px;background:linear-gradient(90deg,#1F3864,#2E74B5);color:white;margin-bottom:14px}
    .hero .t {font-size:30px;font-weight:700}
    .hero .s {font-size:15px;opacity:.92;margin-top:2px}
    .hero .n {font-size:13px;opacity:.8;margin-top:8px}
    </style>
    <div class="hero">
      <div class="t">🧬 Cancer Gene Expression Explorer</div>
      <div class="s">Differentially expressed genes, apoptosis, mutations and cancer vs healthy prediction</div>
      <div class="n">Hira Afzal (SP23-BSI-020) | Hamna Nasim (SP23-BSI-017) | Namra Munir Khokhar (SP23-BSI-040) | COMSATS University Islamabad</div>
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
    "BIRC5": "Survivin, inhibitor of apoptosis",
    "MCL1": "MCL-1, anti-apoptotic BCL-2 family protein",
    "BCL2L1": "BCL-xL, anti-apoptotic protein",
    "CASP8": "Caspase-8, starts death-receptor apoptosis",
    "CASP9": "Caspase-9, starts mitochondrial apoptosis",
}
PRO = {"TP53", "BAX", "BAK1", "CASP3", "CASP6", "CASP7", "CASP8", "CASP9", "BID", "BBC3", "FAS", "BAD", "BCL2L11", "APAF1", "CYCS", "DIABLO", "PMAIP1", "FADD"}
ANTI = {"BCL2", "BCL2L1", "MCL1", "XIAP", "BIRC5", "BIRC2", "BIRC3"}
REAL_FILE = "breast_cancer_gse42568.csv"

# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Data")
options = ["Example (simulated)"]
if os.path.exists(REAL_FILE):
    options.append("Breast cancer (GSE42568, real data)")
options.append("Upload my own")
source = st.sidebar.radio("Data source", options)

mutations = None
if source == "Upload my own":
    data_type = st.sidebar.radio("Values in the file are", ["Raw read counts (RNA-seq)", "Already log2 (microarray)"])
    counts_file = st.sidebar.file_uploader("Expression CSV (rows = samples, columns = genes + 'label')", type="csv")
    mut_file = st.sidebar.file_uploader("Mutations CSV (optional: sample, gene, mutation_type)", type="csv")
    if counts_file is None:
        st.info("Upload an expression CSV in the sidebar, or switch to another data source. See the **Help** tab for the format.")
        st.stop()
    data = pd.read_csv(counts_file, index_col=0)
    mutations = pd.read_csv(mut_file) if mut_file is not None else None
elif source.startswith("Breast"):
    data_type = "Already log2 (microarray)"
    data = pd.read_csv(REAL_FILE, index_col=0)
    st.sidebar.caption("Real public dataset GSE42568: 104 breast cancer and 17 normal breast tissues (Affymetrix microarray, log2 values).")
else:
    data_type = "Raw read counts (RNA-seq)"
    data = pd.read_csv("simulated_counts.csv", index_col=0)
    mutations = pd.read_csv("simulated_mutations.csv")
    st.sidebar.caption("Simulated practice data, not real patients.")
    st.sidebar.download_button("Download example counts CSV", data.to_csv().encode(), "example_counts.csv")

is_log = data_type.startswith("Already")
if source.startswith("Example"):
    st.warning("You are looking at **simulated practice data**, not real patients.")
elif source.startswith("Breast"):
    st.success("Real data: **GSE42568** (public GEO dataset), 104 breast cancer and 17 normal breast tissue samples.")

if "label" not in data.columns:
    st.error("The file needs a column named 'label'.")
    st.stop()

gene_cols = [c for c in data.columns if c != "label"]
data[gene_cols] = data[gene_cols].apply(pd.to_numeric, errors="coerce")
if data[gene_cols].isna().any().any():
    st.error("Gene columns must contain numbers only.")
    st.stop()

st.sidebar.header("2. Groups")
labels = sorted(data["label"].astype(str).unique())
if len(labels) < 2:
    st.error("The 'label' column needs at least two groups.")
    st.stop()
control = st.sidebar.selectbox("Control group", labels, index=labels.index("healthy") if "healthy" in labels else 0)
case_options = [l for l in labels if l != control]
case = st.sidebar.selectbox("Case group", case_options, index=case_options.index("cancer") if "cancer" in case_options else 0)

st.sidebar.header("3. Cut-offs")
padj_cut = st.sidebar.slider("Adjusted p-value below", 0.001, 0.1, 0.05, 0.001)
fc_cut = st.sidebar.slider("|log2 fold change| above", 0.5, 3.0, 1.0, 0.1)
default_apo = [g for g in gene_cols if g in PRO | ANTI]
apoptosis = st.sidebar.multiselect("Apoptosis genes", gene_cols, default=default_apo)

sub = data[data["label"].astype(str).isin([control, case])].copy()
sub["group"] = np.where(sub["label"].astype(str) == case, case, control)
n_case, n_control = int((sub["group"] == case).sum()), int((sub["group"] == control).sum())
if min(n_case, n_control) < 3:
    st.error("Each group needs at least 3 samples.")
    st.stop()

# ------------------------------------------------------------------ analysis
def to_log(df):
    return df.astype(float) if is_log else np.log2(df.astype(float) + 1)


log_counts = to_log(sub[gene_cols]).T          # rows = genes, columns = samples
case_cols = sub.index[sub["group"] == case]
ctrl_cols = sub.index[sub["group"] == control]


@st.cache_data
def run_deg(logc, case_idx, ctrl_idx, padj_cut, fc_cut):
    a, b = logc[list(case_idx)], logc[list(ctrl_idx)]
    res = pd.DataFrame(index=logc.index)
    res["log2FC"] = a.mean(axis=1) - b.mean(axis=1)
    res["p_value"] = np.nan_to_num(stats.ttest_ind(a, b, axis=1).pvalue, nan=1.0)
    res["padj"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["DEG"] = (res["padj"] < padj_cut) & (res["log2FC"].abs() > fc_cut)
    res["direction"] = np.where(~res["DEG"], "not significant", np.where(res["log2FC"] > 0, "up in " + case, "down in " + case))
    res["protein"] = [PROTEIN_INFO.get(g, "") for g in res.index]
    return res


result = run_deg(log_counts, tuple(case_cols), tuple(ctrl_cols), padj_cut, fc_cut)
n_deg = int(result["DEG"].sum())
n_up = int((result["DEG"] & (result["log2FC"] > 0)).sum())
n_down = int((result["DEG"] & (result["log2FC"] < 0)).sum())
top_deg = result[result["DEG"]].sort_values("padj").index.tolist()
present_apo = [g for g in apoptosis if g in result.index]

tabs = st.tabs(["📊 Overview", "🌋 DEGs", "🔥 Heatmap", "💀 Apoptosis", "🤖 Prediction", "🧪 Mutations", "ℹ️ Help"])

# ---- Overview
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", len(sub))
    c2.metric(case, n_case)
    c3.metric(control, n_control)
    c4.metric("Genes", len(gene_cols))
    if n_deg:
        st.info(f"**Key finding:** {n_deg} DEGs ({n_up} up and {n_down} down in {case}). Strongest: " + ", ".join(top_deg[:5]) + ".")
    else:
        st.info("No DEGs with the current cut-offs. Try relaxing them in the sidebar.")

    left, right = st.columns(2)
    with left:
        st.subheader("Samples in two dimensions (PCA)")
        mat = log_counts.T.to_numpy()
        top = np.argsort(mat.var(axis=0))[-500:]
        Z = mat[:, top]
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        pcs = U[:, :2] * S[:2]
        expl = (S ** 2) / np.sum(S ** 2) * 100
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for grp, col in ((case, "firebrick"), (control, "steelblue")):
            m = (sub["group"] == grp).to_numpy()
            ax.scatter(pcs[m, 0], pcs[m, 1], c=col, label=grp, s=36, alpha=0.85)
        ax.set_xlabel(f"PC1 ({expl[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({expl[1]:.0f}%)")
        ax.legend()
        ax.set_title("If the groups separate, expression differs")
        st.pyplot(fig)
    with right:
        st.subheader("Expression per sample")
        if is_log:
            vals = log_counts.median(axis=0)
            ylab = "median log2 expression"
        else:
            vals = sub[gene_cols].sum(axis=1)
            ylab = "total read counts"
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.bar(range(len(vals)), vals.values, color=np.where(sub["group"] == case, "firebrick", "steelblue"), width=1.0)
        ax.set_xlabel(f"samples (red = {case}, blue = {control})")
        ax.set_ylabel(ylab)
        ax.set_xticks([])
        st.pyplot(fig)
    st.subheader("Data table")
    st.dataframe(sub[gene_cols[:40] + ["group"]])
    if len(gene_cols) > 40:
        st.caption(f"Showing the first 40 of {len(gene_cols)} genes.")

# ---- DEGs
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    c1.metric("DEGs", n_deg)
    c2.metric(f"Up in {case}", n_up)
    c3.metric(f"Down in {case}", n_down)

    plot = result.copy()
    plot["nlp"] = -np.log10(plot["padj"] + 1e-300)
    colors = np.where(~plot["DEG"], "lightgrey", np.where(plot["log2FC"] > 0, "firebrick", "steelblue"))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(plot["log2FC"], plot["nlp"], c=colors, s=14 if len(plot) > 100 else 60, alpha=0.8)
    label_set = set(top_deg[:15]) | set(present_apo if len(plot) <= 100 else [g for g in present_apo if g in top_deg])
    if len(plot) <= 100:
        label_set = set(plot.index)
    for g in label_set:
        ax.annotate(g + ("*" if g in apoptosis else ""), (plot.loc[g, "log2FC"], plot.loc[g, "nlp"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(fc_cut, ls="--", c="black", lw=0.8)
    ax.axvline(-fc_cut, ls="--", c="black", lw=0.8)
    ax.axhline(-np.log10(padj_cut), ls="--", c="black", lw=0.8)
    ax.set_xlabel(f"log2 fold change ({case} vs {control})")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Volcano plot (red = up, blue = down, * = apoptosis gene)")
    st.pyplot(fig)

    shown = ["log2FC", "p_value", "padj", "direction"] + (["protein"] if result["protein"].astype(bool).any() else [])
    cu, cd = st.columns(2)
    with cu:
        st.subheader(f"Top genes up in {case}")
        st.dataframe(result[result["DEG"] & (result["log2FC"] > 0)].sort_values("padj").head(10)[shown].round(4))
    with cd:
        st.subheader(f"Top genes down in {case}")
        st.dataframe(result[result["DEG"] & (result["log2FC"] < 0)].sort_values("padj").head(10)[shown].round(4))
    st.subheader("All genes")
    st.dataframe(result.sort_values("padj")[shown + ["DEG"]].round(4))
    st.download_button("Download full DEG table (CSV)", result.round(6).to_csv().encode(), "deg_results.csv")

# ---- Heatmap
with tabs[2]:
    if n_deg < 2:
        st.info("Need at least 2 DEGs for a heatmap. Try relaxing the cut-offs in the sidebar.")
    else:
        top_n = min(30, n_deg)
        genes_h = result.loc[top_deg[:top_n]].sort_values("log2FC").index
        order = list(case_cols) + list(ctrl_cols)
        heat = log_counts.loc[genes_h, order]
        z = heat.sub(heat.mean(axis=1), axis=0).div(heat.std(axis=1) + 1e-9, axis=0)
        fig = plt.figure(figsize=(10, 0.28 * top_n + 2.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 12 + top_n * 0.3], hspace=0.04)
        axb = fig.add_subplot(gs[0])
        axb.imshow(np.array([[1] * len(case_cols) + [0] * len(ctrl_cols)]), aspect="auto", cmap="coolwarm_r")
        axb.set_xticks([])
        axb.set_yticks([0])
        axb.set_yticklabels(["group"])
        ax = fig.add_subplot(gs[1])
        im = ax.imshow(z, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        fig.colorbar(im, ax=ax, label="z-score (red = high, blue = low)", fraction=0.03)
        ax.set_yticks(range(len(z.index)))
        ax.set_yticklabels(z.index, fontsize=8)
        ax.set_xticks([len(case_cols) / 2 - 0.5, len(case_cols) + len(ctrl_cols) / 2 - 0.5])
        ax.set_xticklabels([case, control])
        ax.axvline(len(case_cols) - 0.5, color="black", lw=1)
        axb.set_title(f"Top {top_n} DEGs by adjusted p-value (each column is a sample)")
        st.pyplot(fig)

# ---- Apoptosis
with tabs[3]:
    if not present_apo:
        st.info("No apoptosis genes found in this data. Choose genes in the sidebar.")
    else:
        mc = log_counts.loc[present_apo, case_cols].mean(axis=1)
        mk = log_counts.loc[present_apo, ctrl_cols].mean(axis=1)
        x = np.arange(len(present_apo))
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(present_apo) + 3), 4.2))
        ax.bar(x - 0.18, mc, 0.36, label=case, color="firebrick")
        ax.bar(x + 0.18, mk, 0.36, label=control, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(present_apo, rotation=45 if len(present_apo) > 8 else 0, ha="right" if len(present_apo) > 8 else "center")
        ax.set_ylabel("mean log2 expression")
        ax.legend()
        ax.set_title("Apoptosis genes")
        st.pyplot(fig)
        table = result.loc[present_apo, ["log2FC", "padj", "direction"]].copy()
        table["role"] = ["pro-apoptotic" if g in PRO else "anti-apoptotic" if g in ANTI else "unknown" for g in table.index]
        st.dataframe(table.round(4))
        sig = table[(table["padj"] < padj_cut) & (table["log2FC"].abs() > fc_cut)]
        anti_up = [g for g in sig.index if sig.loc[g, "role"] == "anti-apoptotic" and sig.loc[g, "log2FC"] > 0]
        pro_down = [g for g in sig.index if sig.loc[g, "role"] == "pro-apoptotic" and sig.loc[g, "log2FC"] < 0]
        pro_up = [g for g in sig.index if sig.loc[g, "role"] == "pro-apoptotic" and sig.loc[g, "log2FC"] > 0]
        anti_down = [g for g in sig.index if sig.loc[g, "role"] == "anti-apoptotic" and sig.loc[g, "log2FC"] < 0]
        if len(anti_up) + len(pro_down) > len(pro_up) + len(anti_down):
            st.success(f"Apoptosis looks **suppressed** in {case}: anti-apoptotic up ({', '.join(anti_up) or 'none'}), pro-apoptotic down ({', '.join(pro_down) or 'none'}).")
        elif len(pro_up) + len(anti_down) > 0 and len(pro_up) + len(anti_down) > len(anti_up) + len(pro_down):
            st.warning(f"Apoptosis genes look **increased** in {case}: pro-apoptotic up ({', '.join(pro_up) or 'none'}), anti-apoptotic down ({', '.join(anti_down) or 'none'}).")
        elif sig.empty:
            st.info("No apoptosis gene passes both cut-offs (adjusted p-value and fold change). Try relaxing them in the sidebar.")
        else:
            st.info("Mixed pattern: some apoptosis genes go up and some go down.")
        st.caption("This is an expression-based hint, not proof. It only counts genes that pass both cut-offs in the sidebar.")

# ---- Prediction
with tabs[4]:
    X = to_log(sub[gene_cols])
    y = (sub["group"] == case).astype(int)
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        st.warning("Not enough samples per group to split into training and test sets.")
        st.stop()

    def select_genes(Xs, ys, max_genes=50):
        a, b = Xs[ys == 1], Xs[ys == 0]
        fc = a.mean() - b.mean()
        pv = np.nan_to_num(stats.ttest_ind(a, b).pvalue, nan=1.0)
        padj = pd.Series(multipletests(pv, method="fdr_bh")[1], index=Xs.columns)
        ok = padj[(padj < padj_cut) & (fc.abs() > fc_cut)].sort_values()
        sel = list(ok.index[:max_genes]) if len(ok) else list(padj.nsmallest(3).index)
        return sel

    def fit(Xs, ys):
        sel = select_genes(Xs, ys)
        mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        mdl.fit(Xs[sel], ys)
        return mdl, sel

    model_tr, sel_tr = fit(Xtr, ytr)
    pred = model_tr.predict(Xte[sel_tr])
    prob = model_tr.predict_proba(Xte[sel_tr])[:, 1]
    st.subheader("Model check on unseen test samples")
    st.caption("Genes are chosen on training samples only (no data leakage). The model uses at most 50 genes.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (test)", f"{accuracy_score(yte, pred):.2f}")
    c2.metric("ROC-AUC (test)", f"{roc_auc_score(yte, prob):.2f}" if yte.nunique() > 1 else "n/a")
    c3.metric("Test samples", len(yte))
    g1, g2 = st.columns(2)
    with g1:
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
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
        ax.set_title("Confusion matrix")
        st.pyplot(fig)
    with g2:
        if yte.nunique() > 1:
            fpr, tpr, _ = roc_curve(yte, prob)
            fig, ax = plt.subplots(figsize=(3.8, 3.4))
            ax.plot(fpr, tpr, color="firebrick", lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve")
            st.pyplot(fig)
    st.caption("Genes used: " + ", ".join(sel_tr[:20]) + (" ..." if len(sel_tr) > 20 else "") + ". With few test samples these numbers can change a lot, and on simulated data they are very high.")

    st.subheader("Predict new patients")
    final_model, sel_final = fit(X, y)
    mode = st.radio("Input", ["Pick a sample from the data", "Upload new patient CSV"], horizontal=True)
    if mode == "Pick a sample from the data":
        pick = st.selectbox("Sample", list(sub.index))
        new = sub.loc[[pick], gene_cols]
    else:
        up = st.file_uploader("New patient CSV (rows = patients, columns = genes, same type of values as the training data)", type="csv", key="newpat")
        if up is None:
            st.stop()
        new = pd.read_csv(up, index_col=0)
        missing = [g for g in sel_final if g not in new.columns]
        if missing:
            st.error("Missing gene columns: " + ", ".join(missing[:10]))
            st.stop()
    p_new = final_model.predict_proba(to_log(new[sel_final]))[:, 1]
    out = pd.DataFrame({"prediction": np.where(p_new >= 0.5, case, control), f"probability of {case}": p_new.round(3)}, index=new.index)
    st.dataframe(out)
    st.warning("Learning tool only. Not for medical use.")

# ---- Mutations
with tabs[5]:
    if mutations is None:
        st.info("No mutation data for this data source. Upload a mutation CSV (sample, gene, mutation_type) with your own data.")
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
        st.caption("Mutations come from the mutation file. They cannot be found from expression values alone.")

# ---- Help
with tabs[6]:
    st.markdown(
        """
**Expression CSV** (one row per sample):

| sample | TP53 | BAX | BCL2 | ... | label |
|---|---|---|---|---|---|
| C1 | 155 | 65 | 338 | ... | cancer |
| H1 | 192 | 139 | 131 | ... | healthy |

- First column: sample name. Other columns: gene values (numbers). Last column `label`: group of each sample (for example `healthy`, `benign`, `malignant`, `cancer`).
- Choose **Raw read counts** for RNA-seq counts, or **Already log2** for microarray data that is already log2 scaled.
- Pick the **control** and **case** groups in the sidebar. For malignant vs benign, set control = benign and case = malignant.

**Mutations CSV** (optional): columns `sample`, `gene`, `mutation_type`.

**Real breast cancer data:** GSE42568 (public, GEO) has 104 breast cancer and 17 normal breast tissue samples. It appears as a data source once the file `breast_cancer_gse42568.csv` is in the repository.

**Where the values come from:** Galaxy (featureCounts) or a GEO / TCGA table.

**Not covered:** alignment and read counting are done in Galaxy. Protein levels need proteomics data.
        """
    )
