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

st.set_page_config(page_title="Cancer Gene Expression Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top: 3.8rem; padding-bottom: 1rem;}
    [data-testid="stVerticalBlock"] {gap: .6rem;}
    [data-testid="stMain"] hr {margin: .5rem 0;}
    [data-testid="stMain"] .stButton > button {padding: .15rem .5rem; min-height: 0; font-size: .85rem;}
    [data-testid="stMain"] h2, [data-testid="stMain"] h3 {font-size: 1.05rem !important; margin: .2rem 0 .3rem 0 !important; padding: 0 !important;}
    [data-testid="stMain"] h4 {font-size: .95rem !important; margin: .2rem 0 !important;}
    [data-testid="stMetric"] {background: rgba(46,116,181,0.12); border: 1px solid rgba(46,116,181,0.35);
        padding: 6px 12px; border-radius: 10px;}
    [data-testid="stMetricValue"] {font-size: 1.25rem;}
    [data-testid="stMetricLabel"] {font-size: .8rem;}
    button[data-baseweb="tab"] {font-size: 15px; padding: 8px 12px;}
    .hero {padding:12px 14px;border-radius:12px;background:linear-gradient(90deg,#1F3864,#2E74B5);color:white;margin-bottom:10px}
    .hero .t {font-size:17px;font-weight:700;line-height:1.25}
    .hero .s {font-size:12px;opacity:.92;margin-top:4px}
    .hero .n {font-size:11px;opacity:.8;margin-top:8px;line-height:1.5}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="hero">
      <div class="t">🧬 Cancer Gene Expression Explorer</div>
      <div class="s">Differentially expressed genes, apoptosis, mutations and cancer vs healthy prediction</div>
      <div class="n">Hira Afzal (SP23-BSI-020)<br>Hamna Nasim (SP23-BSI-017)<br>Namra Munir Khokhar (SP23-BSI-040)<br>COMSATS University Islamabad</div>
    </div>
    """,
    unsafe_allow_html=True,
)

def show_fig(fig):
    """Show a figure at its own (small) size instead of stretching it over the whole column."""
    try:
        st.pyplot(fig, width="content")                  # newer Streamlit
    except TypeError:
        try:
            st.pyplot(fig, use_container_width=False)    # older Streamlit
        except TypeError:
            st.pyplot(fig)
    plt.close(fig)


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
    st.caption("⚠️ You are looking at **simulated practice data**, not real patients.")
elif source.startswith("Breast"):
    st.caption("✅ Real data: **GSE42568** (public GEO dataset), 104 breast cancer and 17 normal breast tissue samples.")

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

# ------------------------------------------------------------------ pathways, sequences, variants
PROLIF = {"MKI67": "promote", "PCNA": "promote", "TOP2A": "promote", "MCM2": "promote", "MYC": "promote",
          "EGFR": "promote", "ERBB2": "promote", "KRAS": "promote", "PIK3CA": "promote", "AKT1": "promote",
          "MTOR": "promote", "VEGFA": "promote", "FOXM1": "promote", "TYMS": "promote", "PTEN": "inhibit"}
CYCLE = {"CCND1": "promote", "CCNE1": "promote", "CCNA2": "promote", "CCNB1": "promote", "CDK1": "promote",
         "CDK2": "promote", "CDK4": "promote", "CDK6": "promote", "E2F1": "promote", "PLK1": "promote",
         "AURKA": "promote", "AURKB": "promote", "CDC20": "promote", "MDM2": "promote", "CDKN1A": "inhibit",
         "CDKN1B": "inhibit", "CDKN2A": "inhibit", "RB1": "inhibit", "TP53": "inhibit", "CHEK2": "inhibit", "ATM": "inhibit"}

_B = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: _AA[16 * i + 4 * j + k] for i, a in enumerate(_B) for j, b in enumerate(_B) for k, c in enumerate(_B)}


@st.cache_data(show_spinner="Loading sequence from Ensembl...", ttl=3600)
def _fetch_cds_ok(symbol):
    """Raises on failure, so a failed download is NOT cached (before, one failure blocked the gene for 1 hour)."""
    import time
    import requests
    base = "https://rest.ensembl.org"
    last = "unknown error"
    for attempt in range(3):
        try:
            r = requests.get(f"{base}/lookup/symbol/homo_sapiens/{symbol}", params={"expand": 1},
                             headers={"Content-Type": "application/json", "User-Agent": "cancer-expression-explorer"}, timeout=30)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1)))
                last = "Ensembl is busy (rate limit), try again in a moment"
                continue
            r.raise_for_status()
            trs = [t for t in r.json().get("Transcript", []) if t.get("Translation")]
            canon = [t for t in trs if t.get("is_canonical") == 1] or trs
            if not canon:
                raise ValueError("no protein-coding transcript found")
            tid = canon[0]["id"]
            q = requests.get(f"{base}/sequence/id/{tid}", params={"type": "cds"},
                             headers={"Content-Type": "text/plain", "User-Agent": "cancer-expression-explorer"}, timeout=30)
            q.raise_for_status()
            return q.text.strip().upper(), tid
        except ValueError:
            raise
        except Exception as e:
            last = str(e)
            time.sleep(1)
    raise RuntimeError(last)


def fetch_cds(symbol):
    """Reference coding sequence (CDS) of the canonical human transcript, from the Ensembl REST service."""
    try:
        cds, tid = _fetch_cds_ok(symbol)
        return cds, tid, None
    except Exception as e:
        return None, None, str(e)


def apply_variants(cds, gm):
    """Apply single-base substitutions (CDS position, ref, alt) per patient. Indels are not handled."""
    seqs, rows, notes = {}, [], []
    for smp, grp in gm.groupby("sample"):
        s = list(cds)
        for _, v in grp.iterrows():
            try:
                pos, ref, alt = int(v["position"]), str(v["ref"]).upper(), str(v["alt"]).upper()
            except Exception:
                notes.append(f"{smp}: unreadable variant row")
                continue
            if not (1 <= pos <= len(cds)) or len(ref) != 1 or len(alt) != 1 or cds[pos - 1] != ref:
                notes.append(f"{smp}: position {pos} ref {ref} does not match the reference CDS, skipped")
                continue
            s[pos - 1] = alt
            ci = (pos - 1) // 3
            c_ref = cds[ci * 3: ci * 3 + 3]
            c_alt = list(c_ref)
            c_alt[(pos - 1) % 3] = alt
            c_alt = "".join(c_alt)
            a1, a2 = CODON.get(c_ref, "?"), CODON.get(c_alt, "?")
            effect = "synonymous" if a1 == a2 else "nonsense (stop gained)" if a2 == "*" else "stop lost" if a1 == "*" else "missense"
            rows.append({"patient": smp, "position": pos, "change": f"{ref}>{alt}", "codon": f"{c_ref}>{c_alt}",
                         "amino acid": f"{a1}{ci + 1}{a2}", "effect": effect})
        seqs[smp] = "".join(s)
    return seqs, pd.DataFrame(rows), notes


def aln_html(ref, seqs, groups, width=60):
    """Alignment as HTML: reference on top, patients below, differences in red, * = identical in all rows."""
    import html
    names = ["Reference"] + [f"{i} ({groups.get(i, '')})" for i in seqs]
    rows = [ref] + list(seqs.values())
    pad = max(len(n) for n in names)
    out = []
    for st0 in range(0, len(ref), width):
        seg = [r[st0:st0 + width] for r in rows]
        out.append(f"{'':<{pad}}  {st0 + 1}")
        for n, sgm in zip(names, seg):
            cells = "".join(f'<span style="background:#c0392b;color:#fff">{c}</span>' if (n != "Reference" and c != seg[0][k]) else c
                            for k, c in enumerate(sgm))
            out.append(f"{html.escape(n.ljust(pad))}  {cells}")
        if len(rows) > 1:
            out.append(f"{'':<{pad}}  " + "".join("*" if all(r[k] == seg[0][k] for r in seg) else " " for k in range(len(seg[0]))) + "&#8203;")
        out.append("&#8203;")  # never a truly blank line: a blank line ends the HTML block in st.markdown
    return '<div style="overflow-x:auto"><pre style="font-size:12px;line-height:1.35">' + "\n".join(out) + "</pre></div>"


def to_fasta(entries):
    out = []
    for h, s in entries:
        out.append(">" + h)
        out += [s[i:i + 70] for i in range(0, len(s), 70)]
    return "\n".join(out) + "\n"


TREE_SPECIES = {"pan_troglodytes": "Chimpanzee", "macaca_mulatta": "Macaque", "mus_musculus": "Mouse",
                "rattus_norvegicus": "Rat", "canis_lupus_familiaris": "Dog", "bos_taurus": "Cow",
                "gallus_gallus": "Chicken", "xenopus_tropicalis": "Frog", "danio_rerio": "Zebrafish"}


@st.cache_data(show_spinner="Loading orthologs from Ensembl and measuring distances...", ttl=3600)
def _species_tree_data(symbol):
    """Human protein + its orthologs (Ensembl) -> distance matrix (1 - identity). Raises on failure, so failures are not cached."""
    import time
    import requests
    from Bio import Align
    from Bio.Align import substitution_matrices
    homs, last = None, "unknown error"
    for attempt in range(3):
        try:
            r = requests.get(f"https://rest.ensembl.org/homology/symbol/homo_sapiens/{symbol}",
                             params={"type": "orthologues", "sequence": "protein", "aligned": 0, "target_species": list(TREE_SPECIES)},
                             headers={"Content-Type": "application/json", "User-Agent": "cancer-expression-explorer"}, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1)))
                last = "Ensembl is busy (rate limit), try again in a moment"
                continue
            r.raise_for_status()
            data = r.json().get("data", [])
            homs = data[0].get("homologies", []) if data else []
            break
        except Exception as e:
            last = str(e)
            time.sleep(1)
    if homs is None:
        raise RuntimeError(last)
    seqs = {}
    for h in homs:
        if "Human" not in seqs:
            seqs["Human"] = h["source"]["align_seq"].replace("-", "")
        sp = h["target"]["species"]
        if sp in TREE_SPECIES and TREE_SPECIES[sp] not in seqs:
            seqs[TREE_SPECIES[sp]] = h["target"]["align_seq"].replace("-", "")
    if len(seqs) < 3:
        raise ValueError("fewer than 3 sequences (human + orthologs) found for this gene")
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    names = list(seqs)
    lower = []
    for i, a in enumerate(names):
        row = []
        for b in names[:i]:
            x, y = seqs[a], seqs[b]
            aln = aligner.align(x, y)[0]
            same = sum(1 for (a0, a1), (b0, b1) in zip(*aln.aligned) for u, v in zip(x[a0:a1], y[b0:b1]) if u == v)
            row.append(round(1 - same / max(len(x), len(y)), 4))
        lower.append(row + [0])
    return names, lower


def _tree_from_matrix(names, lower):
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    tree = DistanceTreeConstructor().nj(DistanceMatrix(names, lower))
    try:
        tree.root_at_midpoint()
    except Exception:
        pass
    for c in tree.find_clades():
        if c.branch_length is not None and c.branch_length < 0:
            c.branch_length = 0.0
    return tree


def _draw_tree(tree, title, xlabel):
    from Bio import Phylo
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    Phylo.draw(tree, axes=ax, do_show=False, label_func=lambda c: c.name if c.is_terminal() else "", show_confidence=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("")
    ax.set_yticks([])
    show_fig(fig)


def species_tree_ui(g):
    """Phylogenetic tree of one gene's protein across species (human + orthologs from Ensembl)."""
    st.caption("Evolutionary tree of the human protein and its orthologs in other species (Ensembl). Short branches = very similar, so the gene is well conserved.")
    try:
        names, lower = _species_tree_data(g)
        tree = _tree_from_matrix(names, lower)
    except Exception as e:
        msg = str(e)
        if "Bio" in msg:
            msg += "  (add `biopython` to requirements.txt)"
        st.error(f"Could not build the species tree: {msg}")
        return
    _draw_tree(tree, f"{g}: protein tree", "distance (1 - protein identity)")


def patient_tree_ui(g, cds, seqs, groups):
    """Tree of the reference and the patients that carry a variant in this gene (differences in the coding sequence)."""
    if not cds or not seqs:
        st.info("A patient tree needs the reference sequence and patients with variants (mutation CSV with position, ref, alt).")
        return
    pats = list(seqs)[:40]
    names = ["Reference"] + [f"{i} ({groups.get(i, '')})" for i in pats]
    rows = [cds] + [seqs[i] for i in pats]
    if len(names) < 3:
        st.info("At least 2 patients with variants are needed to draw a tree.")
        return
    lower = [[round(sum(u != v for u, v in zip(rows[i], rows[j])) / len(cds), 5) for j in range(i)] + [0] for i in range(len(rows))]
    try:
        _draw_tree(_tree_from_matrix(names, lower), f"{g}: patient tree", "fraction of bases that differ")
    except Exception as e:
        st.error(f"Could not build the patient tree: {e}")


def gene_panel(g, key):
    """Everything about one gene, in 3 small tabs: expression, sequence + alignment (MSA), SNP verdict."""
    role = "pro-apoptotic" if g in PRO else "anti-apoptotic" if g in ANTI else \
        "cell-cycle promoting" if CYCLE.get(g) == "promote" else "cell-cycle inhibiting" if CYCLE.get(g) == "inhibit" else \
        "proliferation promoting" if PROLIF.get(g) == "promote" else "proliferation inhibiting" if PROLIF.get(g) == "inhibit" else "n/a"
    r = result.loc[g]
    info = PROTEIN_INFO.get(g, "")
    st.markdown(f"**{g}** <span style='font-size:13px;opacity:.8'>{info}</span>", unsafe_allow_html=True)
    st.caption(f"log2 fold change {r['log2FC']:.2f}  |  adjusted p-value {r['padj']:.3g}  |  {r['direction']}  |  {role}")

    gm = None
    if mutations is not None and {"sample", "gene"}.issubset(mutations.columns):
        gm = mutations[(mutations["gene"].astype(str) == g) & mutations["sample"].isin(sub.index)]
    has_pos = gm is not None and {"position", "ref", "alt"}.issubset(gm.columns)
    groups = sub["group"].to_dict()
    applied = pd.DataFrame()
    cds, seqs = None, {}

    t_expr, t_seq, t_snp, t_tree = st.tabs(["📈 Expression", "🧬 Sequence & MSA", "🔬 SNP check", "🌳 Phylogenetic tree"])

    # ---- expression
    with t_expr:
        e1, e2 = st.columns([2, 3])
        with e1:
            fig, ax = plt.subplots(figsize=(3.0, 2.6))
            bp = ax.boxplot([log_counts.loc[g, ctrl_cols].values, log_counts.loc[g, case_cols].values], patch_artist=True, widths=0.55)
            for patch, col in zip(bp["boxes"], ["steelblue", "firebrick"]):
                patch.set_facecolor(col)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([control, case])
            ax.set_ylabel("log2 expression")
            ax.set_title(g)
            show_fig(fig)
        with e2:
            pat = pd.DataFrame({"patient": sub.index, "group": sub["group"].values,
                                "log2 expression": log_counts.loc[g, sub.index].values.round(3)}).sort_values(["group", "patient"])
            st.dataframe(pat, hide_index=True, height=250)

    # ---- sequence + alignment (MSA)
    with t_seq:
        st.caption("Reference sequence comes from Ensembl. Patient sequences are built only if a mutation file with position, ref and alt is uploaded. "
                   "The alignment (MSA) is done inside this app, no script to download.")
        cds = None
        if st.checkbox("Load reference sequence from Ensembl (needs internet)", value=True, key=f"{key}_{g}_seq"):
            cds, tid, err = fetch_cds(g)
            if err:
                st.error(f"Could not load the sequence from Ensembl: {err}")
            else:
                st.caption(f"Ensembl canonical transcript {tid}, coding sequence {len(cds)} bp")
        if not cds:
            pasted = st.text_area("Or paste the reference coding sequence here (A/C/G/T)", key=f"{key}_{g}_paste", height=80)
            clean = "".join(ch for ch in pasted.upper() if ch in "ACGT")
            if len(clean) >= 30:
                cds = clean
                st.caption(f"Using the pasted sequence, {len(cds)} bp")
        if cds:
            seqs = {}
            if has_pos and len(gm):
                seqs, applied, notes = apply_variants(cds, gm)
                for n in notes[:5]:
                    st.warning(n)
            show = {i: seqs[i] for i in list(seqs)[:40]}
            if show:
                st.markdown(aln_html(cds, show, groups), unsafe_allow_html=True)
                st.caption(f"Alignment of the reference and {len(show)} patient(s) carrying a variant. Red = differs from the reference. "
                           "Patients not listed have the reference sequence.")
            else:
                st.info("No patient variants for this gene in the data, so there is nothing to align. The reference sequence is shown below.")
                st.markdown(aln_html(cds, {}, groups), unsafe_allow_html=True)
            entries = [(f"{g}_reference", cds)] + [(f"{i}|{groups[i]}|{g}", seqs.get(i, cds)) for i in sub.index] if has_pos else [(f"{g}_reference", cds)]
            try:
                st.download_button("Download sequences (FASTA)", to_fasta(entries), f"{g}_sequences.fasta", key=f"{key}_{g}_fa", on_click="ignore")
            except TypeError:
                st.download_button("Download sequences (FASTA)", to_fasta(entries), f"{g}_sequences.fasta", key=f"{key}_{g}_fa")

    # ---- SNP verdict
    with t_snp:
        if gm is None:
            st.warning("This data source has no mutation/SNP data, so this cannot be answered here. "
                       "Expression alone can show a gene is up or down, but not which sequence change causes it. "
                       "Upload a mutation CSV (sample, gene, position, ref, alt) to test it.")
        else:
            carriers = set(gm["sample"])
            a = len([x for x in carriers if groups.get(x) == case])
            b = len([x for x in carriers if groups.get(x) == control])
            _, p = stats.fisher_exact([[a, n_case - a], [b, n_control - b]])
            st.write(f"Patients with a variant in **{g}**: {case} **{a}/{n_case}** ({a / n_case * 100:.0f}%), {control} **{b}/{n_control}** ({b / n_control * 100:.0f}%). Fisher exact p = {p:.3g}.")
            if len(applied):
                st.dataframe(applied, hide_index=True, height=200)
                harmful = applied["effect"].isin(["missense", "nonsense (stop gained)", "stop lost"]).any()
            else:
                harmful = None
            if a + b == 0:
                st.success(f"No variant of {g} found in these patients, so a SNP in this gene does not explain the expression change here.")
            elif p < 0.05 and a / n_case > b / n_control:
                st.error(f"Variants in {g} are significantly more common in {case}: a SNP in this gene may be involved." +
                         (" Some change the protein (missense/nonsense)." if harmful else ""))
            else:
                st.info(f"Variants exist in {g}, but they are not significantly enriched in {case}. This data does not show a SNP responsible.")
            st.caption("Association only, not proof of cause. Variants are assumed present in all copies. Insertions and deletions are not handled.")


    # ---- phylogenetic tree
    with t_tree:
        sp_tab, pt_tab = st.tabs(["🌍 Species tree", "🧑 Patient tree"])
        with sp_tab:
            if st.checkbox("Build the species tree (needs internet)", value=True, key=f"{key}_{g}_tree"):
                species_tree_ui(g)
        with pt_tab:
            patient_tree_ui(g, cds, seqs, groups)


# Gene details open in a pop-up, so the page stays clean.
if hasattr(st, "dialog"):
    @st.dialog("Gene details", width="large")
    def show_gene(g, key):
        gene_panel(g, key)
else:  # older Streamlit without pop-ups: fall back to an open box
    def show_gene(g, key):
        with st.expander(f"{g} details", expanded=True):
            gene_panel(g, key)


def gene_buttons(genes, key):
    """One small button per gene. A click opens that gene's details in a pop-up."""
    st.caption("Click a gene to open its details (expression, sequence, alignment, SNP check) in a pop-up.")
    per_row = 8
    for start in range(0, len(genes), per_row):
        cols = st.columns(per_row)
        for col, g in zip(cols, genes[start:start + per_row]):
            if col.button(g, key=f"{key}_btn_{g}", use_container_width=True):
                show_gene(g, key)


def popup(label, title, render, key, width="large"):
    """A button that opens render() in a pop-up (falls back to an open box on old Streamlit)."""
    if st.button(label, key=key):
        if hasattr(st, "dialog"):
            st.dialog(title, width=width)(render)()
        else:
            with st.expander(title, expanded=True):
                render()


tabs = st.tabs(["📊 Overview", "🌋 DEGs", "🔥 Heatmap", "🤖 Prediction", "🧪 Mutations", "💀 Apoptosis", "🔁 Proliferation", "🔄 Cell cycle", "ℹ️ Help"])

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
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for grp, col in ((case, "firebrick"), (control, "steelblue")):
            m = (sub["group"] == grp).to_numpy()
            ax.scatter(pcs[m, 0], pcs[m, 1], c=col, label=grp, s=28, alpha=0.85)
        ax.set_xlabel(f"PC1 ({expl[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({expl[1]:.0f}%)")
        ax.legend(fontsize=8)
        ax.set_title("If the groups separate, expression differs", fontsize=10)
        show_fig(fig)
    with right:
        st.subheader("Expression per sample")
        if is_log:
            vals = log_counts.median(axis=0)
            ylab = "median log2 expression"
        else:
            vals = sub[gene_cols].sum(axis=1)
            ylab = "total read counts"
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.bar(range(len(vals)), vals.values, color=np.where(sub["group"] == case, "firebrick", "steelblue"), width=1.0)
        ax.set_xlabel(f"samples (red = {case}, blue = {control})")
        ax.set_ylabel(ylab)
        ax.set_xticks([])
        show_fig(fig)

    def _data_table():
        st.dataframe(sub[gene_cols[:40] + ["group"]], height=420)
        if len(gene_cols) > 40:
            st.caption(f"Showing the first 40 of {len(gene_cols)} genes.")

    popup("📋 View data table", "Data table", _data_table, "pop_data")

# ---- DEGs
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    c1.metric("DEGs", n_deg)
    c2.metric(f"Up in {case}", n_up)
    c3.metric(f"Down in {case}", n_down)

    shown = ["log2FC", "p_value", "padj", "direction"] + (["protein"] if result["protein"].astype(bool).any() else [])
    vcol, tcol = st.columns([3, 2])
    with vcol:
        plot = result.copy()
        plot["nlp"] = -np.log10(plot["padj"] + 1e-300)
        colors = np.where(~plot["DEG"], "lightgrey", np.where(plot["log2FC"] > 0, "firebrick", "steelblue"))
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
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
        ax.set_title("Volcano plot (red = up, blue = down, * = apoptosis gene)", fontsize=10)
        show_fig(fig)
    with tcol:
        t_up, t_down = st.tabs([f"⬆️ Up in {case}", f"⬇️ Down in {case}"])
        with t_up:
            st.dataframe(result[result["DEG"] & (result["log2FC"] > 0)].sort_values("padj").head(10)[shown].round(4), height=330)
        with t_down:
            st.dataframe(result[result["DEG"] & (result["log2FC"] < 0)].sort_values("padj").head(10)[shown].round(4), height=330)

    def _all_genes():
        st.dataframe(result.sort_values("padj")[shown + ["DEG"]].round(4), height=420)

    b1, b2 = st.columns(2)
    with b1:
        popup("📋 All genes table", "All genes", _all_genes, "pop_all")
    with b2:
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
        fig = plt.figure(figsize=(7.5, 0.22 * top_n + 2))
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
        show_fig(fig)

# ---- Apoptosis
with tabs[5]:
    if not present_apo:
        st.info("No apoptosis genes found in this data. Choose genes in the sidebar.")
    else:
        mc = log_counts.loc[present_apo, case_cols].mean(axis=1)
        mk = log_counts.loc[present_apo, ctrl_cols].mean(axis=1)
        x = np.arange(len(present_apo))
        table = result.loc[present_apo, ["log2FC", "padj", "direction"]].copy()
        table["role"] = ["pro-apoptotic" if g in PRO else "anti-apoptotic" if g in ANTI else "unknown" for g in table.index]
        ca, cb = st.columns([3, 2])
        with ca:
            fig, ax = plt.subplots(figsize=(max(4.5, 0.4 * len(present_apo) + 2), 3.2))
            ax.bar(x - 0.18, mc, 0.36, label=case, color="firebrick")
            ax.bar(x + 0.18, mk, 0.36, label=control, color="steelblue")
            ax.set_xticks(x)
            ax.set_xticklabels(present_apo, rotation=45 if len(present_apo) > 8 else 0, ha="right" if len(present_apo) > 8 else "center")
            ax.set_ylabel("mean log2 expression")
            ax.legend(fontsize=8)
            ax.set_title("Apoptosis genes", fontsize=10)
            show_fig(fig)
        with cb:
            st.dataframe(table.round(4), height=300)
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
        def _tree_popup():
            tg = st.selectbox("Gene", present_apo, key="tree_gene")
            species_tree_ui(tg)

        popup("🌳 Phylogenetic tree", "Phylogenetic tree", _tree_popup, "pop_tree")
        st.markdown("---")
        gene_buttons(present_apo, "apo")

# ---- Mutations
with tabs[4]:
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
        mc1, mc2 = st.columns([3, 2])
        with mc1:
            if len(pct):
                st.bar_chart(pct, height=260)
        with mc2:
            st.dataframe(pct, height=260)

        def _mut_list():
            st.dataframe(m, height=420)

        popup("📋 Mutation list", "Mutation list", _mut_list, "pop_mut")
        st.caption("Mutations come from the mutation file. They cannot be found from expression values alone.")

def render_pathway(title, genes, key, text_up, text_down):
    present = [g for g in genes if g in result.index]
    if not present:
        st.info(f"None of the {title} genes are in this data.")
        return
    tb = result.loc[present, ["log2FC", "padj", "direction"]].copy()
    tb["role"] = [genes[g] for g in present]
    mc = log_counts.loc[present, case_cols].mean(axis=1)
    mk = log_counts.loc[present, ctrl_cols].mean(axis=1)
    x = np.arange(len(present))
    pc, pt = st.columns([3, 2])
    with pc:
        fig, ax = plt.subplots(figsize=(max(4.5, 0.4 * len(present) + 2), 3.2))
        ax.bar(x - 0.18, mc, 0.36, label=case, color="firebrick")
        ax.bar(x + 0.18, mk, 0.36, label=control, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=45, ha="right")
        ax.set_ylabel("mean log2 expression")
        ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        show_fig(fig)
    with pt:
        st.dataframe(tb.round(4), height=300)
    sg = tb[(tb["padj"] < padj_cut) & (tb["log2FC"].abs() > fc_cut)]
    on = [g for g in sg.index if (sg.loc[g, "role"] == "promote" and sg.loc[g, "log2FC"] > 0) or (sg.loc[g, "role"] == "inhibit" and sg.loc[g, "log2FC"] < 0)]
    off = [g for g in sg.index if (sg.loc[g, "role"] == "promote" and sg.loc[g, "log2FC"] < 0) or (sg.loc[g, "role"] == "inhibit" and sg.loc[g, "log2FC"] > 0)]
    if sg.empty:
        st.info(f"No {title} gene passes both cut-offs. Try relaxing them in the sidebar.")
    elif len(on) > len(off):
        st.success(f"{text_up} in {case}: " + ", ".join(on) + ".")
    elif len(off) > len(on):
        st.warning(f"{text_down} in {case}: " + ", ".join(off) + ".")
    else:
        st.info("Mixed pattern: some genes push the pathway up and some push it down.")
    st.caption("Expression-based hint only. 'promote' genes speed the pathway up, 'inhibit' genes slow it down.")
    st.markdown("---")
    gene_buttons(present, key)


# ---- Proliferation
with tabs[6]:
    st.subheader("Cell proliferation genes")
    render_pathway("cell proliferation", PROLIF, "pro", "Proliferation looks more active", "Proliferation looks less active")

# ---- Cell cycle
with tabs[7]:
    st.subheader("Cell cycle pathway genes")
    render_pathway("cell cycle", CYCLE, "cyc", "Cell cycle looks more active (more division)", "Cell cycle looks less active")

# ---- Help
with tabs[8]:
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

**Mutations CSV** (optional): columns `sample`, `gene`, `mutation_type`. For sequences and SNP checks also add `position` (coding-sequence position), `ref` and `alt` (single-base changes).

**Real breast cancer data:** GSE42568 (public, GEO) has 104 breast cancer and 17 normal breast tissue samples. It appears as a data source once the file `breast_cancer_gse42568.csv` is in the repository.

**Where the values come from:** Galaxy (featureCounts) or a GEO / TCGA table.

**Not covered:** alignment and read counting are done in Galaxy. Protein levels need proteomics data.
        """
    )

# ---- Prediction
with tabs[3]:
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
        fig, ax = plt.subplots(figsize=(3.2, 2.7))
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
        show_fig(fig)
    with g2:
        if yte.nunique() > 1:
            fpr, tpr, _ = roc_curve(yte, prob)
            fig, ax = plt.subplots(figsize=(3.2, 2.7))
            ax.plot(fpr, tpr, color="firebrick", lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curve")
            show_fig(fig)
    st.caption("Genes used: " + ", ".join(sel_tr[:20]) + (" ..." if len(sel_tr) > 20 else "") + ". With few test samples these numbers can change a lot, and on simulated data they are very high.")

    st.subheader("Predict new patients")

    def _predict_new():
        final_model, sel_final = fit(X, y)
        mode = st.radio("Input", ["Pick a sample from the data", "Upload new patient CSV"], horizontal=True)
        if mode == "Pick a sample from the data":
            pick = st.selectbox("Sample", list(sub.index))
            new = sub.loc[[pick], gene_cols]
        else:
            up = st.file_uploader("New patient CSV (rows = patients, columns = genes, same type of values as the training data)", type="csv", key="newpat")
            if up is None:
                return
            new = pd.read_csv(up, index_col=0)
            missing = [g for g in sel_final if g not in new.columns]
            if missing:
                st.error("Missing gene columns: " + ", ".join(missing[:10]))
                return
        p_new = final_model.predict_proba(to_log(new[sel_final]))[:, 1]
        out = pd.DataFrame({"prediction": np.where(p_new >= 0.5, case, control), f"probability of {case}": p_new.round(3)}, index=new.index)
        st.dataframe(out)
        st.warning("Learning tool only. Not for medical use.")

    popup("🔮 Open patient predictor", "Predict new patients", _predict_new, "pop_pred")
    st.caption("Learning tool only. Not for medical use.")
