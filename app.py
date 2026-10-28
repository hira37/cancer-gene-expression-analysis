"""Cancer Gene Expression Explorer (single-file version).

Everything is in this one file: key cancer pathways, sequence tools (UniProt, multiple sequence alignment,
phylogenetic trees), pathway results, apoptosis sequences control vs disease, and the Streamlit app.

Run:  streamlit run app.py
"""
import io
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from Bio import Phylo
from Bio.Align import MultipleSeqAlignment, PairwiseAligner, substitution_matrices
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo import Consensus
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# ---- Key cancer pathways -------------------------------------------------------------------------------
# Each pathway has:
#   icon, description : shown at the top of the pathway tab
#   pos_label/neg_label : names for the two gene classes (role +1 / role -1)
#   genes : {symbol: (role, short description)}  role = +1, -1 or 0 (neutral member)
# Gene symbols are official HGNC symbols (also accepted by UniProt for other species).
# To add genes or a whole new pathway, edit the PATHWAYS dictionary below.
PATHWAYS = {
    "Apoptosis": {
        "icon": "💀",
        "description": "Programmed cell death. Cancer cells often switch it off by raising anti-apoptotic genes "
                       "or losing pro-apoptotic genes. Two routes: intrinsic (mitochondria, BCL-2 family, CASP9) "
                       "and extrinsic (death receptors, CASP8).",
        "pos_label": "pro-apoptotic",
        "neg_label": "anti-apoptotic",
        "genes": {
            "TP53": (1, "p53, tumour suppressor; triggers apoptosis after DNA damage"),
            "BAX": (1, "BAX, pro-apoptotic BCL-2 family; pores in mitochondria"),
            "BAK1": (1, "BAK, pro-apoptotic BCL-2 family"),
            "BID": (1, "BID, links death receptors to mitochondria"),
            "BBC3": (1, "PUMA, BH3-only pro-apoptotic protein"),
            "PMAIP1": (1, "NOXA, BH3-only pro-apoptotic protein"),
            "BAD": (1, "BAD, BH3-only protein, blocks BCL-2/BCL-xL"),
            "BCL2L11": (1, "BIM, BH3-only pro-apoptotic protein"),
            "APAF1": (1, "APAF-1, apoptosome scaffold"),
            "CYCS": (1, "Cytochrome c, released from mitochondria"),
            "DIABLO": (1, "SMAC/DIABLO, blocks IAP proteins"),
            "CASP9": (1, "Caspase-9, starts the mitochondrial route"),
            "CASP8": (1, "Caspase-8, starts the death-receptor route"),
            "CASP3": (1, "Caspase-3, main executioner caspase"),
            "CASP6": (1, "Caspase-6, executioner caspase"),
            "CASP7": (1, "Caspase-7, executioner caspase"),
            "FAS": (1, "FAS death receptor"),
            "FADD": (1, "FADD, adaptor of death receptors"),
            "BCL2": (-1, "BCL-2, anti-apoptotic"),
            "BCL2L1": (-1, "BCL-xL, anti-apoptotic"),
            "MCL1": (-1, "MCL-1, anti-apoptotic"),
            "XIAP": (-1, "XIAP, inhibitor of caspases"),
            "BIRC5": (-1, "Survivin, inhibitor of apoptosis"),
            "BIRC2": (-1, "cIAP1, inhibitor of apoptosis"),
            "BIRC3": (-1, "cIAP2, inhibitor of apoptosis"),
            "CFLAR": (-1, "c-FLIP, blocks caspase-8"),
        },
    },
    "p53 signalling": {
        "icon": "🛡️",
        "description": "The p53 'guardian of the genome' route: DNA damage sensors activate p53, which stops the cell "
                       "cycle, repairs DNA or sends the cell to apoptosis. MDM2/MDM4 keep p53 low in healthy cells.",
        "pos_label": "tumour-suppressive",
        "neg_label": "p53 inhibitor",
        "genes": {
            "TP53": (1, "p53 transcription factor"),
            "ATM": (1, "ATM kinase, senses double-strand breaks"),
            "ATR": (1, "ATR kinase, senses replication stress"),
            "CHEK1": (1, "CHK1 kinase"),
            "CHEK2": (1, "CHK2 kinase, activates p53"),
            "CDKN1A": (1, "p21, cell-cycle inhibitor, p53 target"),
            "GADD45A": (1, "GADD45A, DNA-damage inducible, p53 target"),
            "BAX": (1, "BAX, p53 target, apoptosis"),
            "SESN1": (1, "Sestrin-1, p53 target"),
            "MDM2": (-1, "MDM2, ubiquitin ligase that degrades p53"),
            "MDM4": (-1, "MDM4 (MDMX), inhibits p53"),
            "PPM1D": (-1, "WIP1 phosphatase, switches p53 route off"),
        },
    },
    "Cell cycle": {
        "icon": "🔄",
        "description": "Cyclins and CDKs push the cell through G1, S, G2 and mitosis; inhibitors and RB1 act as brakes. "
                       "Cancers usually have high cyclin/CDK and E2F activity and low inhibitors.",
        "pos_label": "promotes progression",
        "neg_label": "restrains cycle",
        "genes": {
            "CCND1": (1, "Cyclin D1, G1 progression"),
            "CCNE1": (1, "Cyclin E1, G1/S transition"),
            "CCNA2": (1, "Cyclin A2, S/G2"),
            "CCNB1": (1, "Cyclin B1, mitosis"),
            "CDK1": (1, "CDK1, mitosis"),
            "CDK2": (1, "CDK2, G1/S"),
            "CDK4": (1, "CDK4, G1"),
            "CDK6": (1, "CDK6, G1"),
            "E2F1": (1, "E2F1, S-phase transcription factor"),
            "CDC25A": (1, "CDC25A phosphatase"),
            "PLK1": (1, "Polo-like kinase 1, mitosis"),
            "AURKA": (1, "Aurora kinase A, mitosis"),
            "MYC": (1, "MYC, drives cell growth"),
            "RB1": (-1, "RB1, blocks E2F"),
            "CDKN1A": (-1, "p21, CDK inhibitor"),
            "CDKN1B": (-1, "p27, CDK inhibitor"),
            "CDKN2A": (-1, "p16INK4a, CDK4/6 inhibitor"),
        },
    },
    "PI3K-AKT-mTOR": {
        "icon": "📈",
        "description": "Growth and survival signalling downstream of receptor tyrosine kinases. PIK3CA is one of the "
                       "most frequently mutated genes in breast cancer; PTEN loss has the same effect.",
        "pos_label": "activator",
        "neg_label": "inhibitor",
        "genes": {
            "PIK3CA": (1, "PI3K catalytic subunit alpha"),
            "PIK3CB": (1, "PI3K catalytic subunit beta"),
            "PIK3R1": (-1, "PI3K regulatory subunit (restrains p110)"),
            "PDPK1": (1, "PDK1 kinase"),
            "AKT1": (1, "AKT1 kinase"),
            "AKT2": (1, "AKT2 kinase"),
            "AKT3": (1, "AKT3 kinase"),
            "MTOR": (1, "mTOR kinase"),
            "RPTOR": (1, "RAPTOR, part of mTORC1"),
            "RPS6KB1": (1, "S6 kinase 1"),
            "EIF4EBP1": (-1, "4E-BP1, translation repressor"),
            "PTEN": (-1, "PTEN phosphatase, turns PI3K signal off"),
            "TSC1": (-1, "TSC1, inhibits mTORC1"),
            "TSC2": (-1, "TSC2, inhibits mTORC1"),
            "FOXO3": (-1, "FOXO3, pro-death transcription factor"),
        },
    },
    "MAPK / RAS": {
        "icon": "🧷",
        "description": "RAS-RAF-MEK-ERK cascade passing growth-factor signals to the nucleus. Hyperactivation drives "
                       "proliferation; DUSP phosphatases and NF1 switch it off.",
        "pos_label": "activator",
        "neg_label": "inhibitor",
        "genes": {
            "EGFR": (1, "EGF receptor"),
            "ERBB2": (1, "HER2 receptor"),
            "GRB2": (1, "GRB2 adaptor"),
            "SOS1": (1, "SOS1, RAS activator"),
            "KRAS": (1, "KRAS GTPase"),
            "HRAS": (1, "HRAS GTPase"),
            "NRAS": (1, "NRAS GTPase"),
            "BRAF": (1, "BRAF kinase"),
            "RAF1": (1, "RAF-1 kinase"),
            "MAP2K1": (1, "MEK1"),
            "MAP2K2": (1, "MEK2"),
            "MAPK1": (1, "ERK2"),
            "MAPK3": (1, "ERK1"),
            "FOS": (1, "FOS, AP-1 transcription factor"),
            "JUN": (1, "JUN, AP-1 transcription factor"),
            "NF1": (-1, "Neurofibromin, RAS GAP"),
            "DUSP1": (-1, "DUSP1 phosphatase"),
            "DUSP6": (-1, "DUSP6 phosphatase"),
        },
    },
    "Wnt / beta-catenin": {
        "icon": "🧵",
        "description": "Controls stem-cell renewal and growth. When the destruction complex (APC, AXIN1, GSK3B) fails, "
                       "beta-catenin enters the nucleus and switches on MYC and CCND1.",
        "pos_label": "activator",
        "neg_label": "inhibitor",
        "genes": {
            "WNT5A": (1, "WNT5A ligand"),
            "FZD7": (1, "Frizzled-7 receptor"),
            "LRP6": (1, "LRP6 co-receptor"),
            "DVL1": (1, "Dishevelled-1"),
            "CTNNB1": (1, "Beta-catenin"),
            "TCF7L2": (1, "TCF7L2 transcription factor"),
            "LEF1": (1, "LEF1 transcription factor"),
            "MYC": (1, "MYC, Wnt target"),
            "CCND1": (1, "Cyclin D1, Wnt target"),
            "GSK3B": (-1, "GSK-3 beta, marks beta-catenin for destruction"),
            "AXIN1": (-1, "AXIN1, destruction complex scaffold"),
            "APC": (-1, "APC, destruction complex"),
            "DKK1": (-1, "DKK1, Wnt antagonist"),
            "SFRP1": (-1, "SFRP1, Wnt antagonist"),
        },
    },
    "NF-kB / inflammation": {
        "icon": "🔥",
        "description": "Inflammatory signalling that also protects cells from death and promotes growth. "
                       "IκB proteins (NFKBIA) hold NF-κB in the cytoplasm until IKK phosphorylates them.",
        "pos_label": "activator",
        "neg_label": "inhibitor",
        "genes": {
            "TNF": (1, "TNF-alpha cytokine"),
            "TRAF2": (1, "TRAF2 adaptor"),
            "TRAF6": (1, "TRAF6 adaptor"),
            "RIPK1": (1, "RIPK1 kinase"),
            "CHUK": (1, "IKK-alpha"),
            "IKBKB": (1, "IKK-beta"),
            "IKBKG": (1, "NEMO, IKK regulatory subunit"),
            "NFKB1": (1, "NF-kB p105/p50"),
            "RELA": (1, "RELA (p65)"),
            "REL": (1, "c-REL"),
            "NFKBIA": (-1, "IkB-alpha, NF-kB inhibitor"),
            "TNFAIP3": (-1, "A20, NF-kB switch-off"),
        },
    },
    "DNA damage repair": {
        "icon": "🧬",
        "description": "Homologous recombination, mismatch repair and base-excision repair. BRCA1/BRCA2 defects are classic "
                       "breast cancer predisposition genes and make tumours sensitive to PARP inhibitors.",
        "pos_label": "repair factor",
        "neg_label": "n/a",
        "genes": {
            "BRCA1": (1, "BRCA1, homologous recombination"),
            "BRCA2": (1, "BRCA2, loads RAD51"),
            "PALB2": (1, "PALB2, links BRCA1 and BRCA2"),
            "RAD51": (1, "RAD51 recombinase"),
            "ATM": (1, "ATM kinase"),
            "ATR": (1, "ATR kinase"),
            "CHEK2": (1, "CHK2 kinase"),
            "MLH1": (1, "MLH1, mismatch repair"),
            "MSH2": (1, "MSH2, mismatch repair"),
            "MSH6": (1, "MSH6, mismatch repair"),
            "PARP1": (1, "PARP1, single-strand break repair"),
            "XRCC1": (1, "XRCC1, base-excision repair"),
            "ERCC1": (1, "ERCC1, nucleotide-excision repair"),
            "FANCD2": (1, "FANCD2, Fanconi pathway"),
        },
    },
    "Hormone receptor / luminal": {
        "icon": "🎀",
        "description": "Oestrogen-receptor driven programme typical of luminal breast cancer (ESR1, GATA3, FOXA1) and "
                       "HER2/ERBB2 signalling. Used to separate breast cancer subtypes.",
        "pos_label": "luminal marker",
        "neg_label": "n/a",
        "genes": {
            "ESR1": (1, "Oestrogen receptor alpha"),
            "PGR": (1, "Progesterone receptor"),
            "GATA3": (1, "GATA3, luminal transcription factor"),
            "FOXA1": (1, "FOXA1, pioneer factor for ER"),
            "XBP1": (1, "XBP1, ER-associated"),
            "TFF1": (1, "TFF1, ER target"),
            "GREB1": (1, "GREB1, ER target"),
            "AR": (1, "Androgen receptor"),
            "ERBB2": (1, "HER2 receptor"),
            "ERBB3": (1, "HER3 receptor"),
            "CCND1": (1, "Cyclin D1, ER target"),
        },
    },
    "Hypoxia / angiogenesis": {
        "icon": "🩸",
        "description": "Low oxygen stabilises HIF1A/EPAS1, which switch on VEGFA and glycolysis genes so that tumours grow "
                       "new blood vessels. VHL and EGLN1 normally destroy HIF in oxygen.",
        "pos_label": "pro-angiogenic / hypoxia",
        "neg_label": "HIF suppressor",
        "genes": {
            "HIF1A": (1, "HIF-1 alpha"),
            "EPAS1": (1, "HIF-2 alpha"),
            "VEGFA": (1, "VEGF-A"),
            "VEGFB": (1, "VEGF-B"),
            "KDR": (1, "VEGFR2"),
            "FLT1": (1, "VEGFR1"),
            "ANGPT2": (1, "Angiopoietin-2"),
            "SLC2A1": (1, "GLUT1, glucose transporter"),
            "LDHA": (1, "Lactate dehydrogenase A"),
            "CA9": (1, "Carbonic anhydrase IX"),
            "VHL": (-1, "VHL, degrades HIF"),
            "EGLN1": (-1, "PHD2, hydroxylates HIF"),
            "HIF1AN": (-1, "FIH, inhibits HIF"),
        },
    },
    "EMT / metastasis": {
        "icon": "🚀",
        "description": "Epithelial-to-mesenchymal transition lets cancer cells lose cell-cell contacts (CDH1) and become "
                       "invasive (VIM, SNAI1/2, TWIST1, ZEB1/2, MMPs).",
        "pos_label": "mesenchymal / invasive",
        "neg_label": "epithelial",
        "genes": {
            "CDH2": (1, "N-cadherin"),
            "VIM": (1, "Vimentin"),
            "SNAI1": (1, "Snail"),
            "SNAI2": (1, "Slug"),
            "TWIST1": (1, "TWIST1"),
            "ZEB1": (1, "ZEB1"),
            "ZEB2": (1, "ZEB2"),
            "FN1": (1, "Fibronectin"),
            "MMP2": (1, "Matrix metalloproteinase 2"),
            "MMP9": (1, "Matrix metalloproteinase 9"),
            "TGFB1": (1, "TGF-beta 1"),
            "TGFBR1": (1, "TGF-beta receptor 1"),
            "CDH1": (-1, "E-cadherin"),
            "CLDN4": (-1, "Claudin-4, tight junction"),
            "EPCAM": (-1, "EpCAM, epithelial marker"),
        },
    },
    "Autophagy": {
        "icon": "♻️",
        "description": "Cells recycle their own components. It can protect tumours under stress, but BECN1 loss is also "
                       "linked to cancer. It talks to apoptosis through BCL-2/BECLIN-1.",
        "pos_label": "pro-autophagy",
        "neg_label": "autophagy inhibitor",
        "genes": {
            "ULK1": (1, "ULK1 kinase, starts autophagy"),
            "BECN1": (1, "Beclin-1"),
            "ATG5": (1, "ATG5"),
            "ATG7": (1, "ATG7"),
            "ATG12": (1, "ATG12"),
            "MAP1LC3B": (1, "LC3B, autophagosome marker"),
            "SQSTM1": (1, "p62, cargo receptor"),
            "BCL2": (-1, "BCL-2, binds Beclin-1"),
            "MTOR": (-1, "mTOR, blocks ULK1"),
        },
    },
}


def pathway_gene_roles(name):
    """Return {gene: role} for a pathway."""
    return {g: v[0] for g, v in PATHWAYS[name]["genes"].items()}


def pathway_gene_info(name):
    return {g: v[1] for g, v in PATHWAYS[name]["genes"].items()}


# ---- Sequence tools -----------------------------------------------------------------------------------
# UniProt organism (taxonomy) ids
SPECIES = {
    "Human": 9606,
    "Mouse": 10090,
    "Rat": 10116,
    "Rabbit": 9986,
    "Dog": 9615,
    "Cow": 9913,
    "Pig": 9823,
    "Chicken": 9031,
    "Frog (X. tropicalis)": 8364,
    "Zebrafish": 7955,
}
DEFAULT_SPECIES = ["Human", "Mouse", "Rat", "Dog", "Cow", "Chicken", "Zebrafish"]

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seq_cache.json")
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"


# ------------------------------------------------------------------ fasta + download
def parse_fasta(text):
    """Parse FASTA text into {name: sequence}. Keeps only letters."""
    seqs, name, buf = {}, None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name = line[1:].split()[0] if line[1:].strip() else f"seq{len(seqs) + 1}"
            while name in seqs:
                name += "_"
            buf = []
        else:
            buf.append(re.sub(r"[^A-Za-z]", "", line).upper())
    if name is not None:
        seqs[name] = "".join(buf)
    return {k: v for k, v in seqs.items() if v}


def to_fasta(seqs, width=60):
    out = []
    for k, v in seqs.items():
        out.append(f">{k}")
        out += [v[i:i + width] for i in range(0, len(v), width)]
    return "\n".join(out) + "\n"


def _load_cache():
    try:
        with open(CACHE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as fh:
            json.dump(cache, fh)
    except Exception:
        pass


def fetch_protein(gene, taxid, timeout=20):
    """Download the canonical protein sequence of a gene in one species from UniProt.

    Returns dict(accession, protein, sequence) or None. Results are cached in seq_cache.json,
    so a second run works offline.
    """
    key = f"{gene.upper()}|{taxid}"
    cache = _load_cache()
    if key in cache:
        return cache[key]
    for reviewed in ("true", "false"):
        query = f"(gene_exact:{gene}) AND (organism_id:{taxid})"
        if reviewed == "true":
            query += " AND (reviewed:true)"
        try:
            r = requests.get(UNIPROT, params={"query": query, "format": "json", "size": 1,
                                              "fields": "accession,protein_name,sequence"}, timeout=timeout)
            r.raise_for_status()
            hits = r.json().get("results", [])
        except Exception:
            return None  # network problem: do not cache
        if hits:
            h = hits[0]
            try:
                name = h["proteinDescription"]["recommendedName"]["fullName"]["value"]
            except Exception:
                name = ""
            rec = {"accession": h["primaryAccession"], "protein": name, "sequence": h["sequence"]["value"]}
            cache[key] = rec
            _save_cache(cache)
            return rec
    cache[key] = None
    _save_cache(cache)
    return None


def fetch_orthologs(gene, species_names, progress=None):
    """{species: sequence} for one gene. Also returns list of species where it was not found."""
    seqs, missing = {}, []
    for i, sp in enumerate(species_names):
        rec = fetch_protein(gene, SPECIES[sp])
        if rec:
            seqs[sp] = rec["sequence"]
        else:
            missing.append(sp)
        if progress:
            progress((i + 1) / len(species_names), f"{gene} in {sp}")
    return seqs, missing


def fetch_family(genes, species="Human", progress=None):
    """{gene: sequence} for several genes of one species."""
    seqs, missing = {}, []
    for i, g in enumerate(genes):
        rec = fetch_protein(g, SPECIES[species])
        if rec:
            seqs[g] = rec["sequence"]
        else:
            missing.append(g)
        if progress:
            progress((i + 1) / len(genes), f"{g} ({species})")
    return seqs, missing


# ------------------------------------------------------------------ alignment
def available_engines():
    engines = ["Built-in (Biopython, center-star)"]
    for exe, label in (("mafft", "MAFFT"), ("clustalo", "Clustal Omega"), ("muscle", "MUSCLE")):
        if shutil.which(exe):
            engines.insert(0, label)
    return engines


def _pairwise_strings(a, b, aligner):
    """Global pairwise alignment of two strings -> two gapped strings."""
    aln = aligner.align(a, b)[0]
    sa, sb, ia, ib = [], [], 0, 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        while ia < a0:
            sa.append(a[ia]); sb.append("-"); ia += 1
        while ib < b0:
            sa.append("-"); sb.append(b[ib]); ib += 1
        for _ in range(a1 - a0):
            sa.append(a[ia]); sb.append(b[ib]); ia += 1; ib += 1
    while ia < len(a):
        sa.append(a[ia]); sb.append("-"); ia += 1
    while ib < len(b):
        sa.append("-"); sb.append(b[ib]); ib += 1
    return "".join(sa), "".join(sb)


def _make_aligner():
    al = PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score = -10
    al.extend_gap_score = -0.5
    al.mode = "global"
    return al


def center_star_msa(seqs):
    """Pure-python multiple alignment: align everything to the most central sequence
    ('once a gap, always a gap'). Good for closely related sequences (orthologs)."""
    names = list(seqs)
    if len(names) == 1:
        return dict(seqs)
    al = _make_aligner()
    n = len(names)
    score = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score[i, j] = score[j, i] = al.score(seqs[names[i]], seqs[names[j]])
    c = int(score.sum(axis=1).argmax())
    center = seqs[names[c]]
    rows = {names[c]: center}
    cur_c = center
    for k in sorted((i for i in range(n) if i != c), key=lambda i: -score[c, i]):
        c_aln, s_aln = _pairwise_strings(center, seqs[names[k]], al)
        i = j = 0
        new_s, new_rows = [], {nm: [] for nm in rows}
        merged_c = []
        while i < len(cur_c) or j < len(c_aln):
            ci = cur_c[i] if i < len(cur_c) else None
            cj = c_aln[j] if j < len(c_aln) else None
            if ci == "-" and cj == "-":
                take_old, take_new = True, True
            elif ci == "-":
                take_old, take_new = True, False
            elif cj == "-" or ci is None:
                take_old, take_new = False, True
            else:
                take_old, take_new = True, True
            for nm in rows:
                new_rows[nm].append(rows[nm][i] if take_old else "-")
            new_s.append(s_aln[j] if take_new else "-")
            merged_c.append(cur_c[i] if take_old else c_aln[j])
            i += int(take_old)
            j += int(take_new)
        rows = {nm: "".join(v) for nm, v in new_rows.items()}
        rows[names[k]] = "".join(new_s)
        cur_c = rows[names[c]]
    return {nm: rows[nm] for nm in names}


def _external_msa(seqs, engine):
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "in.fa"), os.path.join(d, "out.fa")
        with open(inp, "w") as fh:
            fh.write(to_fasta(seqs))
        if engine == "MAFFT":
            res = subprocess.run(["mafft", "--auto", "--quiet", inp], capture_output=True, text=True, timeout=300)
            text = res.stdout
        elif engine == "Clustal Omega":
            subprocess.run(["clustalo", "-i", inp, "-o", out, "--force", "--outfmt=fa"],
                           capture_output=True, text=True, timeout=300, check=True)
            text = open(out).read()
        else:  # MUSCLE v5 syntax, fall back to v3
            res = subprocess.run(["muscle", "-align", inp, "-output", out], capture_output=True, text=True, timeout=300)
            if res.returncode != 0:
                subprocess.run(["muscle", "-in", inp, "-out", out], capture_output=True, text=True, timeout=300, check=True)
            text = open(out).read()
    aln = parse_fasta(text)
    if set(aln) != set(seqs):
        raise RuntimeError("alignment program returned unexpected names")
    return {k: aln[k] for k in seqs}


def run_msa(seqs, engine="Built-in (Biopython, center-star)"):
    """Return (aligned dict, engine actually used)."""
    seqs = {k: re.sub(r"[^A-Z]", "", v.upper()) for k, v in seqs.items()}
    if len(seqs) < 2:
        raise ValueError("Need at least 2 sequences.")
    if not engine.startswith("Built-in"):
        try:
            return _external_msa(seqs, engine), engine
        except Exception:
            pass
    return center_star_msa(seqs), "Built-in (Biopython, center-star)"


# ------------------------------------------------------------------ alignment statistics
def conservation(aln):
    """Per-column identity (fraction of sequences with the most common residue; gaps count against)."""
    rows = list(aln.values())
    L, n = len(rows[0]), len(rows)
    out = np.zeros(L)
    for i in range(L):
        col = [r[i] for r in rows if r[i] != "-"]
        if col:
            out[i] = max(col.count(a) for a in set(col)) / n
    return out


def identity_matrix(aln):
    """Pairwise % identity over aligned (non-gap in both) positions."""
    names = list(aln)
    n = len(names)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pairs = [(a, b) for a, b in zip(aln[names[i]], aln[names[j]]) if a != "-" and b != "-"]
            m[i, j] = 100 * sum(a == b for a, b in pairs) / len(pairs) if pairs else 0
    return names, m


def alignment_stats(aln):
    cons = conservation(aln)
    L = len(cons)
    n = len(aln)
    full = int((cons == 1.0).sum())
    gap_cols = sum(any(r[i] == "-" for r in aln.values()) for i in range(L))
    _, m = identity_matrix(aln)
    off = m[~np.eye(n, dtype=bool)] if n > 1 else np.array([100.0])
    return {"length": L, "sequences": n, "identical columns": full,
            "% identical columns": round(100 * full / L, 1), "columns with gaps": gap_cols,
            "mean pairwise identity %": round(float(off.mean()), 1)}


_AA_CLASS = {}
for _aa in "AVLIMFWC":
    _AA_CLASS[_aa] = "#8fb4e8"   # hydrophobic
for _aa in "KR":
    _AA_CLASS[_aa] = "#e8908f"   # positive
for _aa in "DE":
    _AA_CLASS[_aa] = "#c58fe8"   # negative
for _aa in "STNQ":
    _AA_CLASS[_aa] = "#8fe8a4"   # polar
for _aa in "GP":
    _AA_CLASS[_aa] = "#e8d68f"   # special
for _aa in "HY":
    _AA_CLASS[_aa] = "#8fe8e0"   # aromatic/his


def alignment_html(aln, block=60):
    """Coloured, scrolling HTML view of an alignment with a conservation line."""
    names = list(aln)
    pad = max(len(n) for n in names) + 1
    L = len(next(iter(aln.values())))
    cons = conservation(aln)
    lines = []
    for start in range(0, L, block):
        end = min(start + block, L)
        for nm in names:
            seg = aln[nm][start:end]
            cells = "".join(
                f'<span style="background:{_AA_CLASS.get(ch, "transparent")}">{ch}</span>' if ch != "-"
                else '<span style="color:#999">-</span>' for ch in seg)
            lines.append(f'{nm:<{pad}}{cells}  {start + 1}-{end}')
        mark = "".join("*" if cons[i] == 1.0 else ":" if cons[i] >= 0.8 else "." if cons[i] >= 0.5 else " "
                       for i in range(start, end))
        lines.append(" " * pad + mark)
        lines.append("")
    body = "\n".join(lines)
    return ('<div style="font-family:monospace;font-size:12.5px;line-height:1.35;white-space:pre;overflow-x:auto;'
            'border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:10px;max-height:520px;overflow-y:auto">'
            + body + "</div>")


LEGEND_HTML = (
    '<span style="background:#8fb4e8;padding:1px 6px">hydrophobic AVLIMFWC</span> '
    '<span style="background:#e8908f;padding:1px 6px">positive KR</span> '
    '<span style="background:#c58fe8;padding:1px 6px">negative DE</span> '
    '<span style="background:#8fe8a4;padding:1px 6px">polar STNQ</span> '
    '<span style="background:#e8d68f;padding:1px 6px">G, P</span> '
    '<span style="background:#8fe8e0;padding:1px 6px">H, Y</span> &nbsp; '
    '<code>*</code> identical &nbsp; <code>:</code> &ge;80% &nbsp; <code>.</code> &ge;50%'
)


# ------------------------------------------------------------------ phylogeny
def _to_msa(aln):
    return MultipleSeqAlignment([SeqRecord(Seq(s), id=k, description="") for k, s in aln.items()])


def clean_alignment(aln, max_gap_fraction=0.5):
    """Drop columns that are mostly gaps (they carry little tree signal)."""
    rows = list(aln.values())
    keep = [i for i in range(len(rows[0])) if sum(r[i] == "-" for r in rows) / len(rows) <= max_gap_fraction]
    if len(keep) < 10:
        keep = list(range(len(rows[0])))
    return {k: "".join(v[i] for i in keep) for k, v in aln.items()}


def build_tree(aln, method="nj", bootstrap=0, model="blosum62", seed=1):
    """Distance tree (neighbour joining or UPGMA) with optional bootstrap support (%).

    Returns a Bio.Phylo tree (midpoint rooted for NJ).
    """
    if len(aln) < 3:
        raise ValueError("A tree needs at least 3 sequences.")
    msa = _to_msa(clean_alignment(aln))
    constructor = DistanceTreeConstructor(DistanceCalculator(model), method.lower())
    tree = constructor.build_tree(msa)
    if bootstrap and bootstrap > 0:
        import random
        random.seed(seed)
        trees = list(Consensus.bootstrap_trees(msa, int(bootstrap), constructor))
        tree = Consensus.get_support(tree, trees)
        for cl in tree.get_nonterminals():          # Biopython already gives support in percent
            if cl.confidence is not None:
                cl.confidence = int(round(min(cl.confidence, 100)))
    for cl in tree.get_nonterminals():
        cl.name = None                                # cleaner Newick (support is kept in .confidence)
    for cl in tree.find_clades():
        if cl.branch_length is not None and cl.branch_length < 0:
            cl.branch_length = 0.0                    # NJ can give tiny negative lengths
    if method.lower() == "nj":
        tree.root_at_midpoint()
    tree.ladderize()
    return tree


def draw_tree(tree, ax, title=""):
    def lab(c):
        return c.name if c.is_terminal() else None

    def blab(c):
        return f"{c.confidence:g}" if (not c.is_terminal() and c.confidence is not None) else None

    Phylo.draw(tree, axes=ax, do_show=False, label_func=lab, branch_labels=blab)
    ax.set_title(title)
    ax.set_xlabel("branch length (substitutions per site, BLOSUM62 distance)")
    ax.set_ylabel("")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def tree_newick(tree):
    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    return buf.getvalue()


def concatenate_alignments(per_gene_aln):
    """per_gene_aln: {gene: {species: aligned}}. Keeps species present in every gene.
    Returns concatenated {species: sequence} and the gene boundaries."""
    common = set.intersection(*[set(a) for a in per_gene_aln.values()]) if per_gene_aln else set()
    common = [s for s in next(iter(per_gene_aln.values())) if s in common]
    out = {s: "" for s in common}
    bounds, pos = [], 0
    for g, a in per_gene_aln.items():
        L = len(next(iter(a.values())))
        for s in common:
            out[s] += a[s]
        bounds.append((g, pos, pos + L))
        pos += L
    return out, bounds



# =====================================================================================================
# NEW: helpers for pathway results, sample-wise protein variants, control-vs-disease separation
# (no Streamlit code in this block)
# =====================================================================================================
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from Bio.Phylo.TreeConstruction import DistanceMatrix

GROUP_COLORS = {"disease": "firebrick", "control": "steelblue", "reference": "black"}

THREE2ONE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
             "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
             "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(x))


# ------------------------------------------------------------------ pathway level results
def pathway_score(pname, log_counts):
    """Per-sample pathway score = mean z(pro genes) - mean z(anti genes). Returns (Series or None, present genes)."""
    roles = pathway_gene_roles(pname)
    present = [g for g in roles if g in log_counts.index]
    if len(present) < 2:
        return None, present
    L = log_counts.loc[present]
    z = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1) + 1e-9, axis=0)
    pos = [g for g in present if roles[g] == 1]
    neg = [g for g in present if roles[g] == -1]
    s = pd.Series(0.0, index=L.columns)
    if pos:
        s = s + z.loc[pos].mean()
    if neg:
        s = s - z.loc[neg].mean()
    return s, present


def pathway_table(result, log_counts, case_cols, ctrl_cols):
    """One row per pathway: DEG counts, over-representation test, pathway score test.
    Returns (table, {pathway: per-sample score})."""
    M = len(result)
    N = int(result["DEG"].sum())
    rows, scores = [], {}
    for pname in PATHWAYS:
        roles = pathway_gene_roles(pname)
        present = [g for g in roles if g in result.index]
        if not present:
            continue
        degs = [g for g in present if result.loc[g, "DEG"]]
        up = [g for g in degs if result.loc[g, "log2FC"] > 0]
        down = [g for g in degs if result.loc[g, "log2FC"] < 0]
        k, n = len(degs), len(present)
        p_over = float(stats.hypergeom.sf(k - 1, M, n, N)) if (N > 0 and k > 0) else 1.0
        sc, _ = pathway_score(pname, log_counts)
        diff = d = p_sc = np.nan
        if sc is not None:
            a, b = sc[list(case_cols)], sc[list(ctrl_cols)]
            diff = float(a.mean() - b.mean())
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            d = float(diff / sd) if sd > 0 else 0.0
            p_sc = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
            scores[pname] = sc
        rows.append({"pathway": pname, "genes in data": n, "genes in pathway": len(roles), "DEGs": k,
                     "up": len(up), "down": len(down), "DEG genes": ", ".join(degs),
                     "enrichment p": p_over, "score diff": diff, "effect size (d)": d, "score p": p_sc})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab, scores
    tab["enrichment padj"] = multipletests(tab["enrichment p"].fillna(1.0), method="fdr_bh")[1]
    ok = tab["score p"].notna()
    tab["score padj"] = np.nan
    if ok.any():
        tab.loc[ok, "score padj"] = multipletests(tab.loc[ok, "score p"], method="fdr_bh")[1]
    tab["score direction"] = np.where(tab["score diff"].isna(), "n/a",
                                      np.where(tab["score diff"] > 0, "higher in case", "lower in case"))
    return tab, scores


def mutation_pathway_table(mutations, sub, case, control):
    """Samples with at least one mutation in the pathway genes, by group (Fisher exact test)."""
    m = mutations[mutations["sample"].isin(sub.index)]
    grp = sub["group"]
    n_case, n_ctrl = int((grp == case).sum()), int((grp == control).sum())
    rows = []
    for pname, P in PATHWAYS.items():
        mut_samples = m.loc[m["gene"].isin(P["genes"]), "sample"].unique()
        a = int(sum(grp.get(s) == case for s in mut_samples))
        b = int(sum(grp.get(s) == control for s in mut_samples))
        p = stats.fisher_exact([[a, n_case - a], [b, n_ctrl - b]])[1]
        rows.append({"pathway": pname, f"% {case} mutated": round(100 * a / n_case, 1),
                     f"% {control} mutated": round(100 * b / n_ctrl, 1), "Fisher p": p})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sample-wise protein sequences
def parse_protein_change(text):
    """'p.R175H', 'R175H', 'p.Arg175His', 'p.R213*' -> (from, pos, to). None if not a simple substitution."""
    t = re.sub(r"[()]", "", re.sub(r"^p\.", "", str(text).strip(), flags=re.I))
    m = re.fullmatch(r"([A-Za-z*]{1,3})(\d+)([A-Za-z*]{1,3})", t)
    if not m:
        return None

    def conv(x):
        if x == "*" or x.lower() in ("ter", "x"):
            return "*"
        if len(x) == 1:
            return x.upper()
        return THREE2ONE.get(x.capitalize())

    a, pos, b = m.groups()
    a, b = conv(a), conv(b)
    if a is None or b is None or a == "*":
        return None
    return a, int(pos), b


def apply_variants(ref, changes):
    """Apply (from, pos, to) amino-acid changes to a reference protein. Returns (sequence, applied, skipped)."""
    seq, applied, skipped, stop_at = list(ref), [], [], None
    for a, pos, b in sorted(changes, key=lambda x: x[1]):
        label = f"{a}{pos}{b}"
        if pos < 1 or pos > len(ref) or ref[pos - 1] != a:
            skipped.append(label)
            continue
        if b == "*":
            stop_at = pos - 1 if stop_at is None else min(stop_at, pos - 1)
        else:
            seq[pos - 1] = b
        applied.append(label)
    out = "".join(seq)
    if stop_at is not None and stop_at > 0:
        out = out[:stop_at]
    return out, applied, skipped


def sample_sequences_from_mutations(ref, gene, mutations, samples, group_of, change_col):
    """Every sample gets the reference protein of `gene` plus its own variants.
    Returns ({name: seq}, {name: group}, report rows)."""
    seqs, groups, rows = {}, {}, []
    m = mutations[mutations["gene"].astype(str).str.upper() == gene.upper()]
    for s in samples:
        changes, bad = [], []
        for val in m.loc[m["sample"] == s, change_col].dropna():
            pc = parse_protein_change(val)
            (changes if pc else bad).append(pc if pc else str(val))
        seq, applied, skipped = apply_variants(ref, changes)
        name = f"{group_of[s]}_{safe_name(s)}"
        seqs[name] = seq
        groups[name] = group_of[s]
        rows.append({"sample": s, "group": group_of[s], "gene": gene, "variants applied": ", ".join(applied),
                     "skipped (not simple / wrong residue)": ", ".join(skipped + bad)})
    return seqs, groups, rows


def msa_with_haplotypes(named_seqs, groups, engine):
    """Align only the distinct sequences (haplotypes), then give every sample the row of its haplotype.
    Returns dict(aln = per-sample alignment, hap_aln = alignment of haplotypes with readable names, used, n_hap)."""
    hap = {}
    for n, s in named_seqs.items():
        hap.setdefault(s, []).append(n)
    ids = {s: f"hap{i + 1}" for i, s in enumerate(hap)}
    if len(hap) == 1:                                  # no variation at all
        s = next(iter(hap))
        return {"aln": dict(named_seqs), "hap_aln": {"hap1_all_identical": s}, "used": "none needed", "n_hap": 1}
    aln_u, used = run_msa({ids[s]: s for s in hap}, engine)
    aln = {}
    for s, members in hap.items():
        for n in members:
            aln[n] = aln_u[ids[s]]
    hap_aln = {}
    for s, members in hap.items():
        c = sum(groups[n] == "control" for n in members)
        d = sum(groups[n] == "disease" for n in members)
        ref = "_REF" if "REF" in members else ""
        hap_aln[f"{ids[s]}{ref}_ctrl{c}_dis{d}"] = aln_u[ids[s]]
    return {"aln": aln, "hap_aln": hap_aln, "used": used, "n_hap": len(hap)}


# ------------------------------------------------------------------ control vs disease separation
def aln_distance(aln):
    """Pairwise p-distance (fraction of different residues over columns where both have a residue)."""
    names = list(aln)
    arr = np.array([list(aln[n]) for n in names])
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        valid = (arr != "-") & (arr[i] != "-")
        mism = (arr != arr[i]) & valid
        cnt = valid.sum(axis=1)
        D[i] = np.where(cnt > 0, mism.sum(axis=1) / np.maximum(cnt, 1), 0.0)
    return names, (D + D.T) / 2


def group_separation(names, D, groups, n_perm=999, seed=1):
    """How well do control and disease separate? Within vs between distances, silhouette, permutation p-value."""
    g = np.array([groups[n] for n in names])
    labs = sorted(set(g))
    if len(labs) < 2 or len(names) < 4:
        return None
    iu = np.triu_indices(len(names), 1)

    def stat(gv):
        same = (gv[:, None] == gv[None, :])[iu]
        d = D[iu]
        if same.all() or (~same).all():
            return 0.0, 0.0, 0.0
        w, b = d[same].mean(), d[~same].mean()
        return b - w, w, b

    obs, within, between = stat(g)
    rng = np.random.default_rng(seed)
    cnt = sum(stat(rng.permutation(g))[0] >= obs for _ in range(n_perm))
    p = (1 + cnt) / (1 + n_perm)
    try:
        sil = float(silhouette_score(D, g, metric="precomputed")) if D.max() > 0 else 0.0
    except Exception:
        sil = float("nan")
    within_by = {}
    for lab in labs:
        idx = np.where(g == lab)[0]
        if len(idx) > 1:
            sub_d = D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
            within_by[lab] = float(sub_d.mean())
    return {"within": float(within), "between": float(between), "difference": float(obs), "perm_p": float(p),
            "silhouette": sil, "within_by_group": within_by}


def separation_verdict(s, case, control):
    if s is None:
        return "Not enough samples to test separation."
    if s["between"] == 0 and s["within"] == 0:
        return "All sequences are identical, so there is nothing that could separate the groups."
    if s["perm_p"] < 0.05 and s["silhouette"] > 0.25:
        return (f"**{control} and {case} separate** into different clusters (silhouette {s['silhouette']:.2f}, "
                f"permutation p = {s['perm_p']:.3g}).")
    if s["perm_p"] < 0.05:
        return (f"The groups differ slightly (permutation p = {s['perm_p']:.3g}) but they overlap "
                f"(silhouette {s['silhouette']:.2f}), so they do not form clean clusters.")
    return (f"**{control} and {case} do not separate** (silhouette {s['silhouette']:.2f}, "
            f"permutation p = {s['perm_p']:.3g}).")


def group_monophyly(tree, groups):
    """For each group: is it one clade of the tree (True) or mixed with the other group (False)?"""
    out = {}
    for lab in sorted(set(groups.values()) - {"reference"}):
        tips = [t for t in tree.get_terminals() if groups.get(t.name) == lab]
        out[lab] = bool(len(tips) > 1 and tree.is_monophyletic(tips))
    return out


def draw_group_tree(tree, ax, groups, title=""):
    """Phylogenetic tree with tip labels coloured by group."""
    def lab(c):
        return c.name if c.is_terminal() else None

    def blab(c):
        return f"{c.confidence:g}" if (not c.is_terminal() and c.confidence is not None) else None

    def col(label):
        return GROUP_COLORS.get(groups.get(label, ""), "black")

    Phylo.draw(tree, axes=ax, do_show=False, label_func=lab, branch_labels=blab, label_colors=col)
    ax.set_title(title)
    ax.set_xlabel("branch length")
    ax.set_ylabel("")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def expression_distance_tree(Z):
    """Neighbour-joining tree of samples from a z-scored (samples x genes) table, 1 - correlation distance."""
    names = [str(s) for s in Z.index]
    C = np.nan_to_num(np.corrcoef(Z.values), nan=0.0)
    D = np.clip(1 - C, 0, 2)
    np.fill_diagonal(D, 0)
    lower = [[float(D[i, j]) for j in range(i + 1)] for i in range(len(names))]
    tree = DistanceTreeConstructor().nj(DistanceMatrix(names, lower))
    for cl in tree.get_nonterminals():
        cl.name = None
    for cl in tree.find_clades():
        if cl.branch_length is not None and cl.branch_length < 0:
            cl.branch_length = 0.0
    tree.root_at_midpoint()
    tree.ladderize()
    return tree, names, D


# =====================================================================================================
# STREAMLIT APP
# =====================================================================================================
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
      <div class="s">Differentially expressed genes, key cancer pathways, sequence alignment, phylogenetic trees, mutations and cancer vs healthy prediction</div>
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

st.sidebar.header("4. Sequence analysis")
species_sel = st.sidebar.multiselect("Species for alignments and trees", list(SPECIES), default=DEFAULT_SPECIES)
st.sidebar.caption("Protein sequences come from UniProt (internet needed once, then cached).")

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

tabs = st.tabs(["📊 Overview", "🌋 DEGs", "🔥 Heatmap", "🧭 Key Pathways", "🧮 Pathway Results", "🧬 Apoptosis Sequences & Tree", "🌳 Phylogenetic Trees", "🤖 Prediction", "🧪 Mutations", "ℹ️ Help"])

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

# ---- Key pathways (each pathway = expression sub-tab + multiple sequence alignment sub-tab)
def apo_role(g):
    return 1 if g in PRO else -1 if g in ANTI else PATHWAYS["Apoptosis"]["genes"].get(g, (0,))[0]


def pathway_expression_panel(pname):
    P = PATHWAYS[pname]
    info = pathway_gene_info(pname)
    if pname == "Apoptosis":                       # apoptosis genes come from the sidebar selection
        genes_all = list(dict.fromkeys(list(apoptosis) + list(P["genes"])))
        roles = {g: apo_role(g) for g in genes_all}
        present = [g for g in present_apo]
    else:
        roles = pathway_gene_roles(pname)
        present = [g for g in roles if g in result.index]
    total = len(roles) if pname != "Apoptosis" else len(set(apoptosis) | set(P["genes"]))
    st.caption(P["description"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Pathway genes in this data", f"{len(present)} / {total}")
    sig_genes = [g for g in present if result.loc[g, "DEG"]]
    c2.metric("Significant DEGs", len(sig_genes))
    c3.metric("Classes", f"{P['pos_label']} / {P['neg_label']}" if P["neg_label"] != "n/a" else P["pos_label"])
    if len(present) == 0:
        st.info("None of this pathway's genes are in the selected data. Sequence alignment still works in the next sub-tab. "
                "Try the real GSE42568 data or upload a table that contains these genes.")
        return
    mc = log_counts.loc[present, case_cols].mean(axis=1)
    mk = log_counts.loc[present, ctrl_cols].mean(axis=1)
    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(present) + 3), 4.0))
    ax.bar(x - 0.18, mc, 0.36, label=case, color="firebrick")
    ax.bar(x + 0.18, mk, 0.36, label=control, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=45 if len(present) > 8 else 0, ha="right" if len(present) > 8 else "center")
    ax.set_ylabel("mean log2 expression")
    ax.legend()
    ax.set_title(f"{pname} genes")
    st.pyplot(fig)

    table = result.loc[present, ["log2FC", "padj", "direction"]].copy()
    names = {1: P["pos_label"], -1: P["neg_label"], 0: "unknown"}
    table["role"] = [names[roles.get(g, 0)] if P["neg_label"] != "n/a" or roles.get(g, 0) != 1 else P["pos_label"]
                     for g in table.index]
    table["function"] = [info.get(g, PROTEIN_INFO.get(g, "")) for g in table.index]
    st.dataframe(table.round(4))
    st.download_button("Download pathway table (CSV)", table.round(6).to_csv().encode(),
                       f"{pname.replace(' ', '_').replace('/', '')}_genes.csv", key=f"dl_expr_{pname}")

    sig = table[(table["padj"] < padj_cut) & (table["log2FC"].abs() > fc_cut)]
    # per-sample pathway activity score
    if len(present) >= 2 and any(roles.get(g, 0) != 0 for g in present):
        z = log_counts.loc[present].sub(log_counts.loc[present].mean(axis=1), axis=0)
        z = z.div(log_counts.loc[present].std(axis=1) + 1e-9, axis=0)
        sgn = pd.Series({g: (roles.get(g, 0) or 0) for g in present})
        if (sgn == -1).any():
            score = z.loc[sgn[sgn == 1].index].mean(axis=0).fillna(0) - z.loc[sgn[sgn == -1].index].mean(axis=0).fillna(0)
            score_lab = f"{P['pos_label']} minus {P['neg_label']} (mean z-score)"
        else:
            score = z.loc[sgn[sgn == 1].index].mean(axis=0)
            score_lab = f"mean z-score of {P['pos_label']} genes"
        a, b = score[case_cols], score[ctrl_cols]
        pv = stats.ttest_ind(a, b).pvalue
        fig2, ax2 = plt.subplots(figsize=(4.6, 3.8))
        bp = ax2.boxplot([b, a], patch_artist=True, widths=0.5)
        ax2.set_xticklabels([control, case])
        for patch, col in zip(bp["boxes"], ("steelblue", "firebrick")):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        ax2.set_ylabel(score_lab, fontsize=8)
        ax2.set_title(f"Pathway activity score (t-test p = {pv:.2g})", fontsize=10)
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(fig2)
        with right:
            st.markdown("**Reading the result**")
            if pname == "Apoptosis":
                anti_up = [g for g in sig.index if roles.get(g) == -1 and sig.loc[g, "log2FC"] > 0]
                pro_down = [g for g in sig.index if roles.get(g) == 1 and sig.loc[g, "log2FC"] < 0]
                pro_up = [g for g in sig.index if roles.get(g) == 1 and sig.loc[g, "log2FC"] > 0]
                anti_down = [g for g in sig.index if roles.get(g) == -1 and sig.loc[g, "log2FC"] < 0]
                if len(anti_up) + len(pro_down) > len(pro_up) + len(anti_down):
                    st.success(f"Apoptosis looks **suppressed** in {case}: anti-apoptotic up ({', '.join(anti_up) or 'none'}), pro-apoptotic down ({', '.join(pro_down) or 'none'}).")
                elif len(pro_up) + len(anti_down) > len(anti_up) + len(pro_down):
                    st.warning(f"Apoptosis genes look **increased** in {case}: pro-apoptotic up ({', '.join(pro_up) or 'none'}), anti-apoptotic down ({', '.join(anti_down) or 'none'}).")
                elif sig.empty:
                    st.info("No apoptosis gene passes both cut-offs. Try relaxing them in the sidebar.")
                else:
                    st.info("Mixed pattern: some apoptosis genes go up and some go down.")
            else:
                up_g = [g for g in sig.index if sig.loc[g, "log2FC"] > 0]
                dn_g = [g for g in sig.index if sig.loc[g, "log2FC"] < 0]
                if sig.empty:
                    st.info("No gene of this pathway passes both cut-offs.")
                else:
                    st.info(f"Up in {case}: {', '.join(up_g) or 'none'}. Down in {case}: {', '.join(dn_g) or 'none'}.")
            direction = "higher" if a.mean() > b.mean() else "lower"
            st.write(f"Average pathway score is **{direction}** in {case} than in {control} "
                     + ("(significant)." if pv < 0.05 else "(not significant)."))
    st.caption("Expression-based hint only, not proof of pathway activity.")


def show_alignment(res, key):
    """Show an alignment result (dict with aln, engine, label, missing, seqs)."""
    aln = res["aln"]
    s = alignment_stats(aln)
    m = st.columns(5)
    m[0].metric("Sequences", s["sequences"])
    m[1].metric("Alignment length", s["length"])
    m[2].metric("Identical columns", f"{s['% identical columns']}%")
    m[3].metric("Mean identity", f"{s['mean pairwise identity %']}%")
    m[4].metric("Columns with gaps", s["columns with gaps"])
    st.caption(f"{res['label']} | engine: {res['engine']}"
               + (f" | not found: {', '.join(res['missing'])}" if res["missing"] else ""))
    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
    st.markdown(alignment_html(aln), unsafe_allow_html=True)

    cons = conservation(aln)
    g1, g2 = st.columns(2)
    with g1:
        win = max(1, min(15, len(cons) // 10))
        sm = np.convolve(cons, np.ones(win) / win, mode="same")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.fill_between(range(1, len(sm) + 1), sm, color="seagreen", alpha=0.6)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("alignment position")
        ax.set_ylabel("conservation")
        ax.set_title(f"Conservation (window {win})")
        st.pyplot(fig)
    with g2:
        names, mat = identity_matrix(aln)
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        im = ax.imshow(mat, cmap="viridis", vmin=max(0, mat.min() - 5), vmax=100)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        if len(names) <= 10:
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax, label="% identity", fraction=0.046)
        ax.set_title("Pairwise identity")
        st.pyplot(fig)
    d1, d2 = st.columns(2)
    d1.download_button("Download alignment (FASTA)", to_fasta(aln).encode(), "alignment.fasta", key=f"dla_{key}")
    d2.download_button("Download unaligned sequences (FASTA)", to_fasta(res["seqs"]).encode(), "sequences.fasta", key=f"dls_{key}")


def get_sequences(mode, pname, gene, family_genes, fasta_text, key):
    """Collect the sequences for one analysis. Returns (seqs, missing, label) or None."""
    if mode.startswith("Orthologs"):
        bar = st.progress(0.0, text="Downloading from UniProt...")
        seqs, missing = fetch_orthologs(gene, species_sel, progress=lambda f, t: bar.progress(f, text=t))
        bar.empty()
        return seqs, missing, f"{gene} orthologs across species"
    if mode.startswith("Related"):
        bar = st.progress(0.0, text="Downloading from UniProt...")
        seqs, missing = fetch_family(family_genes, "Human", progress=lambda f, t: bar.progress(f, text=t))
        bar.empty()
        return seqs, missing, f"{pname}: human proteins"
    seqs = parse_fasta(fasta_text or "")
    return seqs, [], "Your own sequences"


def pathway_msa_panel(pname):
    P = PATHWAYS[pname]
    gene_list = list(P["genes"])
    in_data = set(result.index)
    st.caption("Align protein sequences to see which regions are conserved. Sequences are downloaded from UniProt "
               "(internet needed the first time, then cached in `seq_cache.json`).")
    mode = st.radio("What to align", ["Orthologs of one gene across species", "Related genes of this pathway (human)",
                                      "Paste / upload my own FASTA"], key=f"mode_{pname}", horizontal=True)
    gene, family_genes, fasta_text = None, [], None
    if mode.startswith("Orthologs"):
        gene = st.selectbox("Gene", gene_list, key=f"gene_{pname}",
                            format_func=lambda g: f"{g}{' ✓' if g in in_data else ''}  -  {P['genes'][g][1]}")
        st.caption(f"Species used: {', '.join(species_sel) or 'none'} (change in the sidebar, section 4). ✓ = gene is in your expression data.")
    elif mode.startswith("Related"):
        family_genes = st.multiselect("Genes (2 to 15)", gene_list, default=gene_list[:6], key=f"fam_{pname}",
                                      max_selections=15)
        st.caption("Different genes of the same pathway only share local domains, so expect low identity. "
                   "This is most meaningful for real families, for example BCL-2 members or caspases.")
    else:
        up = st.file_uploader("FASTA file (optional)", type=["fa", "fasta", "txt", "faa"], key=f"up_{pname}")
        fasta_text = up.read().decode(errors="ignore") if up else ""
        fasta_text = st.text_area("...or paste FASTA here", fasta_text, height=140, key=f"txt_{pname}")
    engine = st.selectbox("Alignment program", available_engines(), key=f"eng_{pname}")
    if st.button("Fetch sequences and align", type="primary", key=f"btn_{pname}"):
        if mode.startswith("Orthologs") and len(species_sel) < 2:
            st.error("Choose at least 2 species in the sidebar (section 4).")
        elif mode.startswith("Related") and len(family_genes) < 2:
            st.error("Choose at least 2 genes.")
        else:
            seqs, missing, label = get_sequences(mode, pname, gene, family_genes, fasta_text, pname)
            if len(seqs) < 2:
                st.error("Fewer than 2 sequences available. Check the internet connection, gene names or the FASTA text."
                         + (f" Not found: {', '.join(missing)}." if missing else ""))
            else:
                with st.spinner("Aligning..."):
                    aln, used = run_msa(seqs, engine)
                st.session_state[f"msa_{pname}"] = {"aln": aln, "engine": used, "label": label, "missing": missing, "seqs": seqs}
    res = st.session_state.get(f"msa_{pname}")
    if res:
        show_alignment(res, pname)


with tabs[3]:
    st.subheader("Key cancer pathways")
    st.caption("Each pathway has two sub-tabs: **Expression** (your data) and **Multiple sequence alignment** (protein conservation).")
    ptabs = st.tabs([f"{PATHWAYS[p]['icon']} {p}" for p in PATHWAYS])
    for pt, pname in zip(ptabs, PATHWAYS):
        with pt:
            sub_expr, sub_msa = st.tabs(["📊 Expression", "🧫 Multiple sequence alignment"])
            with sub_expr:
                pathway_expression_panel(pname)
            with sub_msa:
                pathway_msa_panel(pname)

# ---- Pathway results (all pathways together: which pathways are changed in cancer?)
with tabs[4]:
    st.subheader("Pathway results: which cancer pathways are changed?")
    st.caption("Every pathway of the **Key Pathways** tab is tested here with your data. Two tests per pathway: "
               "(1) are the DEGs over-represented in the pathway (hypergeometric test), and (2) is the per-sample "
               "pathway score different between the two groups (Welch t-test). Both are corrected for the number of pathways (FDR).")
    ptab, pscores = pathway_table(result, log_counts, case_cols, ctrl_cols)
    if ptab.empty:
        st.info("None of the pathway genes are in this data. Use the real GSE42568 data or a table that contains pathway genes.")
    else:
        n_sc = int((ptab["score padj"] < 0.05).sum())
        n_en = int((ptab["enrichment padj"] < 0.05).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pathways testable", len(ptab))
        c2.metric("Score changed (FDR < 0.05)", n_sc)
        c3.metric("DEGs enriched (FDR < 0.05)", n_en)
        c4.metric("Pathway DEGs in total", int(ptab["DEGs"].sum()))

        show = ptab.sort_values("score padj")[["pathway", "genes in data", "DEGs", "up", "down", "enrichment padj",
                                               "score diff", "effect size (d)", "score padj", "score direction", "DEG genes"]]
        st.dataframe(show.round(4), use_container_width=True)
        st.download_button("Download pathway results (CSV)", show.round(6).to_csv(index=False).encode(), "pathway_results.csv")

        left, right = st.columns(2)
        with left:
            pl = ptab.dropna(subset=["effect size (d)"]).sort_values("effect size (d)")
            fig, ax = plt.subplots(figsize=(5.8, 0.38 * len(pl) + 1.6))
            cols = ["firebrick" if (p < 0.05 and d > 0) else "steelblue" if (p < 0.05 and d < 0) else "lightgrey"
                    for p, d in zip(pl["score padj"], pl["effect size (d)"])]
            ax.barh(pl["pathway"], pl["effect size (d)"], color=cols)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel(f"pathway score: effect size ({case} - {control})")
            ax.set_title("Red = higher in case, blue = lower (FDR < 0.05), grey = n.s.", fontsize=9)
            st.pyplot(fig)
        with right:
            if pscores:
                S = pd.DataFrame(pscores).T                       # pathways x samples
                S = S.sub(S.mean(axis=1), axis=0).div(S.std(axis=1) + 1e-9, axis=0)
                order = list(case_cols) + list(ctrl_cols)
                S = S[order]
                fig, ax = plt.subplots(figsize=(5.8, 0.38 * len(S) + 1.6))
                im = ax.imshow(S.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
                ax.set_yticks(range(len(S)))
                ax.set_yticklabels(S.index, fontsize=8)
                ax.set_xticks([len(case_cols) / 2 - 0.5, len(case_cols) + len(ctrl_cols) / 2 - 0.5])
                ax.set_xticklabels([case, control])
                ax.axvline(len(case_cols) - 0.5, color="black", lw=1)
                fig.colorbar(im, ax=ax, label="pathway score (z)", fraction=0.04)
                ax.set_title("Pathway score in every sample", fontsize=9)
                st.pyplot(fig)

        st.markdown("#### Result for each pathway")
        pick = st.selectbox("Pathway", list(ptab["pathway"]), key="pr_pick",
                            format_func=lambda p: f"{PATHWAYS[p]['icon']} {p}")
        row = ptab[ptab["pathway"] == pick].iloc[0]
        PP = PATHWAYS[pick]
        st.caption(PP["description"])
        roles_p = pathway_gene_roles(pick)
        pres = [g for g in roles_p if g in result.index]
        gt = result.loc[pres, ["log2FC", "padj", "direction"]].copy()
        gt["role"] = [PP["pos_label"] if roles_p[g] == 1 else PP["neg_label"] if roles_p[g] == -1 else "member" for g in gt.index]
        gt["function"] = [PP["genes"][g][1] for g in gt.index]
        g1, g2 = st.columns([1, 1])
        with g1:
            fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(gt) + 1.4))
            gs_ = gt.sort_values("log2FC")
            ax.barh(gs_.index, gs_["log2FC"], color=["firebrick" if v > 0 else "steelblue" for v in gs_["log2FC"]],
                    edgecolor=["black" if result.loc[g, "DEG"] else "none" for g in gs_.index])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel(f"log2 fold change ({case} vs {control}); black outline = DEG")
            ax.set_title(f"{pick}: genes in your data", fontsize=10)
            st.pyplot(fig)
        with g2:
            st.dataframe(gt.round(4), use_container_width=True)
        bits = [f"**{pick}**: {int(row['genes in data'])} of {int(row['genes in pathway'])} genes are in the data, "
                f"{int(row['DEGs'])} are DEGs ({int(row['up'])} up, {int(row['down'])} down in {case})."]
        if not np.isnan(row["score padj"]):
            bits.append(f"Pathway score is **{row['score direction']}** (effect size {row['effect size (d)']:.2f}, FDR {row['score padj']:.3g}).")
        bits.append(f"DEG over-representation FDR: {row['enrichment padj']:.3g}.")
        st.info(" ".join(bits))

        # ---- reading the whole result
        st.markdown("#### How does the cancer look like it arises here?")
        sig = ptab[ptab["score padj"] < 0.05].sort_values("score padj")
        lines = []
        if sig.empty:
            lines.append("No pathway passes FDR < 0.05 with its score. Expression of these pathway genes does not clearly differ between the groups in this data.")
        else:
            lines.append("Pathways with a significantly different score: " + ", ".join(
                f"**{r['pathway']}** ({'higher' if r['score diff'] > 0 else 'lower'} in {case})"
                for _, r in sig.head(6).iterrows()) + ".")
        if "Apoptosis" in set(ptab["pathway"]):
            ar = ptab[ptab["pathway"] == "Apoptosis"].iloc[0]
            if not np.isnan(ar["score padj"]):
                state = "lower (apoptosis is switched down)" if ar["score diff"] < 0 else "higher"
                lines.append(f"Apoptosis score is {state} in {case} (FDR {ar['score padj']:.3g}); {int(ar['DEGs'])} apoptosis genes are DEGs.")
        for ln in lines:
            st.markdown("- " + ln)
        st.caption("Expression shows which pathways are deregulated in the tumour. It does not prove what started the cancer: "
                   "a pathway can change because of mutations, copy number, methylation or the tissue mix.")

        st.markdown("#### SNP / mutation evidence for the same pathways")
        if mutations is None or not {"sample", "gene"}.issubset(mutations.columns):
            st.info("No mutation table loaded. To compare mutation (SNP) burden with expression, choose **Upload my own** "
                    "in the sidebar and add a mutations CSV (columns: sample, gene, optional protein_change).")
        else:
            mt = mutation_pathway_table(mutations, sub, case, control)
            mt["Fisher FDR"] = multipletests(mt["Fisher p"], method="fdr_bh")[1]
            st.dataframe(mt.round(4), use_container_width=True)
            mm = mt.set_index("pathway")[[f"% {case} mutated", f"% {control} mutated"]]
            st.bar_chart(mm)
            both = sorted(set(sig["pathway"]) & set(mt.loc[mt["Fisher FDR"] < 0.05, "pathway"]))
            only_expr = sorted(set(sig["pathway"]) - set(mt.loc[mt["Fisher FDR"] < 0.05, "pathway"]))
            st.markdown(f"- Changed in expression **and** mutated more often: {', '.join(both) or 'none'}.")
            st.markdown(f"- Changed in expression but **not** enriched for mutations: {', '.join(only_expr) or 'none'}. "
                        "These pathways change without a clear SNP signal, which fits your idea that the pathway, not a single SNP, is what differs.")


# ---- Apoptosis sequences: control vs disease -> alignment -> phylogenetic tree -> cluster separation
def show_separation(stats_, mono, case_, control_):
    if stats_ is None:
        st.info("Not enough samples to test separation.")
        return
    m = st.columns(4)
    m[0].metric("Within-group distance", f"{stats_['within']:.4f}")
    m[1].metric("Between-group distance", f"{stats_['between']:.4f}")
    m[2].metric("Silhouette", f"{stats_['silhouette']:.2f}")
    m[3].metric("Permutation p", f"{stats_['perm_p']:.3g}")
    st.markdown(separation_verdict(stats_, case_, control_))
    if mono:
        st.caption("Each group forms one clade of the tree: " + ", ".join(f"{k}: {'yes' if v else 'no (mixed with the other group)'}" for k, v in mono.items()))


def apoptosis_expression_clusters(genes):
    """Samples clustered by apoptosis gene expression (always available, uses your expression data)."""
    genes = [g for g in genes if g in log_counts.index]
    if len(genes) < 3:
        st.info("Need at least 3 apoptosis genes present in the expression data for this tree.")
        return
    X = log_counts.loc[genes].T
    Z = (X - X.mean()) / (X.std() + 1e-9)
    tree, names, D = expression_distance_tree(Z)
    groups = {str(s): ("disease" if sub.loc[s, "group"] == case else "control") for s in Z.index}
    lc, rc = st.columns([1, 1])
    with lc:
        fig, ax = plt.subplots(figsize=(6.2, max(4.0, 0.2 * len(names) + 1.2)))
        draw_group_tree(tree, ax, groups, f"Samples clustered by {len(genes)} apoptosis genes (red = {case}, blue = {control})")
        st.pyplot(fig)
    with rc:
        order = [t.name for t in tree.get_terminals()]
        zz = Z.copy()
        zz.index = names
        zz = zz.loc[order].T
        fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.3 * len(genes) + 1.4)))
        im = ax.imshow(zz.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(zz.index)))
        ax.set_yticklabels(zz.index, fontsize=8)
        ax.set_xticks([])
        bar_cols = [GROUP_COLORS[groups[o]] for o in order]
        for i, c in enumerate(bar_cols):
            ax.add_patch(plt.Rectangle((i - 0.5, -1.2), 1, 0.7, color=c, clip_on=False))
        fig.colorbar(im, ax=ax, label="z-score", fraction=0.04)
        ax.set_title("Apoptosis gene expression, samples in tree order", fontsize=9, pad=18)
        st.pyplot(fig)
    sep = group_separation(names, D, groups)
    # also a flat 2-cluster cut
    link = linkage(Z.values, method="ward")
    cl = fcluster(link, 2, criterion="maxclust")
    ct = pd.crosstab(pd.Series(["disease" if sub.loc[s, "group"] == case else "control" for s in Z.index], name="true group"),
                     pd.Series(cl, name="cluster"))
    show_separation(sep, group_monophyly(tree, groups), case, control)
    st.markdown("**If the samples are cut into 2 clusters (Ward linkage):**")
    st.dataframe(ct)
    if ct.shape == (2, 2):
        pf = stats.fisher_exact(ct.values)[1]
        purity = ct.max(axis=0).sum() / ct.values.sum()
        st.caption(f"Cluster purity {purity:.0%}, Fisher exact p = {pf:.3g}.")
    st.download_button("Download this tree (Newick)", tree_newick(tree).encode(), "apoptosis_expression_tree.nwk", key="dl_exprtree")


with tabs[5]:
    st.subheader("Apoptosis genes: sequences of control and disease, alignment and phylogenetic tree")
    st.warning("**Where do sample-wise sequences come from?** An expression table has no sequence in it. This tab gets the "
               "sequence of every sample in one of two ways: (1) the **UniProt reference protein** of the gene plus the sample's own "
               "variants from a mutation CSV (column `protein_change`, for example `p.R175H`), or (2) a **FASTA** you upload "
               "(for example consensus sequences from your own reads after mapping and variant calling in Galaxy). "
               "Samples without a variant keep the reference sequence.")
    apo_pool = list(dict.fromkeys(list(apoptosis) + list(PATHWAYS["Apoptosis"]["genes"])))
    apo_deg = [g for g in apo_pool if g in result.index and result.loc[g, "DEG"]]
    default_genes = (apo_deg or [g for g in apo_pool if g in result.index] or apo_pool)[:5]
    sel_genes = st.multiselect("Apoptosis genes (1 to 8)", apo_pool, default=default_genes, key="as_genes", max_selections=8,
                               format_func=lambda g: f"{g}{' (DEG)' if g in apo_deg else ''}")
    src = st.radio("Sequence source", ["Reference (UniProt) + sample variants from mutation CSV",
                                       "Upload FASTA of sample sequences (headers: SAMPLE|GENE)"], key="as_src")
    fasta_in = ""
    mut_in, change_col = mutations, None
    if src.startswith("Reference"):
        up_m = st.file_uploader("Mutation CSV with columns sample, gene, protein_change (optional, overrides the sidebar file)",
                                type="csv", key="as_mut")
        if up_m is not None:
            mut_in = pd.read_csv(up_m)
        if mut_in is not None and {"sample", "gene"}.issubset(mut_in.columns):
            cands = [c for c in mut_in.columns if c.lower() in ("protein_change", "aa_change", "hgvsp", "mutation", "amino_acid_change")]
            if cands:
                change_col = cands[0]
                st.caption(f"Using column **{change_col}** for amino-acid changes.")
            else:
                st.info("The mutation table has no protein-change column (protein_change, aa_change, HGVSp). "
                        "Without it every sample keeps the reference sequence, so nothing can separate the groups.")
        else:
            st.info("No mutation table with 'sample' and 'gene' columns. Every sample will keep the reference sequence.")
    else:
        up_f = st.file_uploader("FASTA file (>SAMPLE|GENE)", type=["fa", "fasta", "txt", "faa"], key="as_fa")
        fasta_in = up_f.read().decode(errors="ignore") if up_f else ""
        fasta_in = st.text_area("...or paste FASTA here", fasta_in, height=120, key="as_fa_txt")
    o1, o2, o3, o4 = st.columns(4)
    as_engine = o1.selectbox("Alignment program", available_engines(), key="as_eng")
    as_max = o2.slider("Max samples per group", 5, 60, 15, 5, key="as_max")
    as_method = o3.selectbox("Tree method", ["Neighbour joining", "UPGMA"], key="as_meth")
    as_boot = o4.slider("Bootstrap (0 = none)", 0, 100, 0, 10, key="as_boot")

    if st.button("Extract sequences, align and build trees", type="primary", key="as_btn"):
        try:
            if len(sel_genes) < 1:
                raise ValueError("Choose at least one gene.")
            rng_ = np.random.default_rng(1)

            def take(cols):
                cols = list(cols)
                if len(cols) > as_max:
                    cols = [cols[i] for i in sorted(rng_.choice(len(cols), as_max, replace=False))]
                return cols
            chosen = take(ctrl_cols) + take(case_cols)
            gmap = {s: ("disease" if sub.loc[s, "group"] == case else "control") for s in chosen}
            out = {"genes": {}, "missing": [], "report": [], "refs": {}}
            fa_by_gene = {}
            if src.startswith("Upload"):
                recs = parse_fasta(fasta_in or "")
                if not recs:
                    raise ValueError("No FASTA sequences found.")
                lookup = {str(s): s for s in chosen}
                for h, sq in recs.items():
                    if "|" not in h:
                        continue
                    smp, gn = h.rsplit("|", 1)
                    if smp in lookup:
                        fa_by_gene.setdefault(gn.upper(), {})[lookup[smp]] = sq
            bar = st.progress(0.0, text="Getting sequences...")
            for k, g in enumerate(sel_genes):
                bar.progress(k / len(sel_genes), text=f"{g}: sequences")
                if src.startswith("Reference"):
                    rec = fetch_protein(g, SPECIES["Human"])
                    if not rec:
                        out["missing"].append(g)
                        continue
                    ref = re.sub(r"[UO]", "X", rec["sequence"])
                    out["refs"][g] = ref
                    if change_col:
                        seqs, groups, rows = sample_sequences_from_mutations(ref, g, mut_in, chosen, gmap, change_col)
                    else:
                        seqs = {f"{gmap[s]}_{safe_name(s)}": ref for s in chosen}
                        groups = {f"{gmap[s]}_{safe_name(s)}": gmap[s] for s in chosen}
                        rows = [{"sample": s, "group": gmap[s], "gene": g, "variants applied": "",
                                 "skipped (not simple / wrong residue)": ""} for s in chosen]
                    seqs, groups = dict(seqs), dict(groups)
                    seqs["REF"] = ref
                    groups["REF"] = "reference"
                else:
                    got = fa_by_gene.get(g.upper(), {})
                    if len(got) < 4:
                        out["missing"].append(g)
                        continue
                    seqs = {f"{gmap[s]}_{safe_name(s)}": re.sub(r"[^A-Z]", "", v.upper()).replace("U", "X").replace("O", "X")
                            for s, v in got.items()}
                    groups = {f"{gmap[s]}_{safe_name(s)}": gmap[s] for s in got}
                    rows = [{"sample": s, "group": gmap[s], "gene": g, "variants applied": "from FASTA",
                             "skipped (not simple / wrong residue)": ""} for s in got]
                msa = msa_with_haplotypes(seqs, groups, as_engine)
                entry = {"seqs": seqs, "groups": groups, "msa": msa, "rows": rows, "tree": None, "sep": None, "mono": None}
                sample_groups = {n: v for n, v in groups.items() if v in ("control", "disease")}
                if msa["n_hap"] > 1:
                    names_, D_ = aln_distance({n: msa["aln"][n] for n in sample_groups})
                    entry["sep"] = group_separation(names_, D_, sample_groups)
                    if len(msa["aln"]) >= 3:
                        tree_ = build_tree(msa["aln"], "nj" if as_method.startswith("Neigh") else "upgma", bootstrap=as_boot)
                        entry["tree"] = tree_
                        entry["mono"] = group_monophyly(tree_, groups)
                out["genes"][g] = entry
            bar.empty()
            if not out["genes"]:
                raise ValueError("No sequences could be collected. Missing: " + (", ".join(out["missing"]) or "all genes")
                                 + ". Check the internet connection (UniProt) or the FASTA headers.")
            # all genes joined
            if len(out["genes"]) >= 2:
                per = {g: e["msa"]["aln"] for g, e in out["genes"].items()}
                common = set.intersection(*[set(a) for a in per.values()])
                if len(common) >= 4:
                    per = {g: {n: a[n] for n in a if n in common} for g, a in per.items()}
                    cat, _ = concatenate_alignments(per)
                    cgroups = {n: v for n, v in next(iter(out["genes"].values()))["groups"].items() if n in cat}
                    if len(set(cat.values())) > 1:
                        sg = {n: v for n, v in cgroups.items() if v in ("control", "disease")}
                        names_, D_ = aln_distance({n: cat[n] for n in sg})
                        ctree = build_tree(cat, "nj" if as_method.startswith("Neigh") else "upgma", bootstrap=as_boot)
                        out["combined"] = {"tree": ctree, "aln": cat, "groups": cgroups, "sep": group_separation(names_, D_, sg),
                                           "mono": group_monophyly(ctree, cgroups), "genes": list(per)}
            st.session_state["apo_seq"] = out
        except Exception as e:
            st.session_state.pop("apo_seq", None)
            st.error(f"Could not finish: {e}")

    res_seq = st.session_state.get("apo_seq")
    if res_seq:
        if res_seq["missing"]:
            st.caption("No sequence for: " + ", ".join(res_seq["missing"]))
        gnames = list(res_seq["genes"])
        st.markdown("### 1. Sequences and alignment per gene")
        gtabs = st.tabs(gnames)
        for gt_, g in zip(gtabs, gnames):
            e = res_seq["genes"][g]
            with gt_:
                rep = pd.DataFrame(e["rows"])
                n_var = rep[rep["variants applied"] != ""].groupby("group")["sample"].nunique()
                ncs = int((rep["group"] == "control").sum())
                nds = int((rep["group"] == "disease").sum())
                m = st.columns(4)
                m[0].metric("Control samples", ncs)
                m[1].metric("Disease samples", nds)
                m[2].metric("Distinct sequences", e["msa"]["n_hap"])
                m[3].metric("Samples with a variant (ctrl / disease)", f"{int(n_var.get('control', 0))} / {int(n_var.get('disease', 0))}")
                if e["msa"]["n_hap"] == 1:
                    st.info("All samples have the same sequence for this gene, so there is nothing to align or separate. "
                            "Add variants (mutation CSV with protein_change) or sample FASTA files.")
                else:
                    st.markdown(f"**Multiple sequence alignment** of the distinct sequences (names show how many control and disease samples carry each): "
                                f"engine {e['msa']['used']}")
                    st.markdown(LEGEND_HTML, unsafe_allow_html=True)
                    st.markdown(alignment_html(e["msa"]["hap_aln"]), unsafe_allow_html=True)
                    if e["tree"] is not None:
                        fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.25 * len(e["tree"].get_terminals()) + 1.2)))
                        draw_group_tree(e["tree"], ax, e["groups"], f"{g}: phylogenetic tree (red = {case}, blue = {control})")
                        st.pyplot(fig)
                        show_separation(e["sep"], e["mono"], case, control)
                with st.expander("Variants applied to each sample"):
                    st.dataframe(rep)
                ctrl_fa = {n: s for n, s in e["seqs"].items() if e["groups"].get(n) == "control"}
                dis_fa = {n: s for n, s in e["seqs"].items() if e["groups"].get(n) == "disease"}
                d1, d2, d3 = st.columns(3)
                d1.download_button(f"{g} control sequences (FASTA)", to_fasta(ctrl_fa).encode(), f"{g}_control.fasta", key=f"dlc_{g}")
                d2.download_button(f"{g} disease sequences (FASTA)", to_fasta(dis_fa).encode(), f"{g}_disease.fasta", key=f"dld_{g}")
                d3.download_button(f"{g} alignment (FASTA)", to_fasta(e["msa"]["aln"]).encode(), f"{g}_alignment.fasta", key=f"dla_{g}")

        st.markdown("### 2. Phylogenetic tree and control vs disease cluster separation")
        comb = res_seq.get("combined")
        if comb:
            fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.25 * len(comb["tree"].get_terminals()) + 1.2)))
            draw_group_tree(comb["tree"], ax, comb["groups"], f"All selected genes joined ({', '.join(comb['genes'])})")
            st.pyplot(fig)
            show_separation(comb["sep"], comb["mono"], case, control)
            st.download_button("Download combined tree (Newick)", tree_newick(comb["tree"]).encode(), "apoptosis_sequence_tree.nwk", key="dl_comb")
        elif len(gnames) >= 2:
            st.info("The sequences of the selected genes are identical in all samples, so no combined sequence tree can be drawn.")
        else:
            st.caption("Select 2 or more genes to also get a tree from all genes joined. The single-gene tree is in section 1.")

    st.markdown("### 3. Control vs disease clusters from apoptosis gene expression (your data)")
    st.caption("This part needs no sequences: samples are clustered by their expression of the selected apoptosis genes, drawn as a "
               "neighbour-joining tree. If control and disease form separate clusters, apoptosis expression alone tells them apart.")
    apoptosis_expression_clusters(sel_genes if len(sel_genes) >= 3 else [g for g in apo_pool if g in result.index])


# ---- Phylogenetic trees
with tabs[6]:
    st.subheader("Phylogenetic trees")
    st.caption("Trees are built from protein alignments (BLOSUM62 distances) with neighbour joining or UPGMA. "
               "Numbers on branches are bootstrap support (%): above ~70 is usually considered reliable.")
    pp = st.selectbox("Pathway", list(PATHWAYS), key="phy_path", format_func=lambda p: f"{PATHWAYS[p]['icon']} {p}")
    PG = PATHWAYS[pp]
    pgenes = list(PG["genes"])
    tmode = st.radio("Tree type", ["Gene tree: orthologs of one gene across species",
                                   "Gene family tree: genes of this pathway (human)",
                                   "Species tree: several genes joined together",
                                   "Paste / upload my own FASTA"], key="phy_mode")
    tgene, tfam, tcat, tfasta = None, [], [], None
    if tmode.startswith("Gene tree"):
        tgene = st.selectbox("Gene", pgenes, key="phy_gene", format_func=lambda g: f"{g}  -  {PG['genes'][g][1]}")
    elif tmode.startswith("Gene family"):
        tfam = st.multiselect("Genes (3 to 20)", pgenes, default=pgenes[:8], key="phy_fam", max_selections=20)
    elif tmode.startswith("Species"):
        tcat = st.multiselect("Genes to join (2 to 6)", pgenes, default=pgenes[:3], key="phy_cat", max_selections=6)
        st.caption("Every gene is aligned separately across the species of the sidebar, then the alignments are joined "
                   "end to end. Only species that have all genes are used.")
    else:
        upf = st.file_uploader("FASTA file (optional)", type=["fa", "fasta", "txt", "faa"], key="phy_up")
        tfasta = upf.read().decode(errors="ignore") if upf else ""
        tfasta = st.text_area("...or paste FASTA here (aligned or unaligned)", tfasta, height=140, key="phy_txt")
    oc1, oc2, oc3 = st.columns(3)
    tmethod = oc1.selectbox("Method", ["Neighbour joining", "UPGMA"], key="phy_method")
    tboot = oc2.slider("Bootstrap replicates (0 = none)", 0, 200, 50, 10, key="phy_boot")
    tengine = oc3.selectbox("Alignment program", available_engines(), key="phy_engine")

    if st.button("Build tree", type="primary", key="phy_btn"):
        try:
            title, aln_used, missing = "", None, []
            if tmode.startswith("Gene tree"):
                if len(species_sel) < 3:
                    raise ValueError("Choose at least 3 species in the sidebar (section 4).")
                bar = st.progress(0.0, text="Downloading from UniProt...")
                seqs, missing = fetch_orthologs(tgene, species_sel, progress=lambda f, t: bar.progress(f, text=t))
                bar.empty()
                title = f"{tgene}: orthologs"
                aln_used, eng = run_msa(seqs, tengine)
            elif tmode.startswith("Gene family"):
                if len(tfam) < 3:
                    raise ValueError("Choose at least 3 genes.")
                bar = st.progress(0.0, text="Downloading from UniProt...")
                seqs, missing = fetch_family(tfam, "Human", progress=lambda f, t: bar.progress(f, text=t))
                bar.empty()
                title = f"{pp}: gene family tree (human)"
                aln_used, eng = run_msa(seqs, tengine)
            elif tmode.startswith("Species"):
                if len(tcat) < 2 or len(species_sel) < 3:
                    raise ValueError("Choose at least 2 genes and (sidebar) at least 3 species.")
                per_gene, seqs, missing = {}, {}, []
                bar = st.progress(0.0, text="Downloading from UniProt...")
                for k, g in enumerate(tcat):
                    sq, ms = fetch_orthologs(g, species_sel)
                    missing += [f"{g}/{m}" for m in ms]
                    if len(sq) >= 3:
                        per_gene[g], eng = run_msa(sq, tengine)
                    bar.progress((k + 1) / len(tcat), text=f"{g} aligned")
                bar.empty()
                if len(per_gene) < 2:
                    raise ValueError("Fewer than 2 genes could be downloaded for at least 3 species.")
                cat, bounds = concatenate_alignments(per_gene)
                if len(cat) < 3:
                    raise ValueError("Fewer than 3 species have all the chosen genes.")
                aln_used = cat
                seqs = {s: v.replace("-", "") for s, v in cat.items()}
                title = f"Species tree from {', '.join(per_gene)}"
            else:
                seqs = parse_fasta(tfasta or "")
                if len(seqs) < 3:
                    raise ValueError("Need at least 3 sequences in the FASTA text.")
                lens = {len(v) for v in seqs.values()}
                if len(lens) == 1 and any("-" in v for v in seqs.values()):
                    aln_used, eng = seqs, "already aligned"
                else:
                    aln_used, eng = run_msa(seqs, tengine)
                title = "Your sequences"
            if len(aln_used) < 3:
                raise ValueError("Fewer than 3 sequences were found (" + (", ".join(missing) or "check internet") + ").")
            with st.spinner("Building tree..." + (" (bootstrap can take a moment)" if tboot else "")):
                tree = build_tree(aln_used, "nj" if tmethod.startswith("Neigh") else "upgma", bootstrap=tboot)
            st.session_state["phylo"] = {"tree": tree, "aln": aln_used, "title": title, "missing": missing,
                                         "method": tmethod, "boot": tboot}
        except Exception as e:
            st.session_state.pop("phylo", None)
            st.error(f"Could not build the tree: {e}")

    ph = st.session_state.get("phylo")
    if ph:
        n_tips = len(ph["tree"].get_terminals())
        fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.42 * n_tips + 1.2)))
        draw_tree(ph["tree"], ax, f"{ph['title']} ({ph['method']}" + (f", {ph['boot']} bootstrap)" if ph["boot"] else ")"))
        st.pyplot(fig)
        if ph["missing"]:
            st.caption("Not found in UniProt: " + ", ".join(ph["missing"]))
        names, mat = identity_matrix(ph["aln"])
        with st.expander("Pairwise identity table (%)"):
            st.dataframe(pd.DataFrame(mat, index=names, columns=names).round(1))
        import io as _io
        png = _io.BytesIO()
        fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
        b1, b2, b3 = st.columns(3)
        b1.download_button("Download tree (Newick)", tree_newick(ph["tree"]).encode(), "tree.nwk", key="dl_nwk")
        b2.download_button("Download tree (PNG)", png.getvalue(), "tree.png", mime="image/png", key="dl_png")
        b3.download_button("Download alignment used (FASTA)", to_fasta(ph["aln"]).encode(), "alignment_for_tree.fasta", key="dl_phyaln")
        st.caption("Distance trees from a single gene are a quick teaching tool. For publication use a maximum-likelihood program "
                   "such as IQ-TREE or RAxML on a trimmed alignment.")

# ---- Prediction
def prediction_tab():
    X = to_log(sub[gene_cols])
    y = (sub["group"] == case).astype(int)
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        st.warning("Not enough samples per group to split into training and test sets.")
        return

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


with tabs[7]:
    prediction_tab()

# ---- Mutations
with tabs[8]:
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
with tabs[9]:
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

**Key Pathways tab:** 12 pathways (apoptosis, p53, cell cycle, PI3K-AKT-mTOR, MAPK/RAS, Wnt, NF-kB, DNA repair, hormone receptor, hypoxia, EMT, autophagy). Every pathway has an *Expression* sub-tab (your data) and a *Multiple sequence alignment* sub-tab. Edit `pathways.py` to add genes or whole pathways.

**Multiple sequence alignment:** protein sequences are downloaded from UniProt (internet needed once, then saved in `seq_cache.json`). Align orthologs of one gene across species, related genes of a pathway, or your own FASTA. The built-in aligner needs no extra software; if MAFFT, Clustal Omega or MUSCLE is installed it appears in the program list.

**Phylogenetic Trees tab:** gene trees (orthologs), gene-family trees, species trees (joined genes) or trees from your own FASTA. Neighbour joining or UPGMA, optional bootstrap, download as Newick or PNG.

**Real breast cancer data:** GSE42568 (public, GEO) has 104 breast cancer and 17 normal breast tissue samples. It appears as a data source once the file `breast_cancer_gse42568.csv` is in the repository.

**Pathway Results tab:** all 12 pathways tested together (DEG over-representation and pathway-score test, FDR corrected), a per-pathway result view, and a mutation (SNP) comparison when a mutation table is available.

**Apoptosis Sequences & Tree tab:** sequences of the apoptosis genes for control and disease samples (UniProt reference + sample variants, or your own FASTA), multiple sequence alignment, phylogenetic tree coloured by group and a control vs disease cluster-separation test. Raw reads (FASTQ) must first be mapped and variant-called (for example in Galaxy) before they can be used here.

**Where the values come from:** Galaxy (featureCounts) or a GEO / TCGA table.

**Not covered:** alignment and read counting are done in Galaxy. Protein levels need proteomics data.
        """
    )
