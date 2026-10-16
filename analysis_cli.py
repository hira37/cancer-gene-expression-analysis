"""
Cancer Gene Expression Analysis: terminal (command line) version.

Run:   python analysis_cli.py
Needs: pandas, numpy, scipy, matplotlib

Input: a CSV file, one row per sample, one column per gene (read counts),
and a column named 'label' (for example cancer / healthy).
Optional: a mutation CSV with columns sample, gene, mutation_type.
Results (tables and plots) are saved in the folder 'results'.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save plots as files, no pop-up windows
import matplotlib.pyplot as plt
from scipy import stats

OUT = "results"
PRO = {"TP53", "BAX", "BAK1", "CASP3", "CASP8", "CASP9", "BID", "BBC3", "FAS"}
ANTI = {"BCL2", "BCL2L1", "MCL1", "XIAP", "BIRC5"}
PROTEIN = {
    "TP53": "p53, tumour suppressor", "BAX": "BAX, pro-apoptotic protein",
    "BCL2": "BCL-2, anti-apoptotic protein", "CASP3": "Caspase-3, executes apoptosis",
    "MYC": "MYC, growth transcription factor", "EGFR": "EGF receptor, growth signalling",
    "GAPDH": "housekeeping enzyme", "ACTB": "beta-actin, housekeeping",
    "VEGFA": "VEGF-A, blood vessel growth", "CDKN1A": "p21, cell cycle inhibitor",
}


def ask(text, default=None):
    answer = input(f"{text}" + (f" [{default}]" if default else "") + ": ").strip()
    return answer or default


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def load_data():
    path = ask("Counts CSV file", "simulated_counts.csv")
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    data = pd.read_csv(path, index_col=0)
    if "label" not in data.columns:
        sys.exit("The file needs a column named 'label'.")
    genes = [c for c in data.columns if c != "label"]
    data[genes] = data[genes].apply(pd.to_numeric, errors="coerce")
    if data[genes].isna().any().any():
        sys.exit("Gene columns must contain numbers only.")
    labels = sorted(data["label"].astype(str).unique())
    print("Groups found:", ", ".join(f"{l} ({(data['label'].astype(str) == l).sum()})" for l in labels))
    control = ask("Control group", "healthy" if "healthy" in labels else labels[0])
    case = ask("Case group", "cancer" if "cancer" in labels else labels[-1])
    if control not in labels or case not in labels or control == case:
        sys.exit("Choose two different groups from the list above.")
    sub = data[data["label"].astype(str).isin([control, case])].copy()
    sub["group"] = np.where(sub["label"].astype(str) == case, case, control)

    mut_path = ask("Mutation CSV (press Enter to skip)", "simulated_mutations.csv" if os.path.exists("simulated_mutations.csv") else "")
    muts = pd.read_csv(mut_path) if mut_path and os.path.exists(mut_path) else None
    return sub, genes, control, case, muts


def find_degs(sub, genes, control, case, padj_cut=0.05, fc_cut=1.0):
    logc = np.log2(sub[genes] + 1)
    a = logc[sub["group"] == case]
    b = logc[sub["group"] == control]
    res = pd.DataFrame(index=genes)
    res["log2FC"] = a.mean() - b.mean()
    res["p_value"] = np.nan_to_num(stats.ttest_ind(a, b).pvalue, nan=1.0)
    res["padj"] = bh_adjust(res["p_value"])
    res["DEG"] = (res["padj"] < padj_cut) & (res["log2FC"].abs() > fc_cut)
    res["direction"] = np.where(~res["DEG"], "-", np.where(res["log2FC"] > 0, f"up in {case}", f"down in {case}"))
    res["protein"] = [PROTEIN.get(g, "") for g in genes]
    return res, logc


def cmd_overview(sub, genes, control, case):
    print(f"\nSamples: {len(sub)}   {case}: {(sub['group'] == case).sum()}   {control}: {(sub['group'] == control).sum()}   Genes: {len(genes)}")
    totals = sub.groupby("group")[genes].sum().sum(axis=1)
    print("Total read counts per group:")
    print(totals.to_string())
    print("\nMean read counts per gene:")
    print(sub.groupby("group")[genes].mean().T.round(1).to_string())


def cmd_degs(res, case, logc, sub, control, apoptosis):
    os.makedirs(OUT, exist_ok=True)
    print("\nDEG results (sorted by adjusted p-value):")
    print(res.sort_values("padj")[["log2FC", "p_value", "padj", "DEG", "direction", "protein"]].round(4).to_string())
    print(f"\nDEGs found: {int(res['DEG'].sum())}  ->  {', '.join(res.index[res['DEG']])}")
    res.round(6).to_csv(f"{OUT}/deg_results.csv")

    plot = -np.log10(res["padj"] + 1e-300)
    plt.figure(figsize=(7, 5))
    plt.scatter(res["log2FC"], plot, c=res["DEG"].map({True: "red", False: "grey"}), s=60)
    for g in res.index:
        plt.annotate(g + ("*" if g in apoptosis else ""), (res.loc[g, "log2FC"], plot[g]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.axvline(1, ls="--", c="black", lw=.8); plt.axvline(-1, ls="--", c="black", lw=.8)
    plt.axhline(-np.log10(0.05), ls="--", c="black", lw=.8)
    plt.xlabel(f"log2 fold change ({case} vs {control})"); plt.ylabel("-log10(adjusted p-value)")
    plt.title("Volcano plot (* = apoptosis genes)")
    plt.savefig(f"{OUT}/volcano.png", dpi=150, bbox_inches="tight"); plt.close()

    degs = res[res["DEG"]].sort_values("log2FC").index
    if len(degs) >= 2:
        order = list(sub.index[sub["group"] == case]) + list(sub.index[sub["group"] == control])
        heat = logc.T.loc[degs, order]
        z = heat.sub(heat.mean(axis=1), axis=0).div(heat.std(axis=1), axis=0)
        plt.figure(figsize=(9, 3.5))
        plt.imshow(z, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        plt.colorbar(label="z-score")
        plt.yticks(range(len(z.index)), z.index)
        n1 = (sub["group"] == case).sum()
        plt.xticks([n1 / 2 - .5, n1 + (len(order) - n1) / 2 - .5], [case, control])
        plt.axvline(n1 - .5, color="black")
        plt.title("Heatmap of DEGs")
        plt.savefig(f"{OUT}/heatmap.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved: {OUT}/deg_results.csv, {OUT}/volcano.png" + (f", {OUT}/heatmap.png" if len(degs) >= 2 else ""))


def cmd_apoptosis(res, logc, sub, control, case, apoptosis):
    os.makedirs(OUT, exist_ok=True)
    present = [g for g in apoptosis if g in res.index]
    if not present:
        print("No apoptosis genes found in the data.")
        return
    t = res.loc[present, ["log2FC", "padj", "direction"]].copy()
    t["role"] = ["pro-apoptotic" if g in PRO else "anti-apoptotic" if g in ANTI else "unknown" for g in t.index]
    print("\nApoptosis genes:")
    print(t.round(4).to_string())
    anti_up = [g for g in t.index if t.loc[g, "role"] == "anti-apoptotic" and t.loc[g, "direction"].startswith("up")]
    pro_down = [g for g in t.index if t.loc[g, "role"] == "pro-apoptotic" and t.loc[g, "direction"].startswith("down")]
    pro_up = [g for g in t.index if t.loc[g, "role"] == "pro-apoptotic" and t.loc[g, "direction"].startswith("up")]
    if anti_up or pro_down:
        print(f"\nResult: apoptosis looks SUPPRESSED in {case} (anti-apoptotic up: {', '.join(anti_up) or 'none'}; pro-apoptotic down: {', '.join(pro_down) or 'none'}).")
    elif pro_up:
        print(f"\nResult: apoptosis genes look INCREASED in {case} ({', '.join(pro_up)}).")
    else:
        print("\nResult: no clear apoptosis change with the current cut-offs.")
    mc = logc.loc[sub["group"] == case, present].mean()
    mk = logc.loc[sub["group"] == control, present].mean()
    x = np.arange(len(present))
    plt.figure(figsize=(6, 4))
    plt.bar(x - .18, mc, .36, label=case, color="firebrick")
    plt.bar(x + .18, mk, .36, label=control, color="steelblue")
    plt.xticks(x, present); plt.ylabel("mean log2(count + 1)"); plt.legend(); plt.title("Apoptosis genes")
    plt.savefig(f"{OUT}/apoptosis.png", dpi=150, bbox_inches="tight"); plt.close()
    t.round(6).to_csv(f"{OUT}/apoptosis_genes.csv")
    print(f"Saved: {OUT}/apoptosis.png, {OUT}/apoptosis_genes.csv")


def train_logistic(X, y, steps=3000, lr=0.1):
    """Small logistic regression (numpy only)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + 0.01 * w)
        b -= lr * (p - y).mean()
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w + b)))


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def select(Xs, ys, genes):
    a, b = Xs[ys == 1], Xs[ys == 0]
    fc = a.mean() - b.mean()
    padj = bh_adjust(np.nan_to_num(stats.ttest_ind(a, b).pvalue, nan=1.0))
    sel = [g for g, f, p in zip(genes, fc, padj) if p < 0.05 and abs(f) > 1]
    return sel or list(pd.Series(padj, index=genes).nsmallest(3).index)


def cmd_predict(sub, genes, control, case):
    rng = np.random.default_rng(42)
    X = np.log2(sub[genes] + 1)
    y = (sub["group"] == case).astype(int).to_numpy()
    idx = np.arange(len(y))
    test = np.concatenate([rng.choice(idx[y == k], max(1, int(0.2 * (y == k).sum())), replace=False) for k in (0, 1)])
    train = np.setdiff1d(idx, test)
    sel = select(X.iloc[train], y[train], genes)         # genes chosen on training samples only
    model = train_logistic(X.iloc[train][sel].to_numpy(), y[train])
    prob = model(X.iloc[test][sel].to_numpy())
    pred = (prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y[test] == 1)).sum()); tn = int(((pred == 0) & (y[test] == 0)).sum())
    fp = int(((pred == 1) & (y[test] == 0)).sum()); fn = int(((pred == 0) & (y[test] == 1)).sum())
    print(f"\nGenes used by the model (chosen on training samples only): {', '.join(sel)}")
    print(f"Training samples: {len(train)}   Test samples: {len(test)}")
    print(f"Accuracy: {(tp + tn) / len(test):.2f}   ROC-AUC: {auc(y[test], prob):.2f}")
    print(f"Confusion matrix (test):  {control} correct={tn}, {control} called {case}={fp}, {case} correct={tp}, {case} called {control}={fn}")
    print("Note: very high accuracy is expected on simulated data. Real data is harder.")

    final_sel = select(X, y, genes)
    final = train_logistic(X[final_sel].to_numpy(), y)
    which = ask("\nPredict which sample? (type a sample name, e.g. " + sub.index[0] + ", or press Enter to skip)", "")
    if which:
        if which not in sub.index:
            print("Sample not found.")
            return
        p = float(final(X.loc[[which], final_sel].to_numpy())[0])
        print(f"{which}: predicted {case if p >= .5 else control}  (probability of {case} = {p:.3f})")


def cmd_mutations(sub, muts, control, case):
    if muts is None:
        print("\nNo mutation file was given.")
        return
    m = muts[muts["sample"].isin(sub.index)].copy()
    m["group"] = m["sample"].map(sub["group"])
    n = sub["group"].value_counts()
    freq = m.groupby(["gene", "group"])["sample"].nunique().unstack(fill_value=0)
    for g in (case, control):
        if g not in freq.columns:
            freq[g] = 0
    pct = pd.DataFrame({case: freq[case] / n[case] * 100, control: freq[control] / n[control] * 100}).round(1)
    print("\nPercent of samples with a mutation, by gene:")
    print(pct.to_string())
    print("\nMutation types:")
    print(m.groupby(["gene", "mutation_type"]).size().to_string() if "mutation_type" in m.columns else m.to_string())


def main():
    print("=" * 60)
    print(" Cancer Gene Expression Analysis (terminal version)")
    print(" Hira Afzal | Hamna Nasim | Namra Munir Khokhar")
    print("=" * 60)
    sub, genes, control, case, muts = load_data()
    res, logc = find_degs(sub, genes, control, case)
    apoptosis = [g for g in genes if g in PRO | ANTI]
    while True:
        print("\n----- MENU -----")
        print("1  Overview and read counts")
        print("2  DEGs (table, volcano plot, heatmap)")
        print("3  Apoptosis genes")
        print("4  Predict cancer or healthy (machine learning)")
        print("5  Mutations")
        print("6  Run everything")
        print("0  Exit")
        c = ask("Choose", "6")
        if c == "1":
            cmd_overview(sub, genes, control, case)
        elif c == "2":
            cmd_degs(res, case, logc, sub, control, apoptosis)
        elif c == "3":
            cmd_apoptosis(res, logc, sub, control, case, apoptosis)
        elif c == "4":
            cmd_predict(sub, genes, control, case)
        elif c == "5":
            cmd_mutations(sub, muts, control, case)
        elif c == "6":
            cmd_overview(sub, genes, control, case)
            cmd_degs(res, case, logc, sub, control, apoptosis)
            cmd_apoptosis(res, logc, sub, control, case, apoptosis)
            cmd_predict(sub, genes, control, case)
            cmd_mutations(sub, muts, control, case)
        elif c == "0":
            print("Bye. Results are in the 'results' folder.")
            break
        else:
            print("Please type a number from the menu.")


if __name__ == "__main__":
    main()
