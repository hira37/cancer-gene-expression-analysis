# Cancer Gene Expression Analysis

FYP: finding DEGs and apoptosis-related genes in cancer vs healthy samples

## Group Members
- Hira Afzal (SP23-BSI-020)
- Hamna Nasim (SP23-BSI-017)
- Namra Munir Khokhar (SP23-BSI-040)

COMSATS University Islamabad, Department of Biosciences, BS Bioinformatics

## Project Idea
We analyse gene expression (RNA-seq) data of cancer patients and healthy individuals to find differentially expressed genes (DEGs), look at apoptosis-related genes, and build a model that predicts cancer or healthy samples.

## Workflow
1. Galaxy: quality control, trimming, mapping, counting (practice on tutorial data)
2. Python (Google Colab): DEG analysis, apoptosis genes, plots, ML model
3. GitHub: code and results
4. Streamlit app: prediction demo

## Files
- `cancer_gene_expression_demo.ipynb`: Python (Colab) notebook. DEGs and volcano plot, heatmap, apoptosis genes, and a machine learning model (cancer vs healthy).
- `simulated_counts.csv`: simulated practice data (30 cancer and 30 healthy samples, 10 genes).
- `app.py`: Streamlit GUI with tabs: overview and read counts, DEGs (volcano, table, download), heatmap, apoptosis genes, cancer vs healthy prediction, mutations, help. Supports uploading your own counts CSV (and optional mutation CSV).
- `analysis_cli.py`: terminal version. Run `python analysis_cli.py`, choose from a menu, and get DEGs, apoptosis result, prediction and mutations. Saves tables and plots in a `results` folder.
- `simulated_mutations.csv`: simulated example mutation list for the app.
- `requirements.txt`: Python packages needed for the app.

## Data note
All results so far use **simulated practice data**, not real patients. The code is ready to be used with a real count table (GEO or TCGA).

## Run the terminal version
```
python analysis_cli.py
```
(needs pandas, numpy, scipy, matplotlib)

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```

## Status
Work in progress.
- Galaxy practice done up to trimming (Cutadapt).
- Python: DEGs, volcano plot, heatmap, apoptosis genes and ML model done on practice data.
- App: GUI version ready (file upload supported).
- Next: real dataset (GEO/TCGA), cross-validation, more models, and finishing the Galaxy steps (mapping, counting, DESeq2).
