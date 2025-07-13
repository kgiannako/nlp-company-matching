# 🧠 Company Entity Matching with SBERT & FAISS

This project explores semantic entity matching between company records using state-of-the-art NLP models. Given variations in names, addresses, and descriptions, the goal is to determine whether two company entries refer to the same real-world organisation.

It combines Sentence-BERT embeddings with FAISS approximate nearest neighbour search to support fast, scalable matching.

---

## 🔍 Problem Statement

Real-world datasets contain noisy and inconsistent company information:

"ABC Ltd, 12 High Street, London"
vs
"A.B.C. Limited, 12 High St., London"

Traditional string-matching fails. We apply **semantic similarity** and ANN search to solve the matching problem more robustly.

---

## 🛠️ Techniques Used

- `Sentence-BERT` embeddings:
  - [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- `FAISS` for fast approximate nearest neighbour indexing (HNSW)
- Cosine similarity, AUC, F1, precision/recall metrics
- Jupyter, `scikit-learn`, `seaborn`, `matplotlib`

---

## 📈 Model Comparison

| Model             | Accuracy | Precision | Recall | F1 Score | AUC   |
|------------------|----------|-----------|--------|----------|--------|
| MiniLM-L6-v2      | 0.75     | 0.27      | 0.69   | 0.38     | 0.81   |
| MPNet-Base-v2     | 0.77     | 0.28      | 0.69   | 0.40     | 0.82   |

AUC shows solid ranking ability. Lower precision suggests more data or fine-tuning is needed for reliable thresholding.

---

## 📁 Repository Structure

nlp-company-matching/
├── data/ # Raw and processed input data
├── matching_pipeline.ipynb # Main analysis notebook
├── requirements.txt # Python dependencies
├── .gitignore # Excludes large or temp files
└── README.md # This file

---

## 🔄 Next Steps

- Fine-tune the transformer model on domain-specific entity pairs
- Incorporate more labelled data
- Explore structured field embeddings
- Build an interactive demo (e.g., Streamlit)

---

## 📚 Dataset

Uses the [WDC Entity Matching Dataset (50-pair sample)](https://figshare.com/articles/dataset/WDC_Entity_Matching_Gold_Standard/1304970).

---

## 🌍 Author

📌 Built by [@kgiannako](https://github.com/kgiannako)  
💼 Part of a personal NLP portfolio project.

