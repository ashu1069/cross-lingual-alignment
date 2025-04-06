# Cross-Lingual Word Embedding Alignment: Supervised vs Unsupervised (Hindi-English)

This project implements and compares two methods for cross-lingual alignment of word embeddings between English and Hindi:

- **Supervised Procrustes Alignment**
- **Unsupervised Adversarial + CSLS + Procrustes Refinement (inspired by MUSE)**

---

## Features

- Uses pre-trained FastText embeddings
- Supervised mapping via Procrustes algorithm using bilingual dictionaries
- Unsupervised adversarial training with CSLS and refinement
- Evaluation with Precision@1 and @5
- Cosine similarity analysis of aligned word pairs
- Ablation study for the impact of lexicon size
- t-SNE visualizations of the aligned vector embeddings

---

## Files
Click here: [Files](https://drive.google.com/drive/folders/14qj5tnjOcuYyEN9FRo5tYjcJNnrAdm7n?usp=sharing)

- `filtered_eng_embeddings.npz` — Top 100k English word vectors
- `filtered_hindi_embeddings.npz` — Top 100k Hindi word vectors
- `en-hi.txt` — Bilingual dictionary for training/evaluation

---

## How It Works

1. **Load fastText embeddings**
2. **Supervised mode**: Train an orthogonal mapping `W` using SVD (Procrustes)
3. **Unsupervised mode**:
   - Use adversarial training to align source (Hindi) to target (English)
   - Apply CSLS to reduce hubness
   - Refine mapping via Procrustes
4. Evaluate using a test dictionary
5. Visualize with t-SNE and compare results

---

## Evaluation

- Precision@1 and @5 using a bilingual test dictionary
- Cosine similarity histograms to assess semantic similarity
- Dictionary-size ablation (5k, 10k, 20k pairs)

---

## Visualization

- Static t-SNE: Compare Hindi and English embeddings post-alignment
- Animated t-SNE: View the progression of adversarial alignment over epochs

---

## Getting Started

Run the notebook step-by-step in a Python environment or Colab. Ensure you have:

```bash
pip install torch faiss-cpu matplotlib scikit-learn
