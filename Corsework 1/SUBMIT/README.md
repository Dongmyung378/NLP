# Word Similarity Engine (Lexical & Semantic Proximity Measurement System)

---

## 📋 Overview
An NLP pipeline designed to compute the semantic and lexical similarity between word pairs utilizing the large-scale WikiText-103 corpus. This project involved developing and benchmarking two distinct modeling paradigms—**Sparse Representation (TF-IDF Hybrid) and Dense Representation (FastText)**—to capture both contextual semantics and subword morphological features.

---

## 🛠 Tech Stack
- **Language:** Python
- **NLP (Sparse):** Scikit-learn (TF-IDF, Character N-gram), SciPy
- **NLP (Dense):** Gensim (FastText, Phraser), NLTK
- **Performance Optimization:** In-Memory Caching, Multiprocessing

---

## 💡 Key Engineering Features

### Model 1: Hybrid Vector (TF-IDF + Character N-gram) — `CW1_task1`
- **Source Code:** 👉 [Click here to view the Interactive Notebook via nbviewer](https://nbviewer.org/github/Dongmyung378/NLP/blob/main/Corsework%201/SUBMIT/10879360_CW1_task1.ipynb)
- **Robust OOV (Out-of-Vocabulary) Handling:** Implemented a **Character N-gram Vectorizer** to mitigate fallback errors caused by unseen words, enabling spelling-based morphological inference for OOV terms.
- **Memory & Latency Optimization:** Integrated an in-memory caching mechanism via a `_known_vector_cache` dictionary, eliminating redundant vector computations and accelerating inference speed across large-scale test datasets.

### Model 2: FastText & Bigram Phraser — `CW1_task2`
- **Source Code:** 👉 [Click here to view the Interactive Notebook via nbviewer](https://nbviewer.org/github/Dongmyung378/NLP/blob/main/Corsework%201/SUBMIT/10879360_CW1_task2.ipynb)
- **Subword Information Extraction:** Leveraged Gensim's FastText algorithm to learn internal n-gram representations of words, effectively resolving data sparsity and improving OOV inference accuracy over traditional word-level embeddings.
- **Collocation Modeling (Bigram Phraser):** Built a text-normalization pipeline utilizing `gensim.models.phrases` to detect and bind multi-word expressions (e.g., 'New York') into unified tokens, significantly enhancing contextual representations.
- **Compute Optimization (Multiprocessing):** Maximized hardware utilization by configuring CPU core-based parallel processing, substantially reducing model training time on the high-volume WikiText-103 corpus.

---

## 📊 Results & Insights
- **Performance Evaluation:** The FastText-based Dense Model (Model 2) achieved a predictive accuracy of **69.9%**, outperforming the Hybrid Sparse Model (Model 1) which recorded **68.0%**.
- **Engineering Trade-offs:** While Model 2 demonstrated superior capabilities in semantic context understanding and robust OOV handling, it introduced higher computational complexity and longer training/validation cycles compared to Model 1. This benchmark highlights the engineering necessity of balancing **inference precision against infrastructure compute costs** when selecting a production-level architecture.
