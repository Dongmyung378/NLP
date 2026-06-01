# Multi-Label Text Classification System

---

## 📋 Overview
An end-to-end NLP pipeline engineered to concurrently classify unstructured text data into 8 distinct categorical labels. The project architecture evolved progressively from traditional sequence models to state-of-the-art (SOTA) **DeBERTa-v2**, focusing heavily on algorithmic and infrastructure solutions to overcome severe **Class Imbalance** and **Hardware Compute Constraints**.

---

## 💻 Source Code Links (Fast Raw Text Load)
> ⚠️ **Notice:** Due to the massive training logs and ensemble prediction outputs embedded within the `.ipynb` files, standard web renderers may encounter a 503 Timeout. Please use the high-speed text links below to instantly inspect the raw source code:
- [Task 1 (BiLSTM + Custom Attention) — Raw Code](https://raw.githubusercontent.com/Dongmyung378/NLP/main/Corsework%202/Submit/10879360_task1_code.ipynb)
- [Task 2 (RoBERTa Fine-Tuning with AMP) — Raw Code](https://raw.githubusercontent.com/Dongmyung378/NLP/main/Corsework%202/Submit/10879360_task2_code.ipynb)
- [Task 3 (DeBERTa-v2 Mean Pooling & Seed Ensemble) — Raw Code](https://raw.githubusercontent.com/Dongmyung378/NLP/main/Corsework%202/Submit/10879360_task3_code.ipynb)

---

## 🛠 Tech Stack
- **Frameworks & Libraries:** PyTorch, Hugging Face Transformers
- **Architectures:** BiLSTM+Attention, RoBERTa-Base, DeBERTa-v2 (`DebertaV2Model`)
- **Optimization:** Automatic Mixed Precision (AMP), Asymmetric Loss (ASL)
- **Methodologies:** Seed Ensemble (Seeds: 16, 42, 378), Mean Pooling Token Aggregation

---

## 💡 Model Architecture Evolution

### 1. Baseline: BiLSTM + Custom Attention (`task1`)
- **Metric:** Validation F1-Score: **0.5845**
- **Architecture:** Combined Bidirectional LSTM layers with GloVe embeddings and a **Custom Attention Mechanism** to dynamically weigh contextual keyword importance across token sequences.

### 2. Transfer Learning: RoBERTa Fine-Tuning (`task2`)
- **Metric:** Validation F1-Score: **0.6253 (Best)**
- **Architecture:** Leveraged the deep contextualized representations of a pre-trained RoBERTa-Base model, optimizing it for multi-label text semantics to achieve the highest overall performance in this benchmark.

### 3. SOTA Optimization: DeBERTa-v2 Ensemble (`task3`)
- **Metric:** Validation F1-Score: **0.6142**
- **Key Engineering Implementations:**
  - **Mean Pooling Integration:** Developed a `DebertaMeanPoolingClassifier` that aggregates all token hidden-state vectors rather than relying solely on the standard `[CLS]` token, significantly minimizing contextual information loss.
  - **Tri-Seed Ensemble Strategy:** To maximize generalization and stabilize boundary predictions, implemented a soft-voting ensemble combining probability outputs from three distinct models trained under isolated random seeds (16, 42, and 378).

---

## 🔥 Core Engineering Competencies

### 1. Mitigating Class Imbalance via Asymmetric Loss (ASL)
To combat the inherent data sparsity where negative samples heavily dominate the multi-label dataset, integrated **Asymmetric Loss**. The loss function down-weights the gradients of easy-to-classify negative samples while prioritizing loss propagation for hard positive instances, preventing major model bias.

### 2. Compute Optimization utilizing PyTorch AMP
To bypass hardware memory constraints while fine-tuning heavy Transformer architectures, implemented **Automatic Mixed Precision (AMP)**. Utilizing `torch.amp.autocast` and `GradScaler`, the pipeline effectively halved GPU VRAM footprints and accelerated training throughput with zero degradation in numerical gradient precision.

---

## 📊 Performance Analysis & Engineering Insights

- **Architectural Trade-offs:** The empirical results demonstrated that the RoBERTa-Base model (Model 2) outpaced the newer, more complex DeBERTa-v2 (Model 3) on this specific dataset.
- **Overfitting & Capacity Analysis:** This divergence highlights that scale does not always equate to performance; increasing model parameters on a bounded dataset introduces a substantial risk of **overfitting**. This benchmark reinforces the critical engineering necessity of identifying the optimal **Model Capacity-to-Data Scale trade-off**, prioritizing localized data alignment over blind SOTA model adoption.
