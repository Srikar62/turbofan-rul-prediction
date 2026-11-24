# 🔧 Turbofan Engine RUL Prediction

A deep learning pipeline for **Remaining Useful Life (RUL) prediction** of turbofan engines using the NASA C-MAPSS benchmark dataset. The system features a modern hybrid architecture: **Selective Kernel Conv1D (SK-Conv1D) + Bidirectional LSTM + 3D Attention + Cross-Attention Guided Fusion + Mixture of Experts (MoE)** with **Monte Carlo (MC) Dropout** for calibrated epistemic uncertainty quantification.

---

## 📌 Project Description

Predicting when an aircraft engine will fail is a critical problem in **predictive maintenance**. This project tackles it by estimating the **Remaining Useful Life (RUL)** — the number of operational cycles remaining before failure — for each engine in the C-MAPSS dataset.

### Problem Solved

Traditional threshold-based maintenance strategies lead to either **premature replacements** (costly) or **unexpected failures** (dangerous). This system provides:
- Accurate cycle-level RUL estimates
- Calibrated **uncertainty bounds** (5th–95th percentile confidence intervals)
- Per-dataset tuned models covering single and multi-condition operating regimes

---

## ✨ Features

- **Dual-Stream Temporal Inputs** — Trajectory-level first-order physical differences ($\Delta x_t = x_t - x_{t-1}$) concatenated with raw sensors (`[raw_sensors | delta_sensors]`) for instantaneous degradation velocity sensing.
- **Selective Kernel Conv1D (SK-Conv1D)** — Adaptive multi-scale 1D convolutions ($k=3, 5, 7$) with channel-wise squeeze-and-selection softmax gating dynamically selecting receptive fields between early-life smoothing and late-stage failure spikes.
- **Bidirectional LSTM** — 2-layer BiLSTM ($hidden=128$, output dimension 256) for bidirectional sequential degradation trajectory modeling.
- **3D Attention + AttentionPool** — Parametric 3D energy-based attention over $(B, T, C)$ plus attention-weighted temporal pooling to prioritize degradation-critical cycles.
- **Cross-Attention Guided Fusion** — 4-head multi-head cross-attention where a combined query state (`[pooled_history | last_token | handcrafted_priors]`) cross-attends over all temporal sequence tokens, seamlessly fused with direct handcrafted projections.
- **Handcrafted Statistical Priors** — Windowed mean, standard deviation, linear trend slope, late-phase slope (last 30%), % deviation, and positive CUSUM drift.
- **True Mixture of Experts (MoE)** — 4 specialist expert MLPs with temperature-scaled softmax gating and dual anti-collapse regularisation (variance penalty + entropy maximization).
- **EMA & Top-3 Checkpoint Ensembling** — Exponential Moving Average (EMA) weight stabilization with composite RMSE / NASA Score Top-3 checkpoint ensembling.
- **MC Dropout Probabilistic Inference** — $T=50$ stochastic forward passes for calibrated epistemic uncertainty bounds (5th–95th percentile confidence intervals).
- **Condition-aware Normalisation** — KMeans clustering + per-regime MinMax scaling for multi-condition regimes (FD002, FD004) with piecewise-linear RUL targets.
- **Early Stopping & Warmup LR Scheduling** — Linear warmup (5 epochs) → Cosine annealing with Huber asymmetric late-penalty loss.
- **Full EDA Suite & Predictions Export** — Missing value audits, duplicate checks, degradation plots, and exported predictions per dataset.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.7 (CUDA 11.8) |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualisation** | Matplotlib, Seaborn |
| **Dataset** | NASA C-MAPSS (FD001–FD004) |
| **Hardware** | CUDA GPU (CPU fallback supported) |

---

## 📁 Project Structure

```
turbofan-rul-prediction/
│
├── main.py                   # Entry point — orchestrates all pipeline steps
├── config.py                 # Hyperparameters, constants, device setup
│
├── data_loading.py           # Loads FD001–FD004, selects informative sensors
├── data_processing.py        # RUL computation, condition-based normalisation
├── dataset.py                # PyTorch Dataset (sliding window + HC features)
│
├── model.py                  # Neural network: CNN-BiLSTM-3DAttn + MoE
├── train.py                  # Training loop, scheduler, early stopping
├── evaluate.py               # MC Dropout evaluation, expert utilisation check
├── metrics.py                # RMSE evaluation + NASA asymmetric score function
│
├── visualize.py              # Plots: training curves, RUL predictions + CI bands
├── summary.py                # Final metrics summary table
├── save_models.py            # Saves trained model weights (.pt files)
├── save_rul.py               # Exports RUL predictions to text files
│
├── about_data.py             # Dataset overview and statistics
├── Missingvalueanalysis.py   # Missing value audit
├── duplicatecheck.py         # Duplicate row detection
├── EDA.py                    # Full exploratory data analysis with plots
│
├── requirements.txt          # Python dependencies
├── test_gpu.py               # GPU availability diagnostics
├── .gitignore                # Git ignore rules
│
├── backend/                  # FastAPI real-time monitoring & inference API
│   ├── server.py             # FastAPI app and static file server
│   ├── inference_engine.py   # Telemetry ingestion & live RUL inference
│   ├── shap_explainer.py     # SHAP feature attribution explainer
│   └── scaler_manager.py     # Scaler management for multi-condition regimes
├── frontend/                 # Web monitoring dashboard UI
│   ├── index.html            # Main industrial monitoring interface
│   ├── css/style.css         # Styling and dashboard layout
│   └── js/dashboard.js       # Live telemetry and Chart.js visualizations
│
└── data/                     # C-MAPSS raw data files
    ├── train_FD001–FD004.txt             # Training data per dataset
    ├── test_FD001–FD004.txt              # Test data per dataset
    ├── RUL_FD001–FD004.txt               # Ground truth RUL values
    ├── readme.txt                        # Dataset documentation
    └── Damage Propagation Modeling.pdf   # NASA reference paper
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Srikar62/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Verify GPU Availability

Before installing dependencies, check whether your machine has a CUDA-capable GPU:

```bash
python test_gpu.py
```

Read the output:
- ✅ **CUDA is available** — proceed with the GPU installation below
- ❌ **No CUDA / CPU only** — proceed with the CPU installation below

### 4. Install Dependencies

**Option A — GPU (CUDA 11.8, recommended for faster training):**
```bash
pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn
```

**Option B — CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install everything at once using the requirements file (GPU build):
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python main.py
```

This executes the pipeline steps automatically:

1. Loads datasets and selects informative sensors
2. Runs EDA (missing values, duplicates, degradation plots)
3. Preprocesses data (RUL labels, condition normalisation)
4. Creates PyTorch DataLoaders with sliding windows
5. Trains models for all 4 FD datasets
6. Checks MoE expert utilisation balance
7. Runs MC Dropout evaluation (T=50 samples)
8. Plots results with uncertainty bands
9. Prints final summary, saves model weights and RUL predictions

### 🖥️ Training on CPU

The pipeline runs on CPU automatically if no GPU is detected. However, training is significantly slower.


### Output Files

| Path | Description |
|---|---|
| `model_FD001.pt` – `model_FD004.pt` | Trained model weights (saved in project folder) |
| `predictions/remaininguselife_fd00X.txt` | Predicted RUL values |
| `eda_fd00X.png` | EDA distribution plots |
| `eda_degradation_fd00X.png` | Sensor degradation curves |
| `rul_results.png` | RUL prediction vs ground truth with CI |
---

## 📊 Real-Time Monitoring Dashboard

The project includes a real-time web monitoring dashboard powered by FastAPI and Chart.js for live telemetry streaming, fleet health assessment, and on-demand RUL inference.

### Launch the Dashboard

```bash
python -m backend.server
```

Then open **http://127.0.0.1:8000** in your browser.

### Features

- **Fleet Overview** — Real-time fleet health distribution, urgent maintenance priorities, and risk matrix.
- **Live Engine Monitor** — Live sensor telemetry streams, degradation trajectories, and dynamic RUL predictions with confidence intervals.
- **Upload Engine Data** — Upload custom engine telemetry files (CSV/text) for instant batch inference.
- **Model Benchmarks** — View live benchmark performance against C-MAPSS test datasets.

---

## 🏗️ Architecture & Workflow

### Model Architecture — `TurbofanRULModel`

The architecture integrates **Selective Kernel Convolution (SK-Conv1D)**, **Bidirectional LSTM**, **3D Energy Attention**, **Cross-Attention Guided Fusion**, and **Mixture of Experts (MoE)** with entropy-regularized gating:

```
                    ┌──────────────────────────────────────────────┐
                    │ Raw Sensors + Trajectory Deltas (Δx_t)       │
                    │ Shape: [Batch, Window_Size, 2 × N_Sensors]   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Selective Kernel Conv1D (SK-Conv1D)                                                 │
│   ├── Branch 1: Conv1D (k=3, pad=1) ──┐                                             │
│   ├── Branch 2: Conv1D (k=5, pad=2) ──┼─► Element Sum ──► GAP ──► MLP ──► Softmax   │
│   └── Branch 3: Conv1D (k=7, pad=3) ──┘                           (Dynamic Gating a)│
│   Output: V = a₃·U₃ + a₅·U₅ + a₇·U₇  + GELU + Dropout + LayerNorm                    │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 2-Layer Bidirectional LSTM (BiLSTM)                                                 │
│ Hidden Size = 128 (Forward 128 + Backward 128 = 256 dimensions)                     │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ 3D Energy-Based Attention                                                           │
│ Computes time-frequency-channel energy: e = (x - μ)² / (4(σ² + λ)) + 0.5            │
│ Modulates temporal representations via sigmoid gating: x · σ(e)                     │
└──────────────────────┬───────────────────────────────────────┬──────────────────────┘
                       │                                       │
                       ▼ (t=T)                                 ▼ (All t=1..T)
       ┌───────────────────────────────┐       ┌───────────────────────────────┐
       │ Terminal State (last_token)   │       │ AttentionPool (pooled_learned)│
       │ Shape: [Batch, 256]           │       │ Shape: [Batch, 256]           │
       └───────────────┬───────────────┘       └───────────────┬───────────────┘
                       │                                       │
                       └───────────────────┬───────────────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                       │ + Handcrafted Statistical Priors      │
                       │ x_hc: [Mean, Std, Slope, LateSlope,   │
                       │        %Dev, CUSUM]                   │
                       └───────────────────┬───────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Cross-Attention Guided Fusion                                                       │
│   ├── Query (Q): MLP([pooled_learned || last_token || x_hc]) -> [Batch, 1, 128]     │
│   ├── Key / Value (K, V): Linear(BiLSTM Tokens) -> [Batch, T, 128]                  │
│   ├── Multi-Head Cross-Attention (4 Heads) + Residual Connection + LayerNorm        │
│   └── Direct Handcrafted Projection: LayerNorm(Dropout(GELU(Linear(x_hc))))         │
│   Output Latent: Concat([Attended_Seq_128, Direct_HC_128]) -> [Batch, 256]          │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Mixture of Experts (MoE) Output Head                                                │
│   ├── Router: Gating MLP with Learnable Temperature Softmax                         │
│   ├── Aux Loss: Variance Penalty (var_penalty) + Entropy Bonus (entropy_bonus)      │
│   └── 4 Specialist Expert MLPs (hidden=128, bias initialized to 0.5)                │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Predicted RUL (Engine Cycles) = Clamp(MoE_out, 0, 1) × RUL_Max                      │
│ Epistemic Uncertainty Quantification via MC Dropout (T=50 Forward Passes)           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Component Highlights

1. **Trajectory-Accurate Physical Deltas ($\Delta x_t$)**:
   - First-order backward differences ($\Delta x_t = x_t - x_{t-1}$) are calculated across the entire engine trajectory **prior to window slicing**, guaranteeing physical rate-of-change continuity across sliding-window boundaries.
   - The sequence input doubles in channel depth ($2 \times N_{\text{selected}}$), providing simultaneous awareness of absolute sensor values and instantaneous degradation velocity.

2. **Selective Kernel 1D Convolution (`SK-Conv1D`)**:
   - Instead of static concatenation, parallel Conv1D branches ($k=3, 5, 7$) are dynamically combined via softmax attention:
     $$V = a_3 \cdot U_3 + a_5 \cdot U_5 + a_7 \cdot U_7$$
     where $[a_3, a_5, a_7] = \text{Softmax}(\text{Linear}_2(\text{GELU}(\text{Linear}_1(\text{GAP}(U_3 + U_5 + U_7)))))$.
   - Dynamically selects wide receptive fields ($k=7$) during smooth, early-life operations and compact receptive fields ($k=3$) when rapid sensor fluctuations emerge near component failure.

3. **BiLSTM + 3D Attention & AttentionPool**:
   - Two bidirectional LSTM layers model long-term temporal dependencies in forward and backward time directions.
   - **3D Energy Attention** calculates parametric energy over $(B, T, C)$ to emphasize critical degradation phases.
   - **AttentionPool** replaces simple mean-pooling by learning a softmax-weighted sum over the sequence timeline.

4. **Multi-Modal Cross-Attention Guided Fusion**:
   - Combines the global degradation history (`pooled_learned`), current terminal wear state (`last_token`), and physical priors (`x_hc`) into a unified **Query**.
   - This Query cross-attends over all temporal token Keys/Values, allowing static statistical priors and current state to selectively extract historical degradation evidence.
   - Concatenated with a projected direct handcrafted feature path to preserve non-linear statistical signals without gradient attenuation.

5. **Solving the Expert Utilisation Problem (MoE Collapse)**:
   - **Variance Penalty**: Penalises non-uniform average routing across the batch: $\text{Var}(\bar{g}) \to 0$.
   - **Entropy Bonus**: Encourages maximum entropy of average gate weights: $\mathcal{H}(\bar{g}) \to \log(4)$.
   - **Smart Bias Initialisation**: Output layer bias of each expert is initialized to `0.5`, placing predictions directly in the active gradient region from iteration 1.

6. **Training Mechanics & Robust Ensembling**:
   - **Asymmetric Huber Loss**: Warmup-scaled asymmetric penalization heavily penalizing late predictions (dangerous under-prediction of failure).
   - **Exponential Moving Average (EMA)**: Evaluates an EMA-smoothed model weight shadow copy to eliminate stochastic variance.
   - **Top-3 Checkpoint Ensemble**: Saves and dynamically ensembles the top-3 best validation checkpoints using a combined $\text{RMSE} + 0.004 \times \text{Score}$ criterion.

### Training Hyperparameters

| Hyperparameter | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **Window Size** | 30 | 75 | 30 | 90 |
| **Max RUL (Capped)** | 125 | 125 | 125 | 130 |
| **Max Epochs** | 250 | 350 | 250 | 400 |
| **Learning Rate** | 4e-4 | 2e-4 | 4e-4 | 2e-4 |
| **Batch Size** | 128 | 256 | 128 | 256 |
| **Dropout** | 0.25 | 0.30 | 0.25 | 0.30 |
| **Weight Decay** | 5e-3 | 5e-3 | 5e-3 | 5e-3 |
| **Operating Regimes** | 1 (Single) | 6 (KMeans clustered) | 1 (Single) | 6 (KMeans clustered) |
| **Fault Modes** | 1 (HPC) | 1 (HPC) | 2 (HPC + Fan) | 2 (HPC + Fan) |

### Evaluation Metrics

- **RMSE** — Root Mean Squared Error (in engine cycles)
- **MAE** — Mean Absolute Error (in engine cycles)
- **R²** — Coefficient of Determination ($1.0$ is perfect correlation)
- **NASA Score** — Asymmetric scoring function that penalises late predictions (under-predictions) more than early ones. Here, **`d = Predicted RUL - True RUL`** (the error difference in cycles):

```
Score = Σ (exp(d/10) - 1)   if d ≥ 0  (late prediction)
        Σ (exp(-d/13) - 1)  if d < 0  (early prediction)
```

### 🏆 Final Experimental Results

```
 FINAL RESULTS — CNN-BiLSTM-3DAttn + MoE(4) + MC Dropout
================================================================================
  FD              RMSE         MAE          R²         Score     Uncertainty
--------------------------------------------------------------------------------
  FD001        13.2715      9.7800      0.8903        264.50           1 cyc
  FD002        13.1124      9.2136      0.9067        747.70           2 cyc
  FD003        14.6398     10.0879      0.8603        313.59           2 cyc
  FD004        14.7756     10.0073      0.8893        911.08           2 cyc
--------------------------------------------------------------------------------
  Average      13.9498      9.7722      0.8867        559.22
================================================================================
```

| Dataset | RMSE | MAE | R² | NASA Score | Uncertainty |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FD001** | **13.2715** | 9.7800 | 0.8903 | 264.50 | ± 1 cyc |
| **FD002** | **13.1124** | 9.2136 | 0.9067 | 747.70 | ± 2 cyc |
| **FD003** | **14.6398** | 10.0879 | 0.8603 | 313.59 | ± 2 cyc |
| **FD004** | **14.7756** | 10.0073 | 0.8893 | 911.08 | ± 2 cyc |
| **Average** | **13.9498** | **9.7722** | **0.8867** | **559.22** | - |