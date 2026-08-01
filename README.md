# 🔧 Turbofan Engine RUL Prediction & AeroAgent-Twin

A deep learning pipeline for **Remaining Useful Life (RUL) prediction** of turbofan engines using the NASA C-MAPSS benchmark dataset. The system uses a hybrid **CNN-BiLSTM with 3D Attention + Mixture of Experts (MoE)** architecture with **Monte Carlo (MC) Dropout** for uncertainty quantification.

The **AeroAgent-Twin** extension converts numerical RUL predictions into actionable maintenance diagnostics using a simplified physics digital twin, a local RAG system over turbofan MRO SOPs, and optional Groq/OpenAI-compatible LLM report synthesis.

---

## 📌 Project Description

Predicting when an aircraft engine will fail is a critical problem in **predictive maintenance**. This project tackles it by estimating the **Remaining Useful Life (RUL)** — the number of operational cycles remaining before failure — for each engine in the C-MAPSS dataset.

### Problem Solved

Traditional threshold-based maintenance strategies lead to either **premature replacements** (costly) or **unexpected failures** (dangerous). This system provides:
- Accurate cycle-level RUL estimates
- Calibrated **uncertainty bounds** (2.5th–97.5th percentile confidence intervals)
- Per-dataset tuned models covering single and multi-condition operating regimes

---

## ✨ Features

- **Multi-scale CNN** — parallel convolutions with kernel sizes 3, 5, 7 for capturing short- and long-term degradation patterns
- **Bidirectional LSTM** — 2-layer BiLSTM (hidden=128) for sequential temporal modelling
- **3D Attention + AttentionPool** — highlights the most degradation-informative time steps
- **True Mixture of Experts (MoE)** — 4 specialist expert MLPs with entropy regularisation to prevent expert collapse
- **Handcrafted Features** — mean, slope, std, % deviation, CUSUM, late-slope per sensor window
- **MC Dropout Inference** — T=50 stochastic forward passes for uncertainty estimation
- **Condition-aware Normalisation** — KMeans-based clustering + per-condition MinMaxScaling for multi-regime datasets (FD002, FD004)
- **Piecewise-linear RUL** — capped RUL target for stable training
- **Early Stopping + LR Scheduling** — Linear warmup (5 epochs) → Cosine annealing
- **Full EDA Suite** — missing value analysis, duplicate checks, sensor degradation plots
- **Prediction Export** — RUL predictions saved to `.txt` files per dataset

### AeroAgent-Twin Extension Features

- **Probabilistic Inference API** — async MC Dropout service with sensor anomaly extraction and health classification
- **Physics Digital Twin** — simplified thermodynamic consistency checker mapping sensor deviations to HPC/HPT/combustor degradation
- **Hybrid RAG Retrieval** — TF-IDF + BM25 keyword search (optional sentence-transformer dense embeddings) over MRO SOP corpus
- **Agentic Diagnostic Graph** — deterministic pipeline with optional LangGraph stateful workflow
- **LLM Report Synthesis** — optional Groq/OpenAI grounded diagnostic reports with hallucination guard
- **MRO Work Order Generation** — structured maintenance tickets with urgency classification, action items, parts, and SOP references

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.7 (CUDA 11.8) |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualisation** | Matplotlib, Seaborn |
| **API / Schemas** | FastAPI, Pydantic |
| **RAG** | TF-IDF / sentence-transformers (optional) |
| **LLM Integration** | Groq / OpenAI (optional) |
| **Dataset** | NASA C-MAPSS (FD001–FD004) |
| **Hardware** | CUDA GPU (CPU fallback supported) |

---

## 📁 Project Structure

```
turbofan-rul-prediction/
│
├── main.py                   # Entry point — orchestrates all pipeline steps
├── app_aeroagent.py          # AeroAgent-Twin CLI + FastAPI entrypoint
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
├── api/                      # Probabilistic inference service
│   ├── inference.py          # MC Dropout RUL inference + anomaly extraction
│   └── schemas.py            # Pydantic schemas (TelemetryInput, RULResponse)
│
├── agents/                   # Agentic diagnostic graph
│   ├── graph.py              # Deterministic + LangGraph workflow orchestration
│   ├── llm_report.py         # Optional Groq/OpenAI report synthesis
│   ├── state.py              # AgentState TypedDict
│   └── tools.py              # Tool wrappers (inference, physics, RAG, MRO)
│
├── physics/                  # Thermodynamic digital twin
│   ├── digital_twin.py       # Sensor → HPC/HPT/combustor degradation mapping
│   └── thermo_models.py      # Gas turbine thermodynamic equations
│
├── rag/                      # RAG retrieval system
│   ├── ingest.py             # SOP document loading and chunking
│   └── vector_store.py       # Hybrid vector + keyword retrieval
│
├── mro/                      # MRO work order service
│   └── work_order.py         # Structured work order generation
│
├── data/                     # C-MAPSS raw data files
│   └── mro_manuals/          # Synthetic MRO SOP corpus
│       └── turbofan_mro_sop.json
│
├── requirements.txt          # Python dependencies
├── test_gpu.py               # GPU availability diagnostics
├── predictions/              # Output: remaininguselife_fd00X.txt files
└── turbofan_models/          # Saved model weights (created after training)
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

### 5. Install AeroAgent-Twin Extras (Optional)

For LLM report synthesis via Groq:
```bash
pip install openai python-dotenv
```

For FastAPI serving:
```bash
pip install fastapi uvicorn pydantic
```

For dense embeddings in RAG (optional, default uses TF-IDF):
```bash
pip install sentence-transformers
```

### 6. LLM API Setup (Optional — for Groq Report Synthesis)

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
AEROAGENT_LLM_PROVIDER=groq
AEROAGENT_LLM_MODEL=llama-3.3-70b-versatile
```

> **Note:** Never paste API keys into Python files. `.env` is already in `.gitignore`.

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

### Run AeroAgent-Twin Diagnostics

```bash
python app_aeroagent.py --engine_id 15 --fd_id 2
```

This runs the full agentic diagnostic pipeline:
1. Loads raw telemetry for the specified engine
2. Runs MC Dropout RUL inference with anomaly extraction
3. Validates degradation via the physics digital twin
4. Retrieves relevant MRO SOPs via hybrid RAG
5. Generates a structured MRO work order
6. Outputs a JSON diagnostic report

Save the report to a file:
```bash
python app_aeroagent.py --engine_id 15 --fd_id 2 --output diagnostic_report.json
```

### Run as FastAPI Server

```bash
uvicorn app_aeroagent:app --reload
```

Endpoints:
- `GET /health` — service health check
- `POST /predict` — MC Dropout RUL inference only
- `POST /diagnose` — full diagnostic pipeline

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
| `diagnostic_report.json` | AeroAgent-Twin diagnostic output (optional) |

---

## 🏗️ Architecture & Workflow

### Model Architecture — `TurbofanRULModel`

```
Input: [Batch, Window, Sensors]
                 │
                 ▼
┌─────────────────────────────────┐
│  Multi-scale CNN                │  ← Conv1D (k=3, 5, 7) in parallel
│  + GELU + Dropout + LayerNorm   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  BiLSTM (2-layer, hidden=128)   │  ← Bidirectional, captures time dependencies
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  3D Attention                   │  ← Highlights degradation-informative steps
│  + AttentionPool                │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  Learned FC          HC Feature FC     ← Dual-stream: sequential + handcrafted
        └────────┬────────┘
                 │ Concatenate
                 ▼
┌─────────────────────────────────┐
│  MoE Output Head                │  ← 4 Expert MLPs + gated router
│  (entropy reg. + var. penalty)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Predicted RUL (Engine Cycles)  │
└─────────────────────────────────┘
```

### AeroAgent-Twin — End-to-End Diagnostic Architecture

```
Telemetry Input (engine_id + fd_id)
                 │
                 ▼
┌─────────────────────────────────┐
│  TelemetryInferenceService      │  ← api/inference.py
│  MC Dropout (T=50) + Anomaly    │     Loads model_FD00X.pt
│  Detection + Health Status      │     z-score sensor deviation
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  RUL μ, σ, CI       Anomaly Vector
        │                 │
        │                 ▼
        │   ┌─────────────────────────────────┐
        │   │  ThermodynamicDigitalTwin        │  ← physics/digital_twin.py
        │   │  HPC / HPT / Combustor loss      │     thermo_models.py equations
        │   │  Thermal margin + dominant cause  │
        │   └────────────────┬────────────────┘
        │                    │
        │                    ▼
        │   ┌─────────────────────────────────┐
        │   │  HybridVectorStore (RAG)         │  ← rag/vector_store.py
        │   │  0.65×dense + 0.35×keyword       │     turbofan_mro_sop.json
        │   │  + component/fault_code boost     │
        │   └────────────────┬────────────────┘
        │                    │
        └────────┬───────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  MRO Work Order Builder         │  ← mro/work_order.py
│  Urgency, actions, parts, refs  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Final Diagnostic Report        │  ← agents/graph.py
│  (Optional LLM synthesis via    │     agents/llm_report.py
│   Groq / OpenAI)                │
└─────────────────────────────────┘
```

### Solving the Expert Utilisation Problem (MoE Collapse)

A common flaw in standard Mixture of Experts (MoE) architectures is **Expert Collapse**, where the gating network becomes mathematically "lazy" and routes all data to only 1 or 2 experts, leaving the others completely unutilised (dead). 

This pipeline successfully guarantees even distribution across all 4 experts using three strategic mechanics implemented in `model.py`:

1. **Variance Penalty (`var_penalty`)**: We calculate the statistical variance of the mean gate weights across the batch. Minimising this variance explicitly forces the router to spread the workload uniformly (aiming for exactly 25% throughput per expert).
2. **Entropy Bonus (`entropy_bonus`)**: We calculate the entropy of the mean gate distribution. The loss function includes a penalty if the entropy falls short of the theoretical maximum (`np.log(4)`), actively rewarding the network for maintaining balanced expert probabilities.
3. **Smart Bias Initialization**: Inside `ExpertMLP`, the final output bias of every expert is hard-initialized to `0.5` instead of random noise. Because the final network outputs are clamped between `[0, 1]`, starting at `0.5` ensures every expert begins training dead-center in the "active" gradient zone. This prevents experts from dying in the very first optimization step.

### Training Details

| Parameter | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Window Size | 30 | 60 | 30 | 60 |
| Max RUL | 130 | 125 | 125 | 150 |
| Epochs | 300 | 400 | 300 | 450 |
| Learning Rate | 5e-4 | 2e-4 | 5e-4 | 1e-4 |
| Dropout | 0.35 | 0.35 | 0.40 | 0.35 |
| Conditions | 1 | 6 | 1 | 6 |

### Evaluation Metrics

- **RMSE** — Root Mean Squared Error (in engine cycles)
- **NASA Score** — Asymmetric scoring function that penalises late predictions (under-predictions) more than early ones. Here, **`d = Predicted RUL - True RUL`** (the error difference in cycles):

```
Score = Σ (exp(d/10) - 1)   if d ≥ 0  (late prediction)
        Σ (exp(-d/13) - 1)  if d < 0  (early prediction)
```

### 🏆 Final Experimental Results

| Dataset | RMSE | Score | Uncertainty (Avg) |
|:---:|:---:|:---:|:---:|
| **FD001** | 13.99 | 329.34 | ± 3 cycles |
| **FD002** | 11.90 | 659.17 | ± 3 cycles |
| **FD003** | 12.00 | 259.70 | ± 4 cycles |
| **FD004** | 18.44 | 2209.97 | ± 6 cycles |
| **Average** | **14.08** | **864.55** | - |

---

## 📄 RAG Corpus

The RAG system indexes a synthetic MRO SOP corpus at `data/mro_manuals/turbofan_mro_sop.json` containing 5 procedures:

| SOP ID | Component | Fault Code | Urgency |
|---|---|---|---|
| SOP-HPC-001 | HPC | HPC-EROSION | urgent |
| SOP-HPT-002 | HPT | HPT-THERMAL-WEAR | aircraft_on_ground |
| SOP-COMB-003 | Combustor | COMBUSTOR-THERMAL-STRESS | urgent |
| SOP-SENS-004 | Sensors | SENSOR-CALIBRATION | expedite |
| SOP-BRG-005 | Bearings | BEARING-WEAR | aircraft_on_ground |

> **Note:** This corpus is synthetic and generic — it is not sourced from proprietary OEM maintenance manuals. Intended for research/demo use.

---

## 🔍 Verification

Compile-check the new modules:
```bash
python -m compileall api physics rag agents mro app_aeroagent.py
```

Run an end-to-end smoke test:
```bash
python app_aeroagent.py --engine_id 15 --fd_id 2
```