# ML-FFNN

Repository ini berisi implementasi **Feed-Forward Neural Network (FFNN) from scratch** untuk tugas besar pembelajaran mesin. Proyek dibangun dengan dua pendekatan:

1. **`core/`** — implementasi FFNN dengan **manual backpropagation**.
2. **`autodiff/`** — implementasi ulang FFNN dengan **automatic differentiation** sebagai bonus.

Selain model utama, repository ini juga memuat modul pendukung seperti **activation functions, loss functions, initializers, regularizers, optimizers**, eksperimen pada notebook, serta dokumentasi laporan di folder `doc/`. Dataset yang digunakan adalah **`global_student_placement_and_salary.csv`** dengan target prediksi `placement_status`.

## Struktur Singkat Repository

```text
ML-FFNN/
├── doc/                    # Dokumentasi dan laporan
├── src/
│   ├── data/               # Dataset
│   ├── ffnn/
│   │   ├── activations/    # Fungsi aktivasi
│   │   ├── autodiff/       # FFNN berbasis automatic differentiation
│   │   ├── core/           # FFNN manual backpropagation
│   │   ├── initializers/   # Inisialisasi bobot
│   │   ├── losses/         # Fungsi loss
│   │   ├── optimizers/     # Optimizer
│   │   └── regularizers/   # Regularisasi
│   └── notebook/           # Notebook eksperimen
├── main.py                 # Entry point sederhana
├── test_model.py           # Pengujian model
├── requirements.txt        # Dependensi proyek
└── README.md
```

## Fitur Utama

- Implementasi FFNN from scratch berbasis **NumPy**
- Mendukung mini-batch training
- Mendukung beberapa fungsi aktivasi: **Linear, ReLU, Sigmoid, Tanh, Softmax**
- Mendukung beberapa loss: **MSE, Binary Cross-Entropy, Categorical Cross-Entropy**
- Mendukung initializer, optimizer, dan regularizer modular
- Mendukung eksperimen hyperparameter pada notebook:
  - depth dan width
  - fungsi aktivasi
  - learning rate
  - regularisasi
  - perbandingan dengan `sklearn`
- Bonus:
  - **RMSNorm**
  - **Automatic Differentiation**

## Requirements

Pastikan environment menggunakan **Python 3.13** sesuai rencana implementasi proyek.

Dependensi utama:

- `numpy>=2.1.0`
- `pandas>=2.2.0`
- `scikit-learn>=1.5.0`
- `matplotlib>=3.9.0`
- `ipykernel`
- `notebook`
- `tqdm>=4.66.0`

## Cara Setup

### 1. Clone repository

```bash
git clone <URL_REPOSITORY>
cd ML-FFNN
```

### 2. Buat virtual environment

**Windows (PowerShell / CMD):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Opsional) Install project dalam mode editable

Jika ingin import modul lebih nyaman saat development:

```bash
pip install -e .
```

## Cara Menjalankan Program

### A. Menjalankan notebook eksperimen

Notebook utama ada di folder `src/notebook/`.

```bash
jupyter notebook src/notebook/experiments.ipynb
```

Atau buka file notebook langsung dari VS Code / Jupyter extension.

### B. Menjalankan file utama

Jika `main.py` dipakai sebagai entry point:

```bash
python main.py
```

### C. Menjalankan pengujian model

```bash
python test_model.py
```

## Pembagian Tugas Anggota Kelompok


### Anggota 1 — Muhammad Aulia Azka (13523137)
- Menyusun `experiments.ipynb`
- RMSNorm
- Laporan

### Anggota 2 — Frederiko Eldad Mugiyono (13523147)
- Membuat Core Model
- Laporan

### Anggota 3 — Naufarrel Zhafif Abhista (13523149)
- Mengimplementasikan modul `activations`, `losses`, `initializers`, `optimizers`, dan `regularizers`
- Bonus Autodiff, Adam, LeakyReLU, ELU, Xavier, dan He
- Laporan



