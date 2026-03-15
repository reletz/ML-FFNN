# FFNN Implementation Plan

## Project Structure

```
Tubes1-FFNN/
├── doc/                        # Reports and documentation
│   ├── ffnn_plan.md
│   └── tex/                    # LaTeX source files for the report
├── src/
│   ├── ffnn/                   # Core FFNN library (from-scratch)
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── model.py        # High-level Model class
│   │   │   ├── layer.py        # Dense layer implementation
│   │   │   ├── network.py      # Network orchestration
│   │   │   └── rmsnorm.py      # RMSNorm implementation
│   │   ├── activations/
│   │   │   ├── __init__.py     # Base Activation class + registry
│   │   │   └── Activation.py
│   │   ├── losses/
│   │   │   ├── __init__.py     # Base Loss class + registry
│   │   │   └── Loss.py
│   │   ├── initializers/
│   │   │   ├── __init__.py     # Base Initializer class + registry
│   │   │   └── Initializer.py
│   │   ├── regularizers/
│   │   │   ├── __init__.py     # Base Regularizer class + registry
│   │   │   └── Regularizer.py
│   │   ├── optimizers/
│   │   │   ├── __init__.py     # Base Optimizer class + registry
│   │   │   └── Optimizer.py
│   │   ├── utils/
│   │       ├── __init__.py
│   │       ├── normalization.py          # Data normalization utilities
│   │       └── metrics.py                # Evaluation metrics
│   │   └── autodiff/                     # Automatic Differentiation (40% bonus)
│   │       ├── __init__.py
│   │       ├── tensor.py                 # Tensor class with gradient tracking
│   │       ├── ops.py                    # Differentiable operations
│   │       ├── layer.py                  # AD-based Layer implementation
│   │       ├── network.py                # AD-based Network
│   │       ├── model.py                  # AD-based Model (top-level API)
│   │       └── rmsnorm.py                # AD-based RMSNorm
│   ├── data/
│   │   └── global_student_placement_and_salary.csv
│   └── notebook/
│       └── experiments.ipynb   # All hyperparameter experiments
├── test_model.py               # Unified test module
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## What to Implement in Each Subfolder

### 1. `src/ffnn/activations/`

Each activation is a class inheriting from a base `Activation` class.

| Class     | `forward(x)`           | `backward(x)` (derivative)                     |
|-----------|------------------------|------------------------------------------------|
| Linear    | f(x) = x               | f'(x) = 1                                      |
| ReLU      | f(x) = max(0, x)       | f'(x) = 1 if x > 0, else 0                     |
| Sigmoid   | f(x) = 1 / (1 + e^-x)  | f'(x) = f(x) * (1 - f(x))                      |
| Tanh      | f(x) = tanh(x)         | f'(x) = 1 - tanh²(x)                           |
| Softmax   | f(x_i) = e^x_i / Σe^x_j| Jacobian matrix: diag(s) - s·sᵀ                |

**Requirements:**
- Every class must have `forward(x)` and `backward(x)` methods.
- Both methods must accept and return NumPy arrays.
- Must handle batch inputs (2D arrays where rows are samples).
- The `__init__.py` should export all classes and provide a string-based lookup (registry) so layers can reference activations by name.

**Workflow:**
1. Define abstract base class `Activation` with `forward()` and `backward()`.
2. Implement each concrete activation.
3. Register all in `__init__.py` for lookup by string name.

---

### 2. `src/ffnn/losses/`

Each loss is a class inheriting from a base `Loss` class.

| Class                  | `compute(y_true, y_pred)`                         | `gradient(y_true, y_pred)`                          |
|------------------------|----------------------------------------------------|-----------------------------------------------------|
| MSE                    | (1/n) Σ (y_true - y_pred)²                        | -(2/n)(y_true - y_pred)                             |
| BinaryCrossEntropy     | -(1/n) Σ [y·ln(ŷ) + (1-y)·ln(1-ŷ)]               | -(y/ŷ - (1-y)/(1-ŷ)) / n                           |
| CategoricalCrossEntropy| -(1/n) Σ Σ y_k · ln(ŷ_k)                          | -y/ŷ / n                                           |

**Requirements:**
- `compute()` returns a scalar loss value.
- `gradient()` returns the gradient of the loss w.r.t. predicted output (dL/dŷ), same shape as `y_pred`.
- Must handle batch inputs (average over batch).
- Use natural log (ln = np.log).
- Add small epsilon (e.g., 1e-15) to log arguments to prevent log(0).

**Workflow:**
1. Define abstract base class `Loss` with `compute()` and `gradient()`.
2. Implement each concrete loss function.
3. Register all in `__init__.py`.

---

### 3. `src/ffnn/initializers/`

Each initializer is a class inheriting from a base `Initializer` class.

| Class   | Parameters                         | Behavior                                  |
|---------|------------------------------------|--------------------------------------------|
| Zero    | None                               | All weights and biases set to 0            |
| Uniform | `low`, `high`, `seed`              | Random uniform in [low, high]              |
| Normal  | `mean`, `variance`, `seed`         | Random from N(mean, variance)              |

**Requirements:**
- Each class has an `initialize(shape)` method that returns a NumPy array of the given shape.
- `seed` parameter must be supported for reproducibility using `np.random.RandomState` or `np.random.Generator`.
- Initializers apply to both weights AND biases.

**Workflow:**
1. Define abstract base class `Initializer` with `initialize(shape)`.
2. Implement each concrete initializer with appropriate parameters.
3. Register all in `__init__.py`.

---

### 4. `src/ffnn/regularizers/`

Each regularizer is a class inheriting from a base `Regularizer` class.

| Class | Penalty Term           | Gradient Contribution         |
|-------|------------------------|-------------------------------|
| L1    | λ Σ |w|               | λ · sign(w)                   |
| L2    | (λ/2) Σ w²            | λ · w                         |

**Requirements:**
- `penalty(weights)` returns the regularization penalty value (scalar).
- `gradient(weights)` returns the gradient of the penalty w.r.t. weights (same shape).
- Accept `lambda_` (regularization strength) as a constructor parameter.
- Penalty is added to the total loss; gradient is added to weight gradients during update.

**Workflow:**
1. Define abstract base class `Regularizer` with `penalty()` and `gradient()`.
2. Implement L1 and L2.
3. Register all in `__init__.py`.

---

### 5. `src/ffnn/optimizers/`

Each optimizer is a class inheriting from a base `Optimizer` class.

| Class           | Update Rule                                                         |
|-----------------|---------------------------------------------------------------------|
| GradientDescent | w = w - lr * gradient                                               |
| Adam (bonus)    | Uses first/second moment estimates with bias correction             |

**Requirements:**
- `update(params, grads)` modifies weights in-place given their gradients.
- Accept `learning_rate` as a constructor parameter.
- Adam must track per-parameter first moment (m) and second moment (v) across steps.
- Adam must support `beta1`, `beta2`, `epsilon` hyperparameters.

**Workflow:**
1. Define abstract base class `Optimizer` with `update()`.
2. Implement GradientDescent.
3. Implement Adam (bonus).
4. Register all in `__init__.py`.

---

### 6. `src/ffnn/core/`

This is the central module that ties everything together.

#### `layer.py` — Dense Layer

**Attributes:**
- `weights`: NumPy array of shape `(input_size, output_size)`
- `biases`: NumPy array of shape `(1, output_size)`
- `weight_gradients`: same shape as `weights`, stores dL/dW
- `bias_gradients`: same shape as `biases`, stores dL/db
- `activation`: an Activation instance
- `input_cache`: stores layer input during forward pass (needed for backward)
- `z_cache`: stores pre-activation output (needed for backward)

**Methods:**
- `forward(X)` — Computes `z = X @ W + b`, then `a = activation.forward(z)`. Caches X and z. Returns a.
- `backward(grad_output)` — Receives dL/da from next layer, computes:
  - `dz = grad_output * activation.backward(z_cache)` (element-wise for most; Jacobian for softmax)
  - `dW = input_cache.T @ dz / batch_size`
  - `db = mean(dz, axis=0)`
  - `dX = dz @ W.T` (to pass to previous layer)
  - Stores `dW` in `weight_gradients`, `db` in `bias_gradients`.
  - Returns `dX`.

#### `network.py` — Network

**Attributes:**
- `layers`: list of Layer instances

**Methods:**
- `add_layer(layer)` — Appends a layer.
- `forward(X)` — Sequentially calls `layer.forward()` for each layer. Returns final output.
- `backward(loss_gradient)` — Sequentially calls `layer.backward()` in reverse order, passing gradient from one layer to the previous.

#### `model.py` — Model (Top-Level API)

**Attributes:**
- `network`: a Network instance
- `loss_fn`: a Loss instance
- `optimizer`: an Optimizer instance
- `regularizer`: a Regularizer instance (or None)

**Constructor Parameters:**
- `layer_sizes`: list of ints (e.g., `[input, hidden1, hidden2, output]`)
- `activations`: list of activation names/instances per layer
- `loss`: loss function name/instance
- `initializer`: initializer name/instance with its params
- `regularizer`: regularizer name/instance (optional)
- `optimizer`: optimizer name/instance

**Methods:**
- `fit(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, verbose)` — Training loop:
  1. Shuffle and split data into mini-batches.
  2. For each epoch, for each batch:
     - Forward pass through network.
     - Compute loss.
     - Compute loss gradient.
     - Backward pass through network (populates all layer gradients).
     - Apply regularizer gradient to weight gradients (if applicable).
     - Call optimizer to update all weights.
  3. After each epoch, compute and record training loss and validation loss.
  4. If verbose=1, display progress bar with current train/val loss.
  5. Return history dict: `{"train_loss": [...], "val_loss": [...]}`.
- `predict(X)` — Forward pass only, returns predictions.
- `evaluate(X, y)` — Forward pass + compute loss, return loss value.
- `save(filepath)` — Serialize all weights, biases, and model config to file (e.g., JSON/pickle).
- `load(filepath)` — Deserialize and restore model state.
- `plot_weight_distribution(layer_indices)` — Plot histograms of weight values for specified layers.
- `plot_gradient_distribution(layer_indices)` — Plot histograms of gradient values for specified layers.

#### `rmsnorm.py` (bonus)
- Implements RMS normalization: `x_norm = x / sqrt(mean(x²) + ε) * γ`
- Applied between layers during forward pass.
- Must support manual backpropagation through the normalization operation.

---

### 7. `src/ffnn/utils/`

#### `metrics.py`
- Evaluation metrics: accuracy, precision, recall, F1-score.
- Used for model evaluation in experiments.

---

### 8. `src/ffnn/autodiff/` (40% Bonus — Full FFNN with Automatic Differentiation)

> **PENTING:** Bonus ini adalah **implementasi ulang FFNN menggunakan Automatic Differentiation**, bukan sekedar utility tambahan. Gradien dihitung otomatis melalui computational graph, menggantikan manual backward pass di `core/`.

#### `tensor.py` — Tensor with Gradient Tracking

**Class: `Tensor`**

Wrapper untuk NumPy array yang melacak operasi untuk backward pass otomatis.

**Attributes:**
- `data`: np.ndarray — nilai tensor
- `grad`: np.ndarray — gradien (None sampai backward() dipanggil)
- `requires_grad`: bool — apakah tensor ini perlu gradient
- `_backward`: Callable — fungsi untuk propagate gradient ke parents
- `_prev`: set[Tensor] — parent tensors dalam computation graph
- `_op`: str — nama operasi yang membuat tensor ini (untuk debugging)

**Methods:**
- `backward()` — Trigger reverse-mode autodiff dari tensor ini
- `zero_grad()` — Reset gradient ke None/zeros
- `detach()` — Return tensor baru tanpa gradient tracking

**Operator Overloading (return Tensor baru dengan _backward function):**
- `__add__`, `__radd__` — Element-wise addition
- `__sub__`, `__rsub__` — Element-wise subtraction  
- `__mul__`, `__rmul__` — Element-wise multiplication
- `__truediv__` — Element-wise division
- `__pow__` — Element-wise power
- `__neg__` — Negation
- `__matmul__` — Matrix multiplication (paling penting untuk neural network)

**Static/Class Methods:**
- `Tensor.from_numpy(arr, requires_grad=False)` — Create tensor dari numpy array
- `Tensor.zeros(shape, requires_grad=False)` — Create zero tensor
- `Tensor.ones(shape, requires_grad=False)` — Create ones tensor

---

#### `ops.py` — Differentiable Operations

Fungsi-fungsi yang beroperasi pada Tensor dan mendefinisikan backward pass.

**Reduction Operations:**
- `sum(tensor, axis=None)` — Sum dengan gradient broadcasting
- `mean(tensor, axis=None)` — Mean dengan proper gradient scaling

**Activation Functions (operate on Tensor, return Tensor):**
- `relu(x)` — max(0, x), gradient = 1 if x > 0 else 0
- `sigmoid(x)` — 1/(1+exp(-x)), gradient = sigmoid(x) * (1 - sigmoid(x))
- `tanh(x)` — tanh(x), gradient = 1 - tanh²(x)
- `softmax(x, axis=-1)` — exp(x) / sum(exp(x)), dengan stable computation

**Math Operations:**
- `exp(x)` — Element-wise exponential
- `log(x)` — Element-wise natural log (dengan epsilon untuk stability)
- `sqrt(x)` — Element-wise square root

**Loss Functions (return scalar Tensor):**
- `mse_loss(pred, target)` — Mean squared error
- `binary_cross_entropy(pred, target)` — Binary cross entropy
- `cross_entropy(pred, target)` — Categorical cross entropy

---

#### `layer.py` — AD-based Dense Layer

**Class: `ADLayer`**

**Attributes:**
- `weights`: Tensor — shape (input_size, output_size), requires_grad=True
- `biases`: Tensor — shape (1, output_size), requires_grad=True
- `activation`: str — nama activation function ('relu', 'sigmoid', 'tanh', 'softmax', 'linear')

**Methods:**
- `forward(x: Tensor) -> Tensor` — Compute `activation(x @ weights + biases)`
  - Tidak perlu manual caching! Computation graph otomatis track.
- `parameters() -> list[Tensor]` — Return [weights, biases] untuk optimizer

**Perbedaan dengan `core/layer.py`:**
| Aspect | core/layer.py | autodiff/layer.py |
|--------|---------------|-------------------|
| Forward | Manual cache input, z | Automatic via computation graph |
| Backward | Manual dz, dW, db computation | Automatic via `loss.backward()` |
| Gradients | Stored in `weight_gradients`, `bias_gradients` | Stored in `weights.grad`, `biases.grad` |

---

#### `network.py` — AD-based Network

**Class: `ADNetwork`**

**Attributes:**
- `layers`: list[ADLayer]

**Methods:**
- `add_layer(layer: ADLayer)` — Append layer
- `forward(x: Tensor) -> Tensor` — Sequential forward through all layers
- `parameters() -> list[Tensor]` — Collect all trainable parameters
- `zero_grad()` — Reset all parameter gradients

**Perbedaan dengan `core/network.py`:**
- **TIDAK ADA method `backward()`!** Gradient propagation otomatis via `loss.backward()`.

---

#### `rmsnorm.py` — AD-based RMSNorm

**Class: `ADRMSNorm`**
- Implements RMS normalization using Tensor operations.
- `x_norm = x / sqrt(mean(x²) + ε) * γ`
- `γ` (gamma) is a trainable parameter wrapped in naturally tracking `Tensor`.
- Gradient propagation is automatically handled by Autodiff operations.

---

#### `model.py` — AD-based Model (Top-Level API)

**Class: `ADModel`**

**Attributes:**
- `network`: ADNetwork
- `loss_fn`: str — nama loss function ('mse', 'binary_cross_entropy', 'cross_entropy')
- `optimizer`: Optimizer instance
- `regularizer`: Optional[Regularizer]

**Methods:**
- `fit(X_train, y_train, X_val, y_val, epochs, batch_size, verbose)` — Training loop:
  1. Convert batch to Tensor
  2. Forward pass: `pred = network.forward(x_tensor)`
  3. Compute loss: `loss = loss_fn(pred, y_tensor)`
  4. **`loss.backward()`** — Automatic gradient computation!
  5. Apply regularizer gradient (if any)
  6. Optimizer update using `param.grad`
  7. `network.zero_grad()` — Reset for next iteration
- `predict(X)`, `evaluate(X, y)`, `save()`, `load()` — Same interface as core/model.py

**Training Loop Comparison:**

```
# Manual Backprop (core/model.py)          # Automatic Differentiation (autodiff/model.py)
pred = network.forward(X)                   pred = network.forward(x_tensor)
loss = loss_fn.compute(y, pred)             loss = loss_fn(pred, y_tensor)
loss_grad = loss_fn.gradient(y, pred)       loss.backward()  # ← Magic happens here!
network.backward(loss_grad)                 # No manual backward needed!
optimizer.update(params, grads)             optimizer.update(network.parameters())
                                            network.zero_grad()
```

---

### 9. `src/data/`

- Store `global_student_placement_and_salary.csv` here.
- No implementation code; purely data storage.
- Preprocessing is handled in the notebook or via pandas/sklearn utilities (allowed since it's not part of the FFNN itself).

---

### 10. `src/notebook/`

#### `experiments.ipynb` — Main Experiment Notebook

**Sections:**
1. **Data Loading & Preprocessing**
   - Load CSV with pandas.
   - Encode categorical columns (one-hot or label encoding).
   - Normalize/standardize numerical features.
   - Split into train/validation sets.
   - Target: `placement_status` (binary classification).

2. **Experiment 1: Depth & Width**
   - Fix depth, vary width (3 configs). Fix width, vary depth (3 configs).
   - Train each, plot train/val loss curves, compare final accuracy.

3. **Experiment 2: Activation Functions**
   - Base architecture with ≥3 layers.
   - Swap activation on one chosen hidden layer: Linear, ReLU, Sigmoid, Tanh.
   - Compare loss curves + weight/gradient distributions.

4. **Experiment 3: Learning Rate**
   - 3 different learning rates (e.g., 0.001, 0.01, 0.1).
   - Compare loss curves + weight/gradient distributions.

5. **Experiment 4: Regularization**
   - No regularization vs L1 vs L2.
   - Compare loss curves + weight/gradient distributions.

6. **Experiment 5: sklearn Comparison**
   - Train `sklearn.neural_network.MLPClassifier` with matching hyperparameters.
   - Compare final predictions/accuracy only.

7. **(Bonus) Experiment 6: RMSNorm**
   - Without vs with RMSNorm. Compare loss curves + distributions.

8. **(Bonus) Experiment 7: Adam vs GD**
   - Compare convergence speed between standard GD and Adam optimizer.

---

## Cross-Module Workflow

The following describes how modules interact during training:

```
Model.fit()
  │
  ├── For each epoch:
  │     ├── For each mini-batch:
  │     │     │
  │     │     ├── Network.forward(X_batch)
  │     │     │     └── Layer[i].forward(input)
  │     │     │           ├── z = input @ weights + biases        (linear transform)
  │     │     │           ├── a = Activation.forward(z)           (from activations/)
  │     │     │           └── cache input and z for backward
  │     │     │
  │     │     ├── Loss.compute(y_batch, predictions)              (from losses/)
  │     │     │
  │     │     ├── loss_grad = Loss.gradient(y_batch, predictions)
  │     │     │
  │     │     ├── Network.backward(loss_grad)
  │     │     │     └── Layer[i].backward(grad)  (reverse order)
  │     │     │           ├── dz = grad * Activation.backward(z_cache)
  │     │     │           ├── dW = input_cache.T @ dz / batch_size
  │     │     │           ├── db = mean(dz, axis=0)
  │     │     │           ├── dX = dz @ weights.T   (pass to prev layer)
  │     │     │           └── Store dW, db in layer gradients
  │     │     │
  │     │     ├── Regularizer.gradient(weights)  →  add to dW    (from regularizers/)
  │     │     │
  │     │     └── Optimizer.update(all_params, all_grads)         (from optimizers/)
  │     │           └── w = w - lr * grad  (or Adam rule)
  │     │
  │     ├── Compute epoch train_loss over full training set
  │     ├── Compute epoch val_loss over full validation set
  │     └── Record in history; print if verbose=1
  │
  └── Return history {"train_loss": [...], "val_loss": [...]}
```

---

## Constraints & Rules

### Language & Libraries
- **Python 3.13** only.
- FFNN implementation must use **only NumPy** (or similar math libraries) — no PyTorch, TensorFlow, or Keras for the core model.
- **sklearn** is allowed only for the comparison experiment (`MLPClassifier`) and for data preprocessing utilities.
- **Pandas** is allowed for data loading and manipulation.
- **Matplotlib** is allowed for plotting.

### Implementation Constraints
- All computations must support **batch processing** (2D NumPy arrays).
- Forward and backward propagation must handle arbitrary batch sizes.
- Weight initialization must support **seed** parameter for reproducibility.
- Loss computation uses **natural logarithm** (np.log, base e).
- Binary cross-entropy is a special case of categorical cross-entropy with 2 classes.
- Regularization penalty is **added to loss**; regularization gradient is **added to weight gradients**.
- Gradient descent update rule: `w = w - lr * gradient`.

### Training Constraints
- `fit()` must accept: `batch_size`, `learning_rate`, `epochs`, `verbose`.
- `verbose=0`: no output during training.
- `verbose=1`: progress bar + current train/val loss.
- Training must return a history dict with per-epoch `train_loss` and `val_loss`.

### Model State Constraints
- Model must store **weights** and **biases** for every layer.
- Model must store **weight gradients** and **bias gradients** for every layer.
- `plot_weight_distribution(layer_indices)` must accept a list of layer indices and display histograms.
- `plot_gradient_distribution(layer_indices)` must accept a list of layer indices and display histograms.
- `save()` and `load()` must persist full model state (architecture + weights).

### Experiment Constraints
- Experiments must be in a **separate `.ipynb` file**.
- Dataset: `global_student_placement_and_salary` (target: `placement_status`).
- Depth/Width: 3 width variations (fixed depth) + 3 depth variations (fixed width).
- Activation: test all except softmax on one hidden layer of a ≥3 layer architecture.
- Learning rate: 3 different values.
- Regularization: none vs L1 vs L2.
- sklearn comparison: same hyperparameters, compare predictions only.

---

## Implementation Sequence

### Phase 1: Foundation

#### 1. Core Infrastructure
1. **Set up project structure** — Create all folders and `__init__.py` files.
2. **Implement base classes** — Create abstract base classes in each module:
   - `activations/__init__.py` — `Activation` base class with `forward()` and `backward()`
   - `losses/__init__.py` — `Loss` base class with `compute()` and `gradient()`
   - `initializers/__init__.py` — `Initializer` base class with `initialize()`
   - `regularizers/__init__.py` — `Regularizer` base class with `penalty()` and `gradient()`
   - `optimizers/__init__.py` — `Optimizer` base class with `update()`
   - `core/__init__.py` — Base classes for `Layer`, `Network`, `Model`

#### 2. Core Components
1. **Implement core classes** — Build the central architecture:
   - `core/layer.py` — Dense layer with forward/backward propagation
   - `core/network.py` — Network orchestration with layer management
   - `core/model.py` — Top-level API with training loop
2. **Implement activation functions** — All five activations:
   - `activations/Activation.py` — Linear, ReLU, Sigmoid, Tanh, Softmax
3. **Implement loss functions** — All three losses:
   - `losses/Loss.py` — MSE, Binary cross-entropy, Categorical cross-entropy

### Phase 2: Supporting Components

#### 1. Initializers & Regularizers
1. **Implement initializers** — All three initialization methods:
   - `initializers/Initializer.py` — Zero, Uniform, Normal
2. **Implement regularizers** — Both regularization methods:
   - `regularizers/Regularizer.py` — L1, L2

#### 2. Optimizers & Utilities
1. **Implement optimizers** — Core optimization methods:
   - `optimizers/Optimizer.py` — Gradient Descent and Adam
2. **Implement utilities** — Supporting functionality:
   - `core/rmsnorm.py` — RMSNorm normalization (bonus)
   - `utils/metrics.py` — Evaluation metrics (accuracy, precision, recall)

### Phase 3: Testing & Documentation

#### 1. Testing Suite
1. **Unified Test Module** — Test components and complete model functionality in one place:
   - `test_model.py` — End-to-end and component tests for training, predictions, activations, losses, and network operations

#### 2. Documentation & Examples
1. **Technical Report** — Comprehensive documentation via LaTeX:
   - `doc/tex/` — Source files for the formal PDF report detailing the methodology, architecture, and experiments

### Phase 4: Experiments

#### 1. Experiment Setup
1. **Data preprocessing** — Prepare the dataset:
   - Load `global_student_placement_and_salary.csv`
   - Encode categorical features (one-hot or label encoding)
   - Normalize numerical features
   - Split into train/validation sets
2. **Experiment framework** — Set up experiment infrastructure:
   - Create experiment notebook structure
   - Implement visualization functions for loss curves
   - Add weight/gradient distribution plotting

#### 2. Hyperparameter Analysis
1. **Experiment 1: Depth & Width** — Test architectural variations:
   - Fix depth, vary width (3 configurations)
   - Fix width, vary depth (3 configurations)
   - Train each, plot loss curves, compare final accuracy
2. **Experiment 2: Activation Functions** — Test activation impact:
   - Base architecture with ≥3 layers
   - Swap activation on one hidden layer (Linear, ReLU, Sigmoid, Tanh)
   - Compare loss curves + weight/gradient distributions
3. **Experiment 3: Learning Rate** — Test optimization impact:
   - 3 different learning rates (e.g., 0.001, 0.01, 0.1)
   - Compare loss curves + weight/gradient distributions
4. **Experiment 4: Regularization** — Test regularization impact:
   - No regularization vs L1 vs L2
   - Compare loss curves + weight/gradient distributions
5. **Experiment 5: sklearn Comparison** — Benchmark against library:
   - Train `sklearn.neural_network.MLPClassifier` with matching hyperparameters
   - Compare final predictions/accuracy only

#### Bonus Experiments (Optional)
1. **Experiment 6: RMSNorm** — Test normalization impact:
   - Without vs with RMSNorm
   - Compare loss curves + weight/gradient distributions
2. **Experiment 7: Adam vs GD** — Test optimizer impact:
   - Compare convergence speed between standard GD and Adam

### Phase 5: Polish & Submit
1. **Final testing** — Run complete test suite
2. **Documentation review** — Ensure all documentation is complete
3. **Experiment analysis** — Finalize all experiment results
4. **Repository preparation** — Prepare for submission:
   - Update README.md with setup and usage instructions
   - Ensure all deliverables are included
   - Verify code quality and style consistency

---

## Bonus: Automatic Differentiation Implementation (40% Bonus)

> **NOTE:** Implementasi ini adalah **reimplementasi penuh FFNN menggunakan computational graph** untuk automatic gradient computation. Ini **parallel** dengan `core/` — bukan replacement.

### AD Phase 1: Tensor Foundation

#### Step 1.1: Basic Tensor Class
```python
# autodiff/tensor.py
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

**Tasks:**
1. Implement `__init__()` dengan attributes di atas
2. Implement `__repr__()` untuk debugging
3. Implement `backward()` — topological sort + reverse propagation
4. Implement `zero_grad()` — reset gradient ke zeros

#### Step 1.2: Topological Sort for Backward
```python
def backward(self):
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
        node._backward()
```

---

### AD Phase 2: Operator Overloading

#### Step 2.1: Basic Arithmetic
Implement dengan pattern:
```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other), _op='+')
    
    def _backward():
        if self.requires_grad:
            self.grad = self.grad + out.grad if self.grad is not None else out.grad.copy()
        if other.requires_grad:
            other.grad = other.grad + out.grad if other.grad is not None else out.grad.copy()
    
    out._backward = _backward
    return out
```

**Operators to implement:**
| Operator | Method | Gradient (self) | Gradient (other) |
|----------|--------|-----------------|------------------|
| `+` | `__add__` | `out.grad` | `out.grad` |
| `-` | `__sub__` | `out.grad` | `-out.grad` |
| `*` | `__mul__` | `out.grad * other.data` | `out.grad * self.data` |
| `/` | `__truediv__` | `out.grad / other.data` | `-out.grad * self.data / other.data²` |
| `**` | `__pow__` | `n * self.data^(n-1) * out.grad` | — |
| `-x` | `__neg__` | `-out.grad` | — |

**Reverse operators:**
- `__radd__`, `__rsub__`, `__rmul__` — untuk `scalar + tensor`

#### Step 2.2: Matrix Multiplication (CRITICAL)
```python
def __matmul__(self, other):
    out = Tensor(self.data @ other.data, requires_grad=True, _children=(self, other), _op='@')
    
    def _backward():
        if self.requires_grad:
            grad_self = out.grad @ other.data.T
            self.grad = self.grad + grad_self if self.grad is not None else grad_self
        if other.requires_grad:
            grad_other = self.data.T @ out.grad
            other.grad = other.grad + grad_other if other.grad is not None else grad_other
    
    out._backward = _backward
    return out
```

**Key insight:**
- `C = A @ B` → `dA = dC @ B.T`, `dB = A.T @ dC`

---

### AD Phase 3: Operations (ops.py)

#### Step 3.1: Reduction Operations
```python
def tensor_sum(t, axis=None, keepdims=False):
    out = Tensor(np.sum(t.data, axis=axis, keepdims=keepdims), requires_grad=True, _children=(t,), _op='sum')
    
    def _backward():
        if t.requires_grad:
            # Broadcast gradient back to original shape
            grad = np.ones_like(t.data) * out.grad
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out

def tensor_mean(t, axis=None, keepdims=False):
    out = Tensor(np.mean(t.data, axis=axis, keepdims=keepdims), requires_grad=True, _children=(t,), _op='mean')
    
    def _backward():
        if t.requires_grad:
            n = t.data.size if axis is None else t.data.shape[axis]
            grad = np.ones_like(t.data) * out.grad / n
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out
```

#### Step 3.2: Activation Functions
```python
def relu(t):
    out = Tensor(np.maximum(0, t.data), requires_grad=True, _children=(t,), _op='relu')
    
    def _backward():
        if t.requires_grad:
            grad = out.grad * (t.data > 0).astype(float)
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out

def sigmoid(t):
    s = 1 / (1 + np.exp(-t.data))
    out = Tensor(s, requires_grad=True, _children=(t,), _op='sigmoid')
    
    def _backward():
        if t.requires_grad:
            grad = out.grad * s * (1 - s)
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out

def tanh(t):
    tanh_val = np.tanh(t.data)
    out = Tensor(tanh_val, requires_grad=True, _children=(t,), _op='tanh')
    
    def _backward():
        if t.requires_grad:
            grad = out.grad * (1 - tanh_val ** 2)
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out

def softmax(t, axis=-1):
    exp_t = np.exp(t.data - np.max(t.data, axis=axis, keepdims=True))
    s = exp_t / np.sum(exp_t, axis=axis, keepdims=True)
    out = Tensor(s, requires_grad=True, _children=(t,), _op='softmax')
    
    def _backward():
        if t.requires_grad:
            # Simplified: works when combined with cross-entropy
            grad = out.grad * s * (1 - s)  # Diagonal approximation
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out
```

#### Step 3.3: Math Operations
```python
def exp(t):
    out = Tensor(np.exp(t.data), requires_grad=True, _children=(t,), _op='exp')
    
    def _backward():
        if t.requires_grad:
            grad = out.grad * out.data
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out

def log(t, eps=1e-15):
    out = Tensor(np.log(t.data + eps), requires_grad=True, _children=(t,), _op='log')
    
    def _backward():
        if t.requires_grad:
            grad = out.grad / (t.data + eps)
            t.grad = t.grad + grad if t.grad is not None else grad
    
    out._backward = _backward
    return out
```

#### Step 3.4: Loss Functions
```python
def mse_loss(pred, target):
    diff = pred - target
    loss = tensor_mean(diff * diff)
    return loss

def binary_cross_entropy(pred, target, eps=1e-15):
    # -mean(y*log(p) + (1-y)*log(1-p))
    pred_clipped = Tensor(np.clip(pred.data, eps, 1 - eps), requires_grad=pred.requires_grad, _children=(pred,))
    loss = -tensor_mean(target * log(pred_clipped) + (Tensor(1) - target) * log(Tensor(1) - pred_clipped))
    return loss

def cross_entropy(pred, target, eps=1e-15):
    # -mean(sum(y * log(p)))
    pred_clipped = Tensor(np.clip(pred.data, eps, 1 - eps), requires_grad=pred.requires_grad, _children=(pred,))
    loss = -tensor_mean(tensor_sum(target * log(pred_clipped), axis=-1))
    return loss
```

---

### AD Phase 4: Layer & Network

#### Step 4.1: ADLayer
```python
# autodiff/layer.py
from .tensor import Tensor
from . import ops

class ADLayer:
    def __init__(self, input_size, output_size, activation='relu', initializer=None):
        # Initialize with small random weights
        scale = np.sqrt(2.0 / input_size)  # He initialization
        self.weights = Tensor(np.random.randn(input_size, output_size) * scale, requires_grad=True)
        self.biases = Tensor(np.zeros((1, output_size)), requires_grad=True)
        self.activation = activation
    
    def forward(self, x):
        z = x @ self.weights + self.biases
        
        if self.activation == 'relu':
            return ops.relu(z)
        elif self.activation == 'sigmoid':
            return ops.sigmoid(z)
        elif self.activation == 'tanh':
            return ops.tanh(z)
        elif self.activation == 'softmax':
            return ops.softmax(z)
        else:  # linear
            return z
    
    def parameters(self):
        return [self.weights, self.biases]
```

#### Step 4.2: ADNetwork
```python
# autodiff/network.py
class ADNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = None
```

---

### AD Phase 5: ADModel (Training Loop)

```python
# autodiff/model.py
class ADModel:
    def __init__(self, layer_sizes, activations, loss='mse', optimizer=None, regularizer=None):
        self.network = ADNetwork()
        self.loss_name = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        
        for i in range(len(layer_sizes) - 1):
            activation = activations[i] if i < len(activations) else 'linear'
            layer = ADLayer(layer_sizes[i], layer_sizes[i+1], activation)
            self.network.add_layer(layer)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, verbose=0, learning_rate=None):
        if learning_rate is not None:
            self.optimizer.set_learning_rate(learning_rate)
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = Tensor(X_train[batch_idx], requires_grad=False)
                y_batch = Tensor(y_train[batch_idx], requires_grad=False)
                
                # Forward
                pred = self.network.forward(x_batch)
                
                # Loss
                if self.loss_name == 'mse':
                    loss = ops.mse_loss(pred, y_batch)
                elif self.loss_name == 'binary_cross_entropy':
                    loss = ops.binary_cross_entropy(pred, y_batch)
                else:
                    loss = ops.cross_entropy(pred, y_batch)
                
                # Backward — THE MAGIC!
                loss.backward()
                
                # Regularizer gradient
                if self.regularizer:
                    for param in self.network.parameters():
                        if param.grad is not None:
                            param.grad += self.regularizer.gradient(param.data)
                
                # Optimizer update
                params = self.network.parameters()
                grads = [p.grad for p in params]
                self.optimizer.update([p.data for p in params], grads)
                
                # Reset gradients
                self.network.zero_grad()
                
                epoch_loss += loss.data * len(batch_idx)
            
            # Record history
            history["train_loss"].append(epoch_loss / len(X_train))
            val_loss = self.evaluate(X_val, y_val)
            history["val_loss"].append(val_loss)
            
            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs} - loss: {history['train_loss'][-1]:.4f} - val_loss: {val_loss:.4f}")
        
        return history
    
    def predict(self, X):
        x_tensor = Tensor(X, requires_grad=False)
        return self.network.forward(x_tensor).data
    
    def evaluate(self, X, y):
        pred = self.predict(X)
        if self.loss_name == 'mse':
            return np.mean((y - pred) ** 2)
        elif self.loss_name == 'binary_cross_entropy':
            eps = 1e-15
            pred = np.clip(pred, eps, 1 - eps)
            return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        else:
            eps = 1e-15
            pred = np.clip(pred, eps, 1 - eps)
            return -np.mean(np.sum(y * np.log(pred), axis=-1))
```

---

### AD Implementation Checklist

| Step | Component | Status | Test |
|------|-----------|--------|------|
| 1.1 | `Tensor.__init__()` | ☐ | Create tensor, check attributes |
| 1.2 | `Tensor.backward()` | ☐ | Simple `a + b`, verify grads |
| 2.1 | `__add__`, `__sub__` | ☐ | `(a + b).backward()` |
| 2.2 | `__mul__`, `__truediv__` | ☐ | `(a * b).backward()` |
| 2.3 | `__matmul__` | ☐ | `(A @ B).backward()` — **CRITICAL** |
| 3.1 | `sum()`, `mean()` | ☐ | Verify gradient shapes |
| 3.2 | `relu()`, `sigmoid()`, `tanh()` | ☐ | Compare with manual derivatives |
| 3.3 | `exp()`, `log()` | ☐ | Numerical gradient check |
| 3.4 | `mse_loss()`, `cross_entropy()` | ☐ | Compare with `losses/` module |
| 4.1 | `ADLayer.forward()` | ☐ | Single layer forward |
| 4.2 | `ADNetwork.forward()` | ☐ | Multi-layer forward |
| 5.1 | `ADModel.fit()` | ☐ | Train on XOR problem |
| 5.2 | Compare with `core/Model` | ☐ | Same config → same results |

---

### Validation: Compare AD vs Manual Backprop

```python
# Test script to validate AD implementation
import numpy as np
from ffnn.core.model import Model
from ffnn.autodiff.model import ADModel
from ffnn.optimizers import GradientDescent
from ffnn.activations import ReLU, Sigmoid

# Simple XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Manual backprop model
manual_model = Model(
    layer_sizes=[2, 4, 1],
    activations=[ReLU(), Sigmoid()],
    loss=MSE(),
    initializer=Normal(seed=42),
    optimizer=GradientDescent(learning_rate=0.1)
)

# AD model
ad_model = ADModel(
    layer_sizes=[2, 4, 1],
    activations=['relu', 'sigmoid'],
    loss='mse',
    optimizer=GradientDescent(learning_rate=0.1)
)

# Copy weights to ensure same starting point
for i, layer in enumerate(ad_model.network.layers):
    layer.weights.data = manual_model.network.layers[i].weights.copy()
    layer.biases.data = manual_model.network.layers[i].biases.copy()

# Train both
manual_history = manual_model.fit(X, y, X, y, epochs=100, batch_size=4, verbose=0)
ad_history = ad_model.fit(X, y, X, y, epochs=100, batch_size=4, verbose=0)

# Compare loss curves
print("Manual final loss:", manual_history["train_loss"][-1])
print("AD final loss:", ad_history["train_loss"][-1])
# Should be very close!
```