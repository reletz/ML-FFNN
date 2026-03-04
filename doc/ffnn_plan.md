# FFNN Implementation Plan

## Project Structure

```
Tubes1-FFNN/
├── doc/                        # Reports and documentation
│   └── ffnn_plan.md
├── src/
│   ├── ffnn/                   # Core FFNN library (from-scratch)
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── model.py        # High-level Model class
│   │   │   ├── layer.py        # Dense layer implementation
│   │   │   └── network.py      # Network orchestration
│   │   ├── activations/
│   │   │   ├── __init__.py     # Base Activation class + registry
│   │   │   ├── linear.py
│   │   │   ├── relu.py
│   │   │   ├── sigmoid.py
│   │   │   ├── tanh.py
│   │   │   └── softmax.py
│   │   ├── losses/
│   │   │   ├── __init__.py     # Base Loss class + registry
│   │   │   ├── mse.py
│   │   │   ├── binary_crossentropy.py
│   │   │   └── categorical_crossentropy.py
│   │   ├── initializers/
│   │   │   ├── __init__.py     # Base Initializer class + registry
│   │   │   ├── zero.py
│   │   │   ├── uniform.py
│   │   │   └── normal.py
│   │   ├── regularizers/
│   │   │   ├── __init__.py     # Base Regularizer class + registry
│   │   │   ├── l1.py
│   │   │   └── l2.py
│   │   ├── optimizers/
│   │   │   ├── __init__.py     # Base Optimizer class + registry
│   │   │   ├── gradient_descent.py
│   │   │   └── adam.py         # (bonus)
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── normalization.py          # RMSNorm (bonus)
│   │       └── automatic_differentiation.py  # (bonus)
│   ├── data/
│   │   └── global_student_placement_and_salary.csv
│   └── notebook/
│       └── experiments.ipynb   # All hyperparameter experiments
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

---

### 7. `src/ffnn/utils/`

#### `normalization.py` (bonus — RMSNorm)
- Implements RMS normalization: `x_norm = x / sqrt(mean(x²) + ε) * γ`
- Applied between layers during forward pass.
- Must support backpropagation through the normalization operation.

#### `automatic_differentiation.py` (bonus)
- Implements a computation graph with `Value` nodes that track operations.
- Each `Value` stores its data, gradient, and a backward function.
- Operations (+, *, -, /, **) create new `Value` nodes linked to their parents.
- Calling `.backward()` on the final node propagates gradients through the graph.

---

### 8. `src/data/`

- Store `global_student_placement_and_salary.csv` here.
- No implementation code; purely data storage.
- Preprocessing is handled in the notebook or via pandas/sklearn utilities (allowed since it's not part of the FFNN itself).

---

### 9. `src/notebook/`

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
   - `activations/linear.py` — Identity function
   - `activations/relu.py` — Rectified Linear Unit
   - `activations/sigmoid.py` — Logistic function
   - `activations/tanh.py` — Hyperbolic tangent
   - `activations/softmax.py` — Normalized exponential
3. **Implement loss functions** — All three losses:
   - `losses/mse.py` — Mean Squared Error
   - `losses/binary_crossentropy.py` — Binary classification
   - `losses/categorical_crossentropy.py` — Multi-class classification

### Phase 2: Supporting Components

#### 1. Initializers & Regularizers
1. **Implement initializers** — All three initialization methods:
   - `initializers/zero.py` — Zero initialization
   - `initializers/uniform.py` — Uniform distribution with bounds
   - `initializers/normal.py` — Normal distribution with mean/variance
2. **Implement regularizers** — Both regularization methods:
   - `regularizers/l1.py` — Lasso regularization
   - `regularizers/l2.py` — Ridge regularization

#### 2. Optimizers & Utilities
1. **Implement optimizers** — Core optimization methods:
   - `optimizers/gradient_descent.py` — Standard gradient descent
   - `optimizers/adam.py` — Adam optimizer (bonus feature)
2. **Implement utilities** — Supporting functionality:
   - `utils/normalization.py` — RMSNorm normalization (bonus)
   - `utils/automatic_differentiation.py` — Automatic differentiation (bonus)
   - `utils/metrics.py` — Evaluation metrics (accuracy, precision, recall)

### Phase 3: Testing & Documentation

#### 1. Testing Suite
1. **Unit tests** — Test each component in isolation:
   - `tests/test_activations.py` — Test all activation functions
   - `tests/test_losses.py` — Test loss computations and gradients
   - `tests/test_initializers.py` — Test weight initialization
   - `tests/test_regularizers.py` — Test regularization penalties
   - `tests/test_optimizers.py` — Test parameter updates
2. **Integration tests** — Test complete model functionality:
   - `tests/test_model.py` — Test model training and prediction
   - `tests/test_network.py` — Test network forward/backward propagation

#### 2. Documentation & Examples
1. **API documentation** — Generate comprehensive documentation:
   - `docs/api_reference.md` — Complete API reference
   - `docs/tutorials/` — Getting started, building models, advanced features
2. **Example implementations** — Create working examples:
   - `docs/examples/classification_example.py` — Binary classification
   - `docs/examples/regression_example.py` — Regression task
   - `docs/examples/custom_activations.py` — Adding custom activations

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