import numpy as np
from src.ffnn.core.model import Model
from src.ffnn.activations import get_activation
from src.ffnn.losses import get_loss
from src.ffnn.initializers import get_initializer
from src.ffnn.optimizers import get_optimizer
from src.ffnn.regularizers import get_regularizer

def test_basic_model():
    """Test basic model functionality with synthetic data"""
    print("=" * 60)
    print("Test 1: Basic Model - Binary Classification")
    print("=" * 60)
    
    # Generate synthetic binary classification data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create linearly separable data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    # Split into train and validation
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create model
    model = Model(
        layer_sizes=[n_features, 16, 8, 1],
        activations=[
            get_activation('relu'),
            get_activation('relu'),
            get_activation('sigmoid')
        ],
        loss=get_loss('bce'),
        initializer=get_initializer('normal', mean=0.0, variance=0.01, seed=42),
        optimizer=get_optimizer('gradient_descent', learning_rate=0.01),
        regularizer=None
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    train_loss = model.evaluate(X_train, y_train)
    val_loss = model.evaluate(X_val, y_val)
    
    # Make predictions
    y_pred = model.predict(X_val)
    accuracy = np.mean((y_pred > 0.5) == y_val)
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return model, history


def test_multiclass_classification():
    """Test model with multi-class classification"""
    print("=" * 60)
    print("Test 2: Multi-class Classification (3 classes)")
    print("=" * 60)
    
    # Generate synthetic multi-class data
    np.random.seed(42)
    n_samples = 900
    n_features = 5
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    
    # Create 3 classes
    y_labels = np.zeros(n_samples)
    y_labels[:300] = 0
    y_labels[300:600] = 1
    y_labels[600:] = 2
    
    # Add some signal
    X[:300, 0] += 2
    X[300:600, 1] += 2
    X[600:, 2] += 2
    
    # One-hot encode
    y = np.zeros((n_samples, n_classes))
    y[np.arange(n_samples), y_labels.astype(int)] = 1
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create model
    model = Model(
        layer_sizes=[n_features, 10, n_classes],
        activations=[
            get_activation('relu'),
            get_activation('softmax')
        ],
        loss=get_loss('cce'),
        initializer=get_initializer('xavier', gain=1.0),
        optimizer=get_optimizer('adam', learning_rate=0.01),
        regularizer=get_regularizer('l2', lambda_=0.001)
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_val)
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    
    print(f"\n{'='*60}")
    print(f"Final Multi-class Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return model, history


def test_regression():
    """Test model with regression task"""
    print("=" * 60)
    print("Test 3: Regression Task")
    print("=" * 60)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    y += np.random.randn(n_samples, 1) * 0.1  # Add noise
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create model
    model = Model(
        layer_sizes=[n_features, 16, 8, 1],
        activations=[
            get_activation('relu'),
            get_activation('tanh'),
            get_activation('linear')
        ],
        loss=get_loss('mse'),
        initializer=get_initializer('he', scale=1.0),
        optimizer=get_optimizer('gradient_descent', learning_rate=0.01),
        regularizer=get_regularizer('l1', lambda_=0.001)
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    train_loss = model.evaluate(X_train, y_train)
    val_loss = model.evaluate(X_val, y_val)
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Train MSE: {train_loss:.4f}")
    print(f"  Val MSE: {val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return model, history


def test_save_load():
    """Test model save and load functionality"""
    print("=" * 60)
    print("Test 4: Save and Load Model")
    print("=" * 60)
    
    # Create and train a simple model
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    
    model = Model(
        layer_sizes=[5, 8, 1],
        activations=[get_activation('relu'), get_activation('sigmoid')],
        loss=get_loss('bce'),
        initializer=get_initializer('normal', seed=42),
        optimizer=get_optimizer('gradient_descent', learning_rate=0.01)
    )
    
    # Get prediction before saving
    pred_before = model.predict(X[:10])
    
    # Save model
    model.save('test_model.pkl')
    print("Model saved to 'test_model.pkl'")
    
    # Load model
    loaded_model = Model.load('test_model.pkl')
    print("Model loaded from 'test_model.pkl'")
    
    # Get prediction after loading
    pred_after = loaded_model.predict(X[:10])
    
    # Check if predictions match
    matches = np.allclose(pred_before, pred_after)
    print(f"\nPredictions match: {matches}")
    
    if matches:
        print("✓ Save/Load test passed!")
    else:
        print("✗ Save/Load test failed!")
    
    print(f"{'='*60}\n")
    
    return loaded_model


def test_different_activations():
    """Test model with different activation functions"""
    print("=" * 60)
    print("Test 5: Different Activation Functions")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]
    
    activations_to_test = ['relu', 'sigmoid', 'tanh']
    
    for act_name in activations_to_test:
        print(f"\nTesting with {act_name.upper()} activation...")
        
        model = Model(
            layer_sizes=[10, 16, 1],
            activations=[
                get_activation(act_name),
                get_activation('sigmoid')
            ],
            loss=get_loss('bce'),
            initializer=get_initializer('xavier'),
            optimizer=get_optimizer('adam', learning_rate=0.01)
        )
        
        history = model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=20,
            batch_size=32,
            verbose=0
        )
        
        final_val_loss = history['val_loss'][-1]
        print(f"  Final validation loss: {final_val_loss:.4f}")
    
    print(f"\n{'='*60}\n")

def test_weight_distribution_plotting():
    """Test weight distribution plotting"""
    print("=" * 60)
    print("Test 6: Weight Distribution Plotting")
    print("=" * 60)
    
    # Create and train a model
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = Model(
        layer_sizes=[n_features, 32, 16, 8, 1],
        activations=[
            get_activation('relu'),
            get_activation('relu'),
            get_activation('relu'),
            get_activation('sigmoid')
        ],
        loss=get_loss('bce'),
        initializer=get_initializer('xavier', gain=1.0),
        optimizer=get_optimizer('adam', learning_rate=0.01),
        regularizer=None
    )
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    print("\nPlotting weight distributions for layers [0, 1, 2]...")
    model.plot_weight_distribution([0, 1, 2])
    
    print("\nPlotting weight distributions for all layers [0, 1, 2, 3]...")
    model.plot_weight_distribution([0, 1, 2, 3])
    
    print(f"{'='*60}\n")
    
    return model


def test_gradient_distribution_plotting():
    """Test gradient distribution plotting"""
    print("=" * 60)
    print("Test 7: Gradient Distribution Plotting")
    print("=" * 60)
    
    # Create and train a model
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = Model(
        layer_sizes=[n_features, 32, 16, 8, 1],
        activations=[
            get_activation('relu'),
            get_activation('relu'),
            get_activation('relu'),
            get_activation('sigmoid')
        ],
        loss=get_loss('bce'),
        initializer=get_initializer('he', scale=1.0),
        optimizer=get_optimizer('gradient_descent', learning_rate=0.01),
        regularizer=get_regularizer('l2', lambda_=0.001)
    )
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    print("\nPlotting gradient distributions for layers [0, 1, 2]...")
    model.plot_gradient_distribution([0, 1, 2])
    
    print("\nPlotting gradient distributions for all layers [0, 1, 2, 3]...")
    model.plot_gradient_distribution([0, 1, 2, 3])
    
    print(f"{'='*60}\n")
    
    return model


def test_plotting_edge_cases():
    """Test plotting with edge cases"""
    print("=" * 60)
    print("Test 8: Plotting Edge Cases")
    print("=" * 60)
    
    # Create a simple model
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    
    model = Model(
        layer_sizes=[5, 8, 4, 1],
        activations=[
            get_activation('relu'),
            get_activation('tanh'),
            get_activation('sigmoid')
        ],
        loss=get_loss('bce'),
        initializer=get_initializer('normal', seed=42),
        optimizer=get_optimizer('gradient_descent', learning_rate=0.01)
    )
    
    # Test 1: Plot before training (should work for weights, warn for gradients)
    print("\n1. Testing weight plot before training...")
    model.plot_weight_distribution([0, 1])
    
    print("\n2. Testing gradient plot before training (should warn)...")
    model.plot_gradient_distribution([0, 1])
    
    # Train model
    split_idx = 80
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print("\n3. Training model...")
    model.fit(X_train, y_train, X_val, y_val, epochs=10, batch_size=16, verbose=0)
    
    # Test 2: Plot single layer
    print("\n4. Testing single layer plot...")
    model.plot_weight_distribution([1])
    model.plot_gradient_distribution([1])
    
    # Test 3: Empty list
    print("\n5. Testing empty layer list...")
    model.plot_weight_distribution([])
    model.plot_gradient_distribution([])
    
    # Test 4: Invalid indices
    print("\n6. Testing invalid layer indices (should warn and skip)...")
    model.plot_weight_distribution([0, 10, 1])  # 10 is out of range
    model.plot_gradient_distribution([0, -1, 1])  # -1 is invalid
    
    print(f"\n{'='*60}\n")
    
    return model


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING FFNN MODEL TESTS")
    print("=" * 60 + "\n")
    
    # Run all tests
    try:
        test_basic_model()
        test_multiclass_classification()
        test_regression()
        test_save_load()
        test_different_activations()
        test_weight_distribution_plotting()
        test_gradient_distribution_plotting()
        test_plotting_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()