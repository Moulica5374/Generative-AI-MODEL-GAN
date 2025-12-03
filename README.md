# Generative AI for Imbalanced Dataset Synthesis
## Advanced GAN Architecture for Financial Transaction Data Generation

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Summary

This project demonstrates a **production-grade implementation of Generative Adversarial Networks (GANs)** to solve the critical challenge of severe class imbalance in financial datasets. By generating synthetic minority-class samples, the system enables more robust machine learning model training while preserving data privacy and statistical integrity.

**Business Impact**: Addresses the 100:1 class imbalance ratio typical in financial anomaly datasets, reducing false negatives in production systems.

---

## Problem Statement

### The Challenge
Financial institutions face a critical data science problem:
- **Imbalanced datasets**: Anomalous transactions represent <1% of total volume
- **Regulatory constraints**: Limited access to sensitive financial data
- **Model bias**: Traditional ML models overwhelmingly favor majority class
- **High false negative cost**: Missing anomalies can cost millions

### Traditional Approaches Fall Short
- **SMOTE/ADASYN**: Generate unrealistic synthetic samples
- **Class weighting**: Doesn't increase sample diversity
- **Downsampling**: Loses valuable majority class information
- **Data collection**: Expensive, time-consuming, privacy concerns

### The Solution: Generative AI
This implementation leverages **Generative Adversarial Networks** to create statistically valid synthetic samples that:
- Preserve complex multi-dimensional relationships
- Generate unlimited realistic samples
- Maintain data privacy (no real data exposure)
- Enable balanced training datasets

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GAN Training Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Real Data (492 samples) ──┐                                │
│                             │                                │
│                             ▼                                │
│                      ┌──────────────┐                        │
│  Random Noise ──────►│  Generator   │                        │
│   (Latent Space)     │   Network    │                        │
│                      └──────┬───────┘                        │
│                             │                                │
│                             │ Synthetic Data                 │
│                             │                                │
│                             ▼                                │
│                      ┌──────────────┐                        │
│                      │Discriminator │                        │
│                      │   Network    │                        │
│                      └──────┬───────┘                        │
│                             │                                │
│                             ▼                                │
│                    Real vs Fake Classification               │
│                                                              │
│                    ◄── Adversarial Training ──►              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Generator Network Architecture

**Purpose**: Transform random noise into realistic financial transaction features

```python
Input Layer:        29 dimensions (latent vector)
                    ↓
Hidden Layer 1:     32 neurons + ReLU + BatchNorm
                    ↓
Hidden Layer 2:     64 neurons + ReLU + BatchNorm
                    ↓
Hidden Layer 3:     128 neurons + ReLU + BatchNorm
                    ↓
Output Layer:       29 dimensions (synthetic features)
                    Linear activation

Optimizer:          Adam
Loss Function:      Binary Crossentropy
```

**Design Rationale**:
- **Progressive expansion**: Gradually increases dimensionality to learn complex patterns
- **Batch Normalization**: Stabilizes training, prevents mode collapse
- **He Initialization**: Prevents vanishing gradients in ReLU networks
- **Linear output**: Allows continuous feature generation

### Discriminator Network Architecture

**Purpose**: Distinguish between real and synthetic samples

```python
Input Layer:        29 dimensions (transaction features)
                    ↓
Hidden Layer 1:     128 neurons + ReLU
                    ↓
Hidden Layer 2:     64 neurons + ReLU
                    ↓
Hidden Layer 3:     32 neurons + ReLU
                    ↓
Hidden Layer 4:     32 neurons + ReLU
                    ↓
Hidden Layer 5:     16 neurons + ReLU
                    ↓
Output Layer:       1 dimension (probability)
                    Sigmoid activation

Optimizer:          Adam (lr=0.001)
Loss Function:      Binary Crossentropy
```

**Design Rationale**:
- **Progressive compression**: Extracts hierarchical features
- **Deep architecture**: Captures complex decision boundaries
- **Sigmoid output**: Binary classification (real=1, fake=0)

---

## Technical Implementation

### Data Preprocessing Pipeline

```python
1. Data Ingestion
   └─ Load: 50,492 transactions (50,000 genuine + 492 anomalous)
   
2. Feature Engineering
   ├─ PCA-transformed features (V1-V28)
   ├─ Transaction amount normalization
   └─ Temporal feature extraction
   
3. Class Separation
   └─ Isolate minority class (492 samples)
   
4. Scaling & Normalization
   ├─ StandardScaler fit on minority class
   └─ Zero mean, unit variance
   
5. Dimensionality Analysis
   └─ PCA visualization (2D projection)
```

### Training Strategy

**Adversarial Training Loop**:
```python
For each epoch (1 to 5000):
    # Phase 1: Train Discriminator
    ├─ Sample real data batch
    ├─ Generate fake data batch
    ├─ Train on real (label=1)
    ├─ Train on fake (label=0)
    └─ Calculate discriminator loss
    
    # Phase 2: Train Generator
    ├─ Generate new fake batch
    ├─ Train generator to fool discriminator
    └─ Calculate generator loss
    
    # Phase 3: Monitor Performance
    └─ Every 10 epochs: Visualize synthetic vs real distribution
```

**Hyperparameters**:
- **Epochs**: 5,000
- **Batch Size**: 64
- **Learning Rate**: 0.001 (Adam)
- **Latent Dimension**: 29
- **Monitoring Frequency**: Every 10 epochs

### Evaluation Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Generator Loss** | Binary crossentropy on discriminator feedback | Measures ability to fool discriminator |
| **Discriminator Loss** | Classification accuracy on real vs fake | Measures detection capability |
| **PCA Visualization** | 2D projection overlap analysis | Visual quality assessment |
| **Feature Distribution** | Statistical similarity (mean, std) | Quantitative validation |
| **Correlation Matrix** | Feature relationship preservation | Structural integrity check |

---

## Key Results & Achievements

### Model Performance

 **Successfully trained for 5,000 epochs** with stable convergence  
 **Generated 1,000 high-quality synthetic samples** from 492 real samples  
 **Maintained statistical properties** across 29-dimensional feature space  
 **Achieved visual similarity** in PCA projections (real vs synthetic overlap)  

### Technical Accomplishments

1. **Solved Mode Collapse**
   - Implemented batch normalization
   - Used progressive training strategy
   - Achieved diverse synthetic sample generation

2. **Preserved Complex Relationships**
   - Maintained inter-feature correlations
   - Reproduced statistical distributions
   - Generated samples span full feature space

3. **Production-Ready Pipeline**
   - Modular, reusable architecture
   - Scalable to larger datasets
   - Easy integration with downstream ML models



---

## Technology Stack

### Core Frameworks
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 2.8+ | High-level neural network API |
| **NumPy** | 1.21+ | Numerical computing |
| **Pandas** | 1.3+ | Data manipulation |
| **Scikit-learn** | 1.0+ | ML utilities, preprocessing |

### Visualization & Analysis
| Technology | Purpose |
|------------|---------|
| **Matplotlib** | Static plotting |
| **Seaborn** | Statistical visualization |
| **Plotly** | Interactive dashboards |

### Development Environment
| Tool | Purpose |
|------|---------|
| **Jupyter** | Interactive development |
| **Git** | Version control |
| **Virtual Environment** | Dependency isolation |

---

## Installation & Setup

### Prerequisites
```bash
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum
- GPU recommended (optional, for faster training)
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/moulica5374/generative-ai-imbalanced-data.git
cd generative-ai-imbalanced-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook
```

### Dependencies Installation
```bash
pip install tensorflow
pip install kera
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install plotly
```

---

## Usage Examples

### Basic Usage - Generate Synthetic Data

```python
from models import build_generator, build_discriminator, build_gan, train_gan
import numpy as np

# Load and preprocess your imbalanced dataset
minority_class_data = load_minority_class()  # Shape: (n_samples, 29)

# Build GAN architecture
generator = build_generator(input_dim=29)
discriminator = build_discriminator(input_dim=29)
gan = build_gan(generator, discriminator)

# Train GAN
history = train_gan(
    generator=generator,
    discriminator=discriminator,
    gan=gan,
    real_data=minority_class_data,
    epochs=5000,
    batch_size=64
)

# Generate synthetic samples
num_synthetic_samples = 1000
noise = np.random.normal(0, 1, (num_synthetic_samples, 29))
synthetic_data = generator.predict(noise)

print(f"Generated {synthetic_data.shape[0]} synthetic samples")
```

### Advanced Usage - Custom Architecture

```python
# Customize generator architecture
generator = build_generator(
    input_dim=29,
    hidden_layers=[32, 64, 128, 256]  # Deeper network
)

# Customize discriminator with different learning rate
discriminator = build_discriminator(
    input_dim=29,
    hidden_layers=[256, 128, 64, 32, 16],
    learning_rate=0.0002  # Lower learning rate
)

# Build and train with custom configuration
gan = build_gan(generator, discriminator, learning_rate=0.0002)
history = train_gan(gan, real_data, epochs=10000, batch_size=128)
```

### Visualization & Evaluation

```python
from utils import plot_pca_comparison, plot_training_history

# Visualize real vs synthetic distribution
plot_pca_comparison(
    real_data=minority_class_data,
    synthetic_data=synthetic_data,
    title="Real vs Synthetic Sample Distribution"
)

# Analyze training progression
plot_training_history(history)

# Statistical comparison
from utils import create_comparison_report
stats_report = create_comparison_report(
    real_data=minority_class_data,
    synthetic_data=synthetic_data
)
print(stats_report)
```

---

## Project Structure

```
generative-ai-imbalanced-data/
│
├── models/
│   ├── __init__.py
│   ├── generator.py           # Generator network architecture
│   ├── discriminator.py       # Discriminator network architecture
│   └── gan.py                 # Combined GAN model & training loop
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing utilities
│   └── visualization.py       # Plotting and analysis functions
│
├── notebooks/
│   └── GAN_Training.ipynb     # Interactive training notebook
│
├── data/
│   └── .gitkeep               # Dataset directory (add your data here)
│
├── docs/
│   ├── ARCHITECTURE.md        # Detailed architecture documentation
│   └── API_REFERENCE.md       # API documentation
│
├── tests/
│   └── test_models.py         # Unit tests
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Real-World Applications

### 1. Financial Services
- **Credit card fraud detection**: Balance training datasets
- **Loan default prediction**: Generate minority class examples
- **Money laundering detection**: Synthetic suspicious transaction patterns

### 2. Healthcare
- **Rare disease diagnosis**: Augment limited patient data
- **Medical imaging**: Generate synthetic pathology examples
- **Drug adverse event prediction**: Balance event/non-event samples

### 3. Cybersecurity
- **Intrusion detection**: Synthetic attack pattern generation
- **Malware classification**: Augment malicious sample datasets
- **Anomaly detection**: Create diverse anomaly examples



---



### Training Acceleration

**GPU Utilization**:
```python
import tensorflow as tf




### Hyperparameter Tuning Recommendations

| Parameter | Default | Recommendation | Impact |
|-----------|---------|----------------|--------|
| **Learning Rate** | 0.001 | 0.0001-0.001 | Lower = more stable, slower |
| **Batch Size** | 64 | 32-128 | Higher = faster, less stable |
| **Latent Dim** | 29 | 16-64 | Higher = more diverse samples |
| **Hidden Layers** | [32,64,128] | Experiment | Deeper = more complex patterns |

---

## Technical Challenges & Solutions

### Challenge 1: Mode Collapse
**Problem**: Generator produces limited variety of samples  
**Solution**: 
- Implemented batch normalization
- Used progressive layer expansion
- Monitored diversity metrics

### Challenge 2: Training Instability
**Problem**: Oscillating losses, no convergence  
**Solution**:
- Adjusted learning rates independently
- Implemented gradient clipping
- Used Adam optimizer with β₁=0.5

### Challenge 3: High-Dimensional Data
**Problem**: 29 features create complex latent space  
**Solution**:
- Deep generator architecture
- Progressive dimensionality expansion
- PCA validation

### Challenge 4: Limited Training Data
**Problem**: Only 492 real samples available  
**Solution**:
- Small batch sizes (64)
- Extended training epochs (5000)
- Aggressive data augmentation

---

## Future Enhancements




- [ ] Multi-class synthetic data generation
- [ ] Real-time synthetic data API endpoint
- [ ] Integration with MLOps pipeline
- [ ] A/B testing framework for synthetic data quality



---








## Contact & Professional Profile

**Technical Contact**: your.email@example.com  
**LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)  
**GitHub**: [@yourusername](https://github.com/yourusername)  
**Portfolio**: [yourportfolio.com](https://yourportfolio.com)

---

## Acknowledgments

- TensorFlow and Keras development teams for robust deep learning frameworks
- Scikit-learn contributors for preprocessing utilities
- Research community for GAN architecture innovations

---

<div align="center">

**Built with Generative AI expertise and production ML engineering best practices**

⭐ Star this repository if you find it valuable for your work

</div>