# GAN for Imbalanced Dataset Synthesis
### Solving the 100:1 Class Imbalance Problem in Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

---

##  The Problem

Financial fraud detection faces a critical challenge: **extreme class imbalance**. In real-world datasets, fraudulent transactions represent less than 1% of all data, causing ML models to:
- Overwhelmingly predict "not fraud" 
- Miss critical fraud cases (high false negatives)
- Cost financial institutions millions in losses

Traditional solutions like SMOTE generate unrealistic samples that don't capture complex patterns.

---

##  The Solution

Built a **production-grade GAN architecture** that generates statistically valid synthetic fraud samples, enabling balanced model training without exposing real customer data.

### Key Results
-  **Generated 1,000 synthetic fraud samples** from only 492 real samples
-  **Maintained statistical integrity** across 29-dimensional feature space  
-  **Preserved complex correlations** between features
-  **Achieved visual similarity** in PCA projections (real vs synthetic)

---

##  Technical Architecture

**Generator Network**: Transforms random noise → realistic fraud patterns
- Input: 29D latent vector → Layers: 32→64→128 neurons → Output: 29D synthetic features
- Batch normalization prevents mode collapse
- He initialization for stable training

**Discriminator Network**: Distinguishes real from synthetic samples  
- Input: 29D features → Layers: 128→64→32→32→16 neurons → Output: Real/Fake probability
- Deep architecture captures complex decision boundaries
- Binary classification (real=1, fake=0)

**Training**: Adversarial process over 5,000 epochs with batch size 64
- Discriminator learns to detect fakes
- Generator learns to fool discriminator
- Nash equilibrium produces realistic synthetic data

---

##  Validation

<p align="center">
  <img src="https://github.com/user-attachments/assets/a1a5969e-0580-4663-9164-8f7f82e99ad7" width="700" alt="GAN Architecture Diagram"/>
</p>

**PCA Visualization**: Synthetic samples (orange) cluster with real samples (blue), demonstrating the generator learned the underlying distribution of fraudulent transactions.

---

##  Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/gan-imbalanced-data.git
cd gan-imbalanced-data
pip install -r requirements.txt

# Train GAN
python train_gan.py --epochs 5000 --batch-size 64

# Generate synthetic data
python generate_samples.py --num-samples 1000 --output synthetic_fraud.csv
```

---

## Business Impact

**For Financial Institutions:**
- Reduce false negatives in fraud detection by providing balanced training data
- Enable privacy-preserving data augmentation (no real customer data exposed)
- Scale fraud detection models with unlimited synthetic samples

**Production Applications:**
- Credit card fraud detection
- Money laundering detection  
- Loan default prediction
- Insurance claims fraud

---

##  Tech Stack

**Core**: Python 3.8+ • TensorFlow 2.x • Keras  
**Data**: NumPy • Pandas • Scikit-learn  
**Visualization**: Matplotlib • Seaborn • Plotly

---

##  Project Structure

```
gan-imbalanced-data/
├── train_gan.py              # Main training script
├── generate_samples.py       # Generate synthetic data
├── models/
│   ├── generator.py         # Generator architecture
│   ├── discriminator.py     # Discriminator architecture
│   └── gan.py               # Combined GAN model
├── utils/
│   ├── preprocessing.py     # Data preprocessing
│   └── visualization.py     # Plotting utilities
├── notebooks/
│   └── GAN_Training.ipynb   # Interactive exploration
├── requirements.txt
└── README.md
```

---

## Future Enhancements

- [ ] Conditional GAN (CGAN) for controlled synthetic data generation
- [ ] Wasserstein GAN (WGAN) for improved training stability
- [ ] Integration with MLOps pipeline (MLflow, DVC)
- [ ] Real-time synthetic data API endpoint

---

##  Author

**Moulica** - Data Scientist & ML Engineer  
📧 your.email@example.com  
💼 [LinkedIn](https://linkedin.com/in/yourprofile) • [Portfolio](https://yourportfolio.com) • [GitHub](https://github.com/yourusername)

MS in Artificial Intelligence @ Iowa State University | Graduate Research Assistant  
Focus: Neural Network Activation Manipulation, Automated Program Repair, MLOps

---

##  License

MIT License - see [LICENSE](LICENSE) file for details

---

<div align="center">
  
** If this project helped you, please star it!**

Built with expertise in Deep Learning, GANs, and Production ML Systems

</div>
