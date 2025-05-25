# Soil-classification
Submission for annam.ai competition
# Soil Type Classification using ResNet18

This project classifies soil images into four categories using a fine-tuned ResNet18 model. It is structured for reproducibility, modularity, and ease of experimentation.

---

## 🧠 Model

- **Backbone**: `ResNet18` from `torchvision.models`
- **Classifier**: Final FC layer replaced with `nn.Linear(in_features, 5)`
- **Pretrained**: ImageNet weights used for transfer learning


---



## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/satviknight/soil-classification.git
cd soil-classification

# (Recommended) Create a virtual environment
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```
---

## 🚀 Training
To train the model from scratch or fine-tune it:

Place your dataset in the proper structure as described above.

Open and run all cells in notebooks/training.ipynb.

The best model will be saved as best_model.pth.
---

## 🔍 Inference
To generate predictions on the test set:

Make sure best_model.pth is available (from training).

Open and run all cells in notebooks/inference.ipynb.

The output file submission.csv will be created.

---

 ## Dependencies
Python 3.8+

PyTorch

torchvision

pandas

scikit-learn

matplotlib

PIL (Pillow)

You can install them using:

pip install torch torchvision pandas scikit-learn matplotlib pillow

---

## 👤 Author
Satvik Keshtwal – @satviknight          Shalini Mittal - @sha-lini-mittal
