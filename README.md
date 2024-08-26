# Egyptian Hieroglypics Models Analysis 🏺

## 📜 Project Description

This project focuses on the analysis and recognition of Egyptian hieroglyphs using advanced machine learning and deep learning models. The main goal is to build a system that can automatically recognize and classify hieroglyphic symbols from images with high accuracy.


## 🎯 Project Objectives

-  Develop deep learning models for automatic recognition of Egyptian hieroglyphs.
-  Enhance model performance through techniques like data augmentation.
-  Compare different models and evaluate their effectiveness in hieroglyph recognition.


## 🛠️ Technologies Used

The project leverages the following technologies:

- LIME 🔍: Enhances the explainability and interpretability of model predictions.
- YOLOv5 👁️: Implements object detection and real-time hieroglyph recognition.
- CodeCarbon 🌱: Monitors and tracks the carbon footprint associated with model training.
- Hydra 🔧: Facilitates advanced configuration management and experimentation through modular YAML files.
- scikit-learn 📊: Provides evaluation metrics and traditional machine learning models.
- PyTorch 🤖: Core deep learning library for model development.
- Matplotlib 📈: Used for data visualization and performance analysis.
- Python 🐍: The primary programming language used in the project.


## 📁 Dataset

The primary dataset consists of images of Egyptian hieroglyphs called GlyphDataset, which have been pre-processed and split into training, validation, and test sets.

- Dataset Source: The dataset was created using publicly available hieroglyph images [here](https://iamai.nl/downloads/GlyphDataset.zip).
- Preprocessing: Images were resized, normalized, and augmented to improve model performance (with our personal preference).
- Support: A supporting dataset was extracted using utils from [YOLOv5](https://github.com/ultralytics/yolov5.git).


## 🤖 Models

Multiple models were implemented to classify hieroglyphs, including:

- **Convolutional Neural Networks** 🧠 (in particular, **GlyphNet** and **ATCNet**).
- **ResNet** 🚀 (specifically, **TResNet**).
- **Ensemble Learning** 🔄 with 2 and 4 CNN (**TResNet**).

Each model has been trained and evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, other key performance indicators like **emissions** and **performance efficiency** were measured to ensure the models are both effective and sustainable.


## 📊 Results & Evaluation

For a detailed documentation and in-depth study of the results, please refer to the **[📄 Documentation](https://github.com/Endless077/Egyptian_Hieroglyphics_Models_Analysis/blob/main/docs.pdf)** (in Italian).

After training the models, a comprehensive study was conducted to analyze various performance metrics, including:

- **Confusion Matrix**: Visual representation of the model's predictions.
- **Precision & Recall Graphs**: Comparison of performance across models.
- **Accuracy**: Model accuracy scores across different models.

In addition to traditional performance metrics, we also focused on:

- **Emission Evaluation** 🌍: Assessing the environmental impact of model training, including energy consumption and carbon emissions.
- **Performance Efficiency** ⚡: Analyzing the computational efficiency of the models.
- **Explainability** 🔍: Using the **LIME** (Local Interpretable Model-agnostic Explanations) library to improve the interpretability and transparency of model predictions.


## 🛠️ Installation and Setup

To get started with the Egyptian Hieroglyphics Models Analysis project, follow these steps:

### 🖥️ **Clone the Repository:**

   Begin by cloning the repository to your local machine. Open a terminal and run:
   ```bash
   git clone https://github.com/Endless077/Egyptian_Hieroglyphics_Models_Analysis.git
   cd Egyptian_Hieroglyphics_Models_Analysis
   ```

### ⚙️ **Set Up the Environment:**

We provide two methods to set up your environment: using `requirements.txt` or `environment.yml`.

- **Using `requirements.txt`:**
  Install the necessary packages using pip. Ensure you have `pip` installed, then run:
  ```
  pip install -r requirements.txt
  ```

- **Using `environment.yml`:**
  Alternatively, you can use Conda to create an environment. If you have Conda installed, create and activate the environment with:
  ```
  conda env create -f environment.yml
  conda activate egyptian-hieroglyphs
  ```

### 📜 **Explore the Scripts:**

Once the environment is set up, you can explore the project scripts. Each script includes auto-explanatory comments to help you understand its functionality and purpose. Open the scripts in your preferred code editor to review the comments and get insights into the code logic.

### 📧 **Contact a Collaborator:**

If you encounter any issues or need further clarification, feel free to reach out to one of the project collaborators. You can contact them through their GitHub profiles.
They will be happy to assist you with any questions or technical difficulties.


## 👋 Authors

- [Fulvio Serao](https://github.com/Fulvioserao99)
- [Antonio Garofalo](https://github.com/Endless077)
- [Alessia Ture](https://github.com/a-ture)


## 📚 Reference

The development of this project was inspired by the following key document:

[**A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification**](https://ieeexplore.ieee.org/document/9506887)  
-  *Authors: Andrea Barucci, Costanza Cucci, Massimiliano Franci, Marco Loschiavo, Fabrizio Argenti*  
-  *Published in IEEE Access, 2021*  
-  DOI: [10.1109/ACCESS.2021.3110082](https://doi.org/10.1109/ACCESS.2021.3110082)  

  This paper explores how deep learning can decode ancient Egyptian hieroglyphs, providing foundational insights and methodologies that influenced this project.


## 💾 License

This project is licensed under the GNU General Public License v3.0.

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

![Static Badge](https://img.shields.io/badge/UniSA-EHMA-red?style=plastic)
