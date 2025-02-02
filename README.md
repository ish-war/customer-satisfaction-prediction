# Web Interface
![Screenshot 2025-02-02 150423](https://github.com/user-attachments/assets/90f83e62-47de-486f-9545-24b0482484c4)

![Screenshot 2025-02-02 150503](https://github.com/user-attachments/assets/a8337610-d923-4a35-8b45-1c4157eb957e)



# Customer Satisfaction (CSAT) Prediction using Deep Learning

## Overview
This project aims to predict Customer Satisfaction (CSAT) scores using a Deep Learning Artificial Neural Network (ANN) model. In the context of e-commerce, understanding customer satisfaction through their interactions and feedback is crucial for enhancing service quality, customer retention, and overall business growth. The model is trained on customer interaction data from an e-commerce platform, helping businesses understand and improve customer experience. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment via a Streamlit web application.

## Features
- **Predict CSAT Scores**: Uses an ANN model to forecast customer satisfaction.
- **Preprocessed Data**: Cleansed and engineered features for better model performance.
- **Exploratory Data Analysis (EDA)**: Performed data visualization to understand patterns and correlations in the dataset.
- **Feature Engineering**: Applied techniques such as scaling, encoding categorical variables, and handling missing values.
- **Advanced Deep Learning Techniques**:
  - **Dropout Layers**: Added to prevent overfitting and improve generalization.
  - **Regularization**: Used L1 regularization techniques to optimize the model.
  - **Batch Normalization**: Implemented to stabilize learning and speed up training.
  - **Early Stopping**: Enabled to prevent overfitting by monitoring validation loss.
- **Interactive Web Application**: Built with Streamlit for easy local deployment.
- **Deployment Ready**: Supports local deployment with Docker and Streamlit.

## Tech Stack
- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Frameworks**: Streamlit

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (latest version)
- Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ish-war/customer-satisfaction-prediction.git
   cd your-repository
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the web application in your browser.
2. Upload customer interaction data or input values manually.
3. Click on the **Predict** button to get the CSAT score.
4. Analyze results and insights.

## Model Details
- **Architecture**: Multi-layer ANN with ReLU activation.
- **Loss Function**: categorical cross-entropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy
- **Hyperparameter Tuning**: Conducted experiments with different learning rates, batch sizes, and activation functions to improve performance.

## Contributing
Feel free to contribute by submitting issues or pull requests.

