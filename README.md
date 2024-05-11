# NEWS-DETECTION MACHINE-LEARNING & BERT transformer

### Natural Language Processing 

**Author:** Mostafa Magdy Hassan  

---
![Project Image](https://github.com/Mostafa10770/Fake-News-Detection/blob/main/Flask_deploy.png)


## Project Description:

The NEWS-DETECTION-MACHINE-LEARNING project aims to develop a robust machine learning system for the detection of fake news. Leveraging Natural Language Processing (NLP) techniques, the project seeks to contribute to the field of computer science by creating an accurate model capable of discerning between authentic and deceptive news articles. The ultimate goal is to deploy this model using Flask, enabling real-time prediction through a user-friendly web application.

### Objectives:

- Implement a machine learning model utilizing NLP techniques for text analysis and classification.
- Train the model on a diverse dataset comprising both genuine and deceptive news articles.
- Conduct rigorous testing and validation to evaluate the model's performance.
- Deploy the trained model via Flask to provide a seamless user experience for fake news prediction.

### Challenges:

The project encounters challenges such as:

- Developing a robust NLP model capable of understanding context and subtle linguistic nuances.
- Ensuring the model's adaptability to diverse and evolving patterns of fake news.
- Designing an efficient and user-friendly web application for model deployment.

---

## Problem Statement:

### Problem Definition:

The project endeavors to tackle the escalating concern of fake news by constructing an intelligent system capable of distinguishing between genuine and misleading information. With misinformation rampant across various online platforms, there is an imperative need for advanced tools proficient in analyzing and categorizing textual content effectively.

### Significance:

Fake news erodes public trust in information sources, potentially leading to significant social and political ramifications. This project aligns with the learning objectives of the course by offering hands-on experience in implementing machine learning solutions to address real-world problems. It emphasizes the practical application of NLP techniques for information verification.

---

### Tools:

- Python for coding and implementation.
- Scikit-learn and TensorFlow/Keras for machine learning model development.
- NLP libraries such as NLTK and spaCy for text processing.
- Flask for deploying the model as a web application.


---

## Data:

### Data Sources:

The project will utilize the "WELFake" dataset, a comprehensive collection of 72,134 news articles meticulously curated for fake news detection. This dataset amalgamates four well-known news datasets, namely Kaggle, McIntire, Reuters, and BuzzFeed Political, totaling 35,028 real and 37,106 fake news instances. The amalgamation of these datasets aims to prevent overfitting of classifiers and enhance the quality of the training data for more effective machine learning.

### Dataset Description:

The "WELFake" dataset comprises diverse news articles covering various topics from different sources. It consists of four key columns:

1. Serial Number: A unique identifier assigned to each data entry.
2. Title: Descriptive text summarizing the news heading.
3. Text: The main body of the news article, providing detailed content.
4. Label: A binary classification indicating whether the news is fake (0) or real (1).

### Dataset Size:

The dataset contains 72,134 entries, ensuring robust model training. It exhibits a balanced distribution between real and fake news instances, further enhancing the efficacy of machine learning training.

### Preprocessing Steps:

Given the structure of the dataset, several preprocessing steps will be undertaken to prepare the data for ML model training:

- Text Cleaning: Removal of unnecessary characters, punctuation, and special symbols.
- Tokenization: Breaking down the text into individual words or sub-words.
- Vectorization: Converting text into numerical vectors for ML model input.
- Handling Missing Data: Addressing any potential gaps or inconsistencies in the dataset.

---

## Evaluation Metrics:

1. **Accuracy**:
   - Definition: The ratio of correctly predicted instances to the total instances.
   - Justification: Provides an overall assessment of the model's correctness but may not be sufficient for imbalanced datasets.

2. **Precision**:
   - Definition: The ratio of true positive predictions to the total positive predictions.
   - Justification: Focuses on the accuracy of positive predictions, essential for scenarios where false positives are costly.

3. **Recall (Sensitivity)**:
   - Definition: The ratio of true positive predictions to the total actual positive instances.
   - Justification: Emphasizes the model's ability to capture all positive instances, crucial for scenarios where false negatives are detrimental.

4. **F1-Score**:
   - Definition: The harmonic mean of precision and recall, balancing both metrics.
   - Justification: Offers a balanced assessment, particularly valuable when there is an imbalance between real and fake news instances.

---

## Expected Results:

### Project Goals:

1. Develop and deploy a machine learning model for fake news prediction using the "WELFake" dataset.
2. Achieve high accuracy, precision, recall, and F1-score in model predictions.
3. Deploy the model via Flask to create a user-friendly web application for real-time fake news prediction.

---

## Future Implications:

The success of the project sets the stage for future enhancements in fake news detection models, adaptation to evolving challenges, and potential contributions to ongoing research in the field.

---

## References:

- WELFake Dataset. (2021) [IEEE Xplore](https://ieeexplore.ieee.org/document/9395133)
- WELFake on Kaggle. [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)
- Flask Documentation. (2023)
- TensorFlow Documentation. (2023)
- Scikit-learn Documentation. (2023)
- NLTK Documentation. (2023)

---




### Prerequisites:

1. Python installed on your system.
2. Required Python packages installed. You can install them using `pip`. Run the following command in your terminal:

   ```
   pip install scikit-learn tensorflow keras nltk spacy flask
   ```

3. Clone the project repository from GitHub:

   ```
   git clone 'https://github.com/Mostafa10770/Fake-News-Detection'
   ```

4. Navigate to the project directory:

   ```
   cd NEWS-DETECTION-MACHINE-LEARNING
   ```

### Running the Project:

1. **Data Preparation**:
   - Ensure you have downloaded the "WELFake" dataset and placed it in the appropriate directory (`data/` or as specified in the project).
   - Preprocess the dataset using the provided scripts or Jupyter notebooks if available.

2. **Training the Model**:
   - Run the Python script or notebook for training the machine learning model. This script will load the dataset, preprocess it, train the model, and save the trained model weights.
   - Example command:
   
     ```
     python train_model.py
     ```

3. **Running the Flask Web Application**:
   - Once the model is trained, start the Flask web application to deploy the model for real-time fake news prediction.
   - Run the Flask app using the following command:
   
     ```
     python Fake_News_Det.py
     ```
