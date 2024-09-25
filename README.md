# **Is-It-AI**

This project aims to develop a machine learning system capable of differentiating between AI-generated and human-written essays. The solution leverages various machine learning models, including traditional classifiers like Logistic Regression and Random Forest, as well as advanced models like BERT. The models are integrated into a Django web application, allowing users to input text and receive predictions on whether the text is AI-generated or human-written.

## **Features**
- **Multi-Model Classification**: Uses a variety of models including Logistic Regression, SVC, Random Forest, XGBoost, and BERT.
- **Ensemble Voting**: Combines the predictions from all models to give a more robust classification result.
- **Django Integration**: Provides a user-friendly web interface where users can input text and get predictions.
- **Real-Time Predictions**: Text inputs are vectorized and processed by the models to return instant predictions.
- **AI Percentage**: The application provides a percentage showing how likely the text is to be AI-generated based on the collective predictions from all models.

## **Data**
All the data used for training the models can be accessed through [this link](https://finkiukim-my.sharepoint.com/:f:/g/personal/evgenija_jankulovska_students_finki_ukim_mk/EtAqlAeB5rxIglx1bF3_9ksBfs_RJI50qw4lrEaTO1PsKQ?e=WN1XW7).

## **Documentation**
For further details on the project and its structure, please refer to the [documentation here](https://docs.google.com/document/d/1LxY7EqX4QYmuh4MiiZ37hdKAlcFBpOo2R5Xz2y84C4g/edit?usp=sharing).

## **Models Used**
- **Logistic Regression**: A linear classifier effective for binary classification.
- **Support Vector Classifier (SVC)**: Identifies hyperplanes that best separate AI-generated and human-written texts.
- **Random Forest Classifier**: An ensemble model that reduces variance and improves predictions.
- **XGBoost**: Gradient-boosted decision trees, particularly effective for handling complex datasets.
- **Multinomial Naive Bayes**: A simple, probabilistic classifier suitable for text classification.
- **K-Nearest Neighbors (KNN)**: Classifies data points based on the closest features in vector space.
- **MLP Classifier**: A neural network classifier capable of handling non-linear relationships in the data.
- **BERT**: A transformer-based model that fine-tunes pre-trained embeddings to capture contextual relationships in text.
- **VotingClassifier**: Aggregates predictions from all models to form a consensus, leading to better overall performance.

## **Project Setup**

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Evgenija-J/Is-It-AI.git
   ```
2. Install the necessary Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Django Application**
1. Run migrations for the Django backend:
   ```bash
   python manage.py migrate
   ```
2. Start the Django server:
   ```bash
   python manage.py runserver
   ```
3. Visit `http://127.0.0.1:8000/` in your browser to interact with the web application.

### **Model Prediction**
You can enter text in the form provided on the website. The system will then process the text through multiple models and return:
- Predictions from each model (AI-generated or human-written).
- An overall AI percentage, representing the likelihood of the text being AI-generated based on all models.

## **File Structure**
- **/savedModels/**: Contains pre-trained models saved as `.joblib` files.
- **/data/**: Contains datasets used for training and testing the models.
- **/templates/**: Contains the HTML templates for rendering the web pages.
- **/views.py**: Handles the logic for loading models, processing text, and rendering predictions in the web interface.
- **/static/**: Stores static assets like CSS and JS files.

## **Future Improvements**
- Expanding the dataset to include articles, creative writing, and other types of text.
- Adding more advanced features like syntactic and grammatical analysis to further improve model performance.
- Implementing real-time model updates to continuously improve accuracy as new data becomes available.
