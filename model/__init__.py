import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from sklearn import naive_bayes, svm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder


class TextClassifier:
    def __init__(self, stop_words: list = ENGLISH_STOP_WORDS):
        stop_words = list(stop_words) if not isinstance(stop_words, list) else stop_words
        self.stop_words = stop_words
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.label_encoder = LabelEncoder()
        self.nb_classifier = naive_bayes.MultinomialNB()
        self.svm_classifier = svm.SVC(probability=True)
    
    def clean_html(self, text):
        return BeautifulSoup(text, "html.parser").get_text()
    
    def preprocess_text(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return " ".join(filtered_tokens)
    
    def load_and_preprocess_data(self, texts_path, labels_path):
        texts = open(texts_path).read().strip().split("\n")
        labels = open(labels_path).read().strip().split("\n")
        
        train_texts, train_labels, test_texts, test_labels = [], [], [], []
        for i in range(len(labels)):
            label_info = labels[i].split("\t")
            if label_info[1] == 'train':
                train_texts.append(texts[i])
                train_labels.append(label_info[2])
            elif label_info[1] == 'test':
                test_texts.append(texts[i])
                test_labels.append(label_info[2])
        
        trainDF = pd.DataFrame({'text': train_texts, 'label': train_labels})
        testDF = pd.DataFrame({'text': test_texts, 'label': test_labels})
        
        trainDF['clean_text'] = trainDF['text'].apply(lambda x: self.clean_html(x)).apply(self.preprocess_text)
        testDF['clean_text'] = testDF['text'].apply(lambda x: self.clean_html(x)).apply(self.preprocess_text)
        
        self.x_train = self.tfidf_vectorizer.fit_transform(trainDF['clean_text'])
        self.x_test = self.tfidf_vectorizer.transform(testDF['clean_text'])
        
        self.train_labels = self.label_encoder.fit_transform(trainDF['label'])
        self.test_labels = self.label_encoder.transform(testDF['label'])
    
    def train(self):
        self.nb_classifier.fit(self.x_train, self.train_labels)
        self.svm_classifier.fit(self.x_train, self.train_labels)
    
    def evaluate(self):
        nb_predictions = self.nb_classifier.predict(self.x_test)
        svm_predictions = self.svm_classifier.predict(self.x_test)
        
        nb_probabilities = self.nb_classifier.predict_proba(self.x_test)
        svm_probabilities = self.svm_classifier.predict_proba(self.x_test)
        
        nb_auc_roc = roc_auc_score(self.test_labels, nb_probabilities, multi_class='ovr')
        svm_auc_roc = roc_auc_score(self.test_labels, svm_probabilities, multi_class='ovr')
        
        metrics = {
            'Naive Bayes': {
                'accuracy': accuracy_score(self.test_labels, nb_predictions),
                'precision': precision_score(self.test_labels, nb_predictions, average='weighted'),
                'recall': recall_score(self.test_labels, nb_predictions, average='weighted'),
                'f1': f1_score(self.test_labels, nb_predictions, average='weighted'),
                'confusion_matrix': confusion_matrix(self.test_labels, nb_predictions)
            },
            'SVM': {
                'accuracy': accuracy_score(self.test_labels, svm_predictions),
                'precision': precision_score(self.test_labels, svm_predictions, average='weighted'),
                'recall': recall_score(self.test_labels, svm_predictions, average='weighted'),
                'f1': f1_score(self.test_labels, svm_predictions, average='weighted'),
                'confusion_matrix': confusion_matrix(self.test_labels, svm_predictions)
            }
        }
    
    def classify_text(self, input_text):
        clean_text = self.preprocess_text(self.clean_html(input_text))
        input_vector = self.tfidf_vectorizer.transform([clean_text])
        
        nb_prediction = self.nb_classifier.predict(input_vector)
        svm_prediction = self.svm_classifier.predict(input_vector)
        
        nb_label = self.label_encoder.inverse_transform(nb_prediction)
        svm_label = self.label_encoder.inverse_transform(svm_prediction)
        
        return nb_label[0], svm_label[0]
