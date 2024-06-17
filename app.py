from model import TextClassifier
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
classifier = TextClassifier()
classifier.load_and_preprocess_data('data/R8.txt', 'data/R8_labels.txt')
classifier.train()
classifier.evaluate()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['message']
    nb_result, svm_result = classifier.classify_text(user_message)
    response = nb_result if nb_result == svm_result else f"{nb_result}/{svm_result}"
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
