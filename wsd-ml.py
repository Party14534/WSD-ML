"""
Zachariah Dellimore V00980652

Using scikit learn I was able to improve upon my decision list classifier
accuracy by 10-15% and the models performed much better than the most frequent
sense baseline which was 52.4%

Models used
Gaussian Naive Bayes
    Accuracy: 87.3015873015873%
    Matrix:
              product | phone |
            -------------------
    product |   50   |   12  |
            -------------------
    phone   |   4   |   60  |

    It performed the worst of the models used but still performed much better
    than my decision list classifier and much better than the most frequent
    sense baseline.

Logisitic Regression
    Accuracy: 91.26984126984127%
    Matrix:
              product | phone |
            -------------------
    product |   53   |   10  |
            -------------------
    phone   |   1   |   62  |

    Logistic Regression tied the MultiLayer Perceptron in performance but
    also computed much faster. An accuracy of 91% is much higher than
    my decision list classifier and is much better than the most frequent sense
    baseline.

MultiLayer Perceptron
    Accuracy: 91.26984126984127%
    Matrix:
              product | phone |
            -------------------
    product |   53   |   10  |
            -------------------
    phone   |   1   |   62  |

    The MultiLayer Perceptron tied the Logistic Regression model in accuracy
    but took longer to compute. The accuracy of the model is much higher than
    my decision list classifier and performed better than the most frequent
    sense baseline.

"""

import html.parser
import copy
import string
import sys
from typing import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Decision:
    feature = ""
    answer = ""
    liklihood = 0.0


# Sentence structure to save important data
class Sentence:
    sentence = ""
    word = ""
    answer = ""


# Parser implementation
class MyParser(html.parser.HTMLParser):
    values = []

    def handle_starttag(self, tag, attrs):
        if valid_tag(tag):
            if tag == 'answer':
                self.values.append(tag)
                self.values.append(attrs[1][1])
            else:
                self.values.append(tag)

    def handle_endtag(self, tag):
        self.values.append(tag)

    def handle_data(self, data):
        self.values.append(data)


sentences = []
features = OrderedDict()
feature_count = {}
words = {}
answer_count = {}
num_answers = 0
model_type = "NaiveBayes"


# Only care about specific tags
def valid_tag(tag: str):
    match(tag):
        case 'answer' | 'instance' | 's' | 'head' | 'context':
            return True

    return False


def main():
    global model_type

    # Get the model to use
    if len(sys.argv) < 3:
        print("Invalid number of arguments!")
        exit(1)
    elif len(sys.argv) == 4:
        match(sys.argv[3]):
            case "NaiveBayes" | "LogisticRegression" | "MLP":
                model_type = sys.argv[3]
            case _:
                print("Invalid model name")
                exit(1)

    # Load file data
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    train_file = open(train_filename, 'r')
    test_file = open(test_filename, 'r')

    train_data = train_file.read()
    test_data = test_file.read()

    # Parse data
    parser = MyParser()
    parser.feed(train_data)

    # Convert the parser data to sentences
    global words
    sentence = Sentence()
    creatingSentence = False
    insideHead = False
    insideContext = False
    insideS = False
    insideAnswer = False
    for tag in parser.values:
        if not creatingSentence:
            if tag == 'instance':
                creatingSentence = True
        else:
            match(tag):
                # The cases are used to tell which tags we are currently
                # inside to know where each string should go
                case 'instance':
                    # Instance tags surround data
                    creatingSentence = False

                    # Filter the sentence to remove unnecessary punctuation
                    # for disambiguation
                    filtered_sentence = sentence.sentence.translate(
                            str.maketrans('', '', string.punctuation))
                    sentence.sentence = filtered_sentence
                    for word in filtered_sentence.split(' '):
                        words[word] = True

                    # append sentence to list of sentences
                    sentences.append(copy.deepcopy(sentence))
                    sentence.sentence = ""
                    sentence.word = ""
                    sentence.answer = ""
                case 'context':
                    insideContext = not insideContext
                case 's':
                    insideS = not insideS
                case 'head':
                    insideHead = not insideHead
                    insideS = not insideS
                case 'answer':
                    insideAnswer = not insideAnswer
                case _:
                    if insideAnswer:
                        # This is where we get the feature
                        sentence.answer = tag
                    elif insideS:
                        # This is where we get the sentence data
                        sentence.sentence += tag
                    elif insideHead:
                        # This is where we get the word to be disambiguated
                        sentence.word = tag
                        sentence.sentence += tag

    # Create test sentences
    test_parser = MyParser()
    test_parser.values = []
    test_parser.feed(test_data)

    # Parse the test data and convert it to sentences
    test_sentences = []
    test_sentence = Sentence()
    test_sentence.sentence = ""
    test_sentence.word = ""
    test_sentence.answer = ""

    # This is the same as parsing the training data above
    for tag in test_parser.values:
        if not creatingSentence:
            if tag == 'instance':
                creatingSentence = True
        else:
            match(tag):
                case 'instance':
                    creatingSentence = False
                    test_sentences.append(copy.deepcopy(test_sentence))
                    test_sentence.sentence = ""
                    test_sentence.word = ""
                    test_sentence.answer = ""
                case 'context':
                    insideContext = not insideContext
                case 's':
                    insideS = not insideS
                case 'head':
                    insideHead = not insideHead
                    insideS = not insideS
                case 'answer':
                    insideAnswer = not insideAnswer
                case _:
                    if insideAnswer:
                        test_sentence.answer = tag
                    elif insideS:
                        test_sentence.sentence += tag
                    elif insideHead:
                        test_sentence.word = tag
                        test_sentence.sentence += tag

    # Only care about the strings
    test_sentence_data = []
    for sentence in test_sentences:
        test_sentence_data.append(sentence.sentence)

    # Format the data for the ML models
    x_train = []
    x_test = []
    y_train = []
    for sentence in sentences:
        x_train.append(sentence.sentence)
        y_train.append(sentence.answer)

    for sentence in test_sentence_data:
        x_test.append(sentence)

    # Use a vectorizer so the models can use the data
    vectorizer = TfidfVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    match(model_type):
        case "NaiveBayes":
            nb = GaussianNB()

            # The gaussian naive bayes model needs dense data and without
            # using toarray() the data would be sparse
            # It shows that there is an error in my coding environment but
            # the code runs without issues
            x_train_vectorized = x_train_vectorized.toarray()
            x_test_vectorized = x_test_vectorized.toarray()
            nb.fit(x_train_vectorized, y_train)

            predictions = nb.predict(x_test_vectorized)
            for prediction in predictions:
                print(prediction)

        case "LogisticRegression":
            lr = LogisticRegression()
            lr.fit(x_train_vectorized, y_train)

            predictions = lr.predict(x_test_vectorized)
            for prediction in predictions:
                print(prediction)

        case "MLP":
            mlp = MLPClassifier()
            mlp.fit(x_train_vectorized, y_train)

            predictions = mlp.predict(x_test_vectorized)
            for prediction in predictions:
                print(prediction)


if __name__ == "__main__":
    main()
