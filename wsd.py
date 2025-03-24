import html.parser
import copy
import string
import sys
from typing import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


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


# Only care about specific tags
def valid_tag(tag: str):
    match(tag):
        case 'answer' | 'instance' | 's' | 'head' | 'context':
            return True

    return False


def main():
    if len(sys.argv) < 4:
        print("Invalid number of arguments!")
        exit(1)

    # Load file data
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    model_filename = sys.argv[3]
    train_file = open(train_filename, 'r')
    test_file = open(test_filename, 'r')
    model_file = open(model_filename, 'w')

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
                case 'instance':
                    creatingSentence = False
                    filtered_sentence = sentence.sentence.translate(
                            str.maketrans('', '', string.punctuation))
                    sentence.sentence = filtered_sentence
                    for word in filtered_sentence.split(' '):
                        words[word] = True

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
                        sentence.answer = tag
                    elif insideS:
                        sentence.sentence += tag
                    elif insideHead:
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

    x_train = []
    x_test = []
    y_train = []
    for sentence in sentences:
        x_train.append(sentence.sentence)
        y_train.append(sentence.answer)

    for sentence in test_sentence_data:
        x_test.append(sentence)

    vectorizer = TfidfVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    clf = MultinomialNB()
    clf.fit(x_train_vectorized, y_train)

    predictions = clf.predict(x_test_vectorized)
    for prediction in predictions:
        print(prediction)


if __name__ == "__main__":
    main()
