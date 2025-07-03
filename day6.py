# import necessary libs
import nltk # natural languate processing toolkit
from nltk.corpus import movie_reviews # contains labeled movie reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

# download the nltk data files - only needed to be run once!
# nltk.download('movie_reviews')
# nltk.download('punkt') # split sentences into words
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# pre-process the dataset and extract the features
def extract_features(words):
    return {word: True for word in words}

    # ['good', 'movie'] becomes {'good':True, 'movie':True}


# Load the movie_reviews dataset from nltk
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)
             ]


# shuffle the dataset to ensure random distribution
random.shuffle(documents)

# prepare the dataset for training and testing
featuresets = [(extract_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]

# training the naive bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# evaluate the classifier on the test set
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# show the most informative features
classifier.show_most_informative_features(10)

# test on the new input sentences
def analyze_sentiment(text):
    # tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]

    # predict sentiment
    features = extract_features(words)
    return classifier.classify(features)


# test the classifier with some custom text inputs
# test_sentences = [
#     "This is an absolutely  fantastic movei! The acting was amazing!",
#     "I hated the movie, it was a waste of time and money.",
#     "The plot was a bit dull, but the performances were great.",
#     "I have mixed feelings about this film. It was okay, not great but not terrible."
# ]


# for sentence in test_sentences:
#     print(f"Sentence: {sentence}")
#     print(f"Predicted sentiment: {analyze_sentiment(sentence)}")
#     print()

print("What did you think of the Harry Potter movie? ")
my_review = input()

sent = analyze_sentiment(my_review)
print(f"Predicted sentiment: {sent}")

print(f"This will be recorded as a {'Good 🙂' if sent == 'pos' else 'BAD ☹'}")