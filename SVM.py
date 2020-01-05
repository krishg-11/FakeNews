import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, multilabel_confusion_matrix

df = pd.read_json("data.json")

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100

#use Tokenizer to convert text to numbers
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, char_level=False, split=' ')
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = df['label']
print('Shape of label tensor:', Y.shape)

#split data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1)
print("X_train shape:", X_train.shape,"; Y_train shape:",Y_train.shape)
print("X_test shape:", X_test.shape,"; Y_test shape:",Y_test.shape)


#hyperparameters
kernel = 'linear' #rbf, linear, poly (choose polynomiald degree (int); default=3), sigmoid, or precomputed
probability = False #True or False

#make model
print("making model")
svclassifier = SVC(kernel=kernel, probability=probability)

#train model
print("training model")
svclassifier.fit(X_train, Y_train)

#test model
print("testing model")
Y_pred = svclassifier.predict(X_test)

print("\nConfusion matrix:\n", confusion_matrix(Y_test,Y_pred))
print('\n', classification_report(Y_test,Y_pred))
print("Accuracy:", accuracy_score(Y_test, Y_pred))
