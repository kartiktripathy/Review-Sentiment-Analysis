'''Review Sentiment Analysis using Natural Language Processing '''

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
'''this downloads and imports the stopwords that we will use to remove from the texts,which include the,a,is etc'''
from nltk.stem.porter import PorterStemmer
'''this will help us to take only the root of the word which indicates enough about the meaning eg loved->love
if we do not do this , there would be a separate feature generated for loved and love ..though they mean the same thing'''
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    ''' '[^a-zA-Z]' means which is not a-z or A-Z '''
    '''in the second qoute we put what we have replace the contents of the first qoute with'''
    '''here we are removing everything which is not a-z or A-Z by a space , sub function helps us to do that'''
    review = review.lower() #to convert everything to lowercase
    review = review.split() #to split the words of a particular review as elements of a list
    ps = PorterStemmer() #creating an object of the Porter Stemmer Class
    all_stopwords = stopwords.words('english')#storing all the stopwords of eng in a variable
    all_stopwords.remove('not')#removing 'not' from the list of stopwords ...so that they are not removed from the reviews
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    '''Here we rermove all the stopwords from the particular review and ad stemming to it by using a for loop 
    in each review that will run word by word'''
    review = ' '.join(review)# we join all the stemmed words with a space between them
    corpus.append(review) # add the cleaned review to the corpus list

#Creating the Bag OF Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
'''max features represents the size of the vocabulary in bag of words , it removes more unnecessary words'''
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,-1].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Training the model on the Training Set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test Set Results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix :")
cm = confusion_matrix(y_test,y_pred)
print(cm)

#Applying k-fold cross validation score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
'''cv = no of folds u want, 10 is the classic value'''
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))# mean avg of all the accuracies in all the folds
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100)) #Variance among the accuracies in the folds


#Predicting if a single review is positive or negative and user interface
new_review = input("Please provide a review : ")
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
if(new_y_pred==1):
    print("That's a positive review . Thank You !")
else:
    print("Thats a negative review . Sorry for the inconvenience !!")
