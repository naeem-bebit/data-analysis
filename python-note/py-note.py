# Dictionary True
n = int(input().strip())
check = {True: "Not Weird", False: "Weird"}
print(check[n % 2 == 0 and (n in range(2, 6) or n > 20)])

#  convert a list of integers into a single integer
for i in range(1, (n + 1)):
    print(i, end="")

int("".join(map(str, range(1, (int(input())+1)))))

# ---- data modelling
df = pd.read_csv('YoutubeSpamMergedData.csv')
df_data = df[["CONTENT","CLASS"]]
# Features and Labels
df_x = df_data['CONTENT']
df_y = df_data.CLASS
# Extract Feature With CountVectorizer
corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
model_cv= clf.fit(X_train,y_train)
model_cv.score(X_test,y_test)
# joblib.dump(model_cv, 'model.pkl')
filename = 'model.sav'

pickle.dump(model_cv, open(filename, 'wb'))

pickle.dump(cv, open('cv', 'wb'))

import pickle
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
count_vect = pickle.load(open('cv', 'rb'))
# result = loaded_model.predict(count_vect.transform([data_to_be_predicted]))

# data = ['your video is awesome']
data = ['please click on the link']
vect = count_vect.transform(data).toarray()
my_prediction = loaded_model.predict(vect)
my_prediction
