import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from neuralNetwork import NeuralNetwork
digits = load_digits()
X = digits.data
y = digits.target

# preprocessing
X -= X.min()
X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
y_train = LabelBinarizer().fit_transform(y_train)


nn = NeuralNetwork([64,100,10],'logistic')
nn.fit(X_train,y_train,epochs=3000)

predictions = nn.predict(X_test)
predictions = np.array(predictions)

print (confusion_matrix(y_test, predictions))
print (classification_report(y_test, predictions))



