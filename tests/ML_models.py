from sklearn import feature_extraction
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import load_data
#print(X.shape)
def Normalize(A,rl=0,rr=10000):
    min_num = A.min()
    max_num = A.max()
    B=(A-min_num)/(max_num-min_num)*(rr-rl)
    return B
#X=Normalize(X)
#print(X[0])

def MLModel(X_test, X_train, y_test, y_train):
    print("Testbenching a linear classifier...")
    clf=LinearSVC()#svm
    #clf=MultinomialNB()
    #clf = SGDClassifier()#线性回归
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)

if __name__ == "__main__":
    #读入数据
    #x,y=load_data.get_wordVec2d_data()
    x,y=load_data.get_tfidf_data()
    #切分成训练集和测试集
    X_test, X_train, y_test, y_train=load_data.shuffle_split_data(x,y)
    #ML模型训练
    MLModel(X_test, X_train, y_test, y_train)
