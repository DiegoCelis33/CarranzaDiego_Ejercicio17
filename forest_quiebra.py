#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
from scipy.io import arff


# In[169]:


data1 = arff.loadarff('1year.arff')
data2 = arff.loadarff('2year.arff')
data3 = arff.loadarff('3year.arff')
data4 = arff.loadarff('4year.arff')
data5 = arff.loadarff('5year.arff')


data1 = pd.DataFrame(data1[0])
data2 = pd.DataFrame(data2[0])
data3 = pd.DataFrame(data3[0])
data4 = pd.DataFrame(data4[0])
data5 = pd.DataFrame(data5[0])

#data = pd.concat([data1, data2,data3,data4,data5], axis=0)



data = pd.concat([data1, data2,data3,data4,data5])

sd = getattr(data, "class")
data['class']=sd.astype(int)

data = data.dropna()


predictors = list(data.keys())
predictors.remove('class')

#print(predictors, np.shape(np.array(predictors)))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    data[predictors], data['class'], test_size=0.5)

X_test, X_validation, y_test, y_validation = sklearn.model_selection.train_test_split(
                                    data[predictors], data['class'], test_size=0.2)

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_features='sqrt')

n_trees = np.arange(1,100,25)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), len(predictors)))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test)))
    feature_importance[i, :] = clf.feature_importances_

maximo = n_trees[np.argmax(f1_test)]




# In[158]:


#plt.scatter(n_trees, f1_test)


# In[186]:


feature_importance = np.zeros((maximo, len(predictors)))

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=maximo, max_features='sqrt')
clf.fit(X_validation, y_validation)
f1_validation = sklearn.metrics.f1_score(y_validation, clf.predict(X_validation))
feature_importance[i, :] = clf.feature_importances_
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index=predictors)
print(a)
plt.figure()
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.title('M='+str(maximo))
plt.savefig("features.png")


# In[171]:


f1_validation 


# In[ ]:




