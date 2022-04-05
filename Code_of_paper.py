#!/usr/bin/env python
# coding: utf-8

# #  Cuprate superconducting materials above liquid nitrogen temperature from machine learning

# ## Exploratory analysis of data

# In[2]:


import os
import sys
import time
import matplotlib
import pathlib
import graphviz
import itertools
import pycaret
import pyforest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
#from sklearn import *
import seaborn as sns
from sklearn import tree 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
                }
def plot_roc_curve(fprs,tprs):
    plt.figure(figsize=(8,6),dpi=80)
    plt.plot(fprs,tprs)
    plt.plot([0,1],linestyle='--')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
                }
    plt.ylabel('TP rate',font2)
    plt.xlabel('FP rate',font2)
    plt.title('ROC',font2)
    plt.savefig('ROC.jpg', dpi=300) 
    plt.show()

def plot_cnf_matirx(cnf_matrix,description):
    class_names = [0,1]
    fig = plt.gcf( )
    fig.set_size_inches(15.5, 10.5)
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False  
    fig,ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'OrRd',fmt = 'g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title(description, y = 1.1,fontsize=16)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
                }
    plt.ylabel('True0/1',font2)
    plt.xlabel('Pred0/1',font2)
    fig = plt.gcf( )
    fig.set_size_inches(5.5, 4.5)
    plt.savefig('cnf_matirx.jpg', dpi=300)
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[3]:


data = pd.read_csv("./data.csv")
S1 = pd.read_csv("./S1.csv")
S2 = pd.read_csv("./S2.csv")
df =S1


# In[4]:


data.describe()


# In[5]:


df


# In[8]:


distribution = []
elem = []
for i  in  df.columns[2:]  :
    ca = (df[i]==0).sum()
    realele = 12340-ca
    distribution.append(realele)
    elem.append(i)
fengfu = pd.DataFrame(distribution)
elem = pd.DataFrame(elem)
elem["elem"]=elem
elem["distribution"]=distribution
x = elem["elem"]
y = elem["distribution"]
#seaborn.barplot(data=fengfu)
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False     
plt.figure(figsize=(40,20))
plt.bar(x,y)
plt.savefig('distribution.jpg', dpi=300) 
plt.show()


# In[9]:


#Superconducting transition temperature frequency diagram of all materials
fig = plt.gcf( )
fig.set_size_inches(15.5, 10.5)
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False     
plt.hist(data["Tc"], bins=100, color=None,edgecolor="green", alpha=0.8)
plt.title('Superconducting transition temperature frequency diagram')
plt.xlabel("Tc(K)")
plt.ylabel("Count number")
plt.show()


# In[10]:


#Superconducting transition temperature frequency diagram of materiasl which Tc > 77K
fig = plt.gcf( )
fig.set_size_inches(15.5, 10.5)
dataTc = data[data["Tc"]>=77]
dataTc
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False    
#plt.hist(dataTc, bins=100,log=False, color=None)
plt.hist(dataTc["Tc"], bins=100, normed=0,  edgecolor="green", alpha=0.8)
plt.xlabel("Tc(K)")
plt.ylabel("Count number")
plt.title('Superconduct transition temperature count diagram')


# ## Descriptors populate the data overview

# ### The descriptor is generated by AFS1 and AFS2 software

# In[11]:


S1 = pd.read_csv("./S1.csv")
S2 = pd.read_csv("./S2.csv")


# In[12]:


S1


# In[13]:


S2


# ## Classification model training

# ### Use pycaret to make a rough assessment of multimodels

# In[14]:


CLS1 = pd.read_csv("./CLS1.csv")
CLS2 = pd.read_csv("./CLS2.csv")
dataset = CLS1
dataset 


# In[15]:


data = dataset.sample(frac=0.8, random_state=2020).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'Tc', session_id=123)


# In[ ]:


compare_models()    ##Run to get model ranking


# In[ ]:


dataset = CLS2
data = dataset.sample(frac=0.8, random_state=2020).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'Tc', session_id=123)
compare_models()


# ### Select the appropriate model for classification (RFC,KNN,DT,ADC)

# ###  RFC

# In[17]:


# RFC
data = CLS1
X = data.iloc[:,2:].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train,y_train)
print("RFCscore",rfc.score(X_test, y_test.astype('int')))
cv = ShuffleSplit(n_splits=10, test_size=.2)
scores = cross_val_score(rfc, X, y, cv=cv)
print(scores)
print(scores.mean())
ll = np.linspace(0,9,10)
ll_x = ll
ll_y = scores
plt.plot(ll_x,ll_y)
plt.show( )


# ###  KNN

# In[18]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_search = [
    {
        "weights":["uniform"],
        "n_neighbors":[i for i in range(1,11)]
    },
    {
        "weights":["distance"],
        "n_neighbors":[i for i in range(1,11)],
        "p":[i for i in range(1,6)]
    }
]
knn_clf = KNeighborsClassifier()


# In[19]:


grid_search = GridSearchCV(knn_clf, param_search)


# In[21]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\ngrid_search.best_estimator_')


# In[22]:


# KNN
X = data.iloc[ : ,2:].values
y = data.iloc[ : ,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
from sklearn.neighbors import KNeighborsClassifier
reg = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=6, p=2,
                     weights='distance')
reg.fit(X_train, y_train.astype('int'))
print(reg.score(X_test, y_test.astype('int')))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=20, test_size=.3)
scores = cross_val_score(reg, X, y, cv=cv)
print(scores)
print(scores.mean())
ll = np.linspace(0,9,20)
ll_x = ll
ll_y = scores
plt.plot(ll_x,ll_y)
plt.show( )


# ###  DT

# In[24]:


# DT
data = CLS1
X = data.iloc[:,2:].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)
data


# In[25]:


clf = tree.DecisionTreeClassifier(max_depth=7,max_features=30,criterion="entropy") #初始化树模型
clf = clf.fit(X, y)                          #实例化训练集
score = clf.score(X, y)                        #返回预测的准确度
print("DT",score)
cv = ShuffleSplit(n_splits=10, test_size=.2)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)
print(scores.mean())
ll = np.linspace(0,9,10)
ll_x = ll
ll_y = scores
plt.plot(ll_x,ll_y,"--")
plt.show( )


# #### Visualization of decision trees

# In[26]:


feature_name = data.iloc[:,2:].columns
import graphviz
dot_data = tree.export_graphviz(clf
                                ,out_file=None
                                ,feature_names= feature_name 
                                ,class_names=["Tc>77K","Tc<77K"]
                                ,filled=True
                                ,rounded=True
                               )
graph = graphviz.Source(dot_data)
graph


# In[ ]:


# graph.render(filename="tree")   ##save DT graph as a pdf


# ###  ADC

# In[27]:


# Ada
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators = 900)
data = CLS1
X = data.iloc[:,2:].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
abc.fit(X_train,y_train)
print("ABCscore",abc.score(X_test, y_test.astype('int')))
cv = ShuffleSplit(n_splits=10, test_size=.2)
scores = cross_val_score(abc, X, y, cv=cv)
print(scores)
print(scores.mean())
ll = np.linspace(0,9,10)
ll_x = ll
ll_y = scores
plt.plot(ll_x,ll_y)
plt.show( )


# ### RFC : Analysis of multiple achievement indicators

# In[28]:


# RFC
data = CLS1
X = data.iloc[:,2:].values
y = data.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)


# In[29]:


[*zip(feature_name,clf.feature_importances_)]


# In[30]:


clf = rfc
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos[-10:], feature_importance[sorted_idx[-10:]], align='center')
plt.yticks(pos[-10:], data.iloc[:,2:].columns[sorted_idx[-10:]],fontsize=20)
plt.xlabel('Relative Importance',font2)
plt.title('Variable Importance',font2)
fig = plt.gcf( )
fig.set_size_inches(30,10)
#plt.savefig('Variable Importance',dpi=300,bbox_inches="tight")
plt.show()


# #### F1 score

# In[31]:


from sklearn.metrics import f1_score
y_pred = rfc.predict(X_test)
f1_score(y_test,y_pred)


# ####  confusion matrix 

# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[33]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
cnf_matrix


# In[34]:


plot_cnf_matirx(cnf_matrix,'Confusion matrix -- RFC')


# #### Accuracy and call-back curves

# In[35]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
rf_clf = rfc
y_probabilities_rf = rf_clf.predict_proba(X_test)[:,1]

roc_auc_score(y_test,y_probabilities_rf)


# In[36]:


from sklearn.metrics import roc_curve
fprs4,tprs4,thresholds4 = roc_curve(y_test,y_probabilities_rf)
fig = plt.gcf( )
fig.set_size_inches(15.5, 10.5)

plot_roc_curve(fprs4,tprs4)

#plt.savefig('ROC.jpg', dpi=300) 


# #### The threshold of model

# In[37]:


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

thresholds = [0.4,0.45,0.47,0.49,0.5,0.51,0.53,0.55,0.56,0.57,0.58,0.60]
plt.figure(figsize=(10,10))
m = 1
for i in thresholds:
    y_test_predictions_high_recall = y_probabilities_rf > i
    
    plt.subplot(4,4,m)
    m += 1
    
    cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

 #  print (i,"Recall:{}".format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
 #   acc = rfc.score(X_test,y_test)
 #   print(acc)
   
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i)
 #  plt.savefig("before", dpi=300)


# #### Cost sensitive learning

# #####  Bayes formula：P(A|B)=P(B|A)*P(A)/P(B)

# In[38]:


len(data[data["Tc"]==0])


# In[39]:


len(data[data["Tc"]==1])


# In[43]:


log = []
for i in range(100,900,5):
#    print(i/1000)
    log.append(i/1000)


# In[48]:


mg = []
thresholds = log
m = 1
for i in thresholds:
    y_test_predictions_high_recall = y_probabilities_rf > i    
    cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
#    print(cnf_matrix)
    C01 = cnf_matrix[0][1]
    C10 = cnf_matrix[1][0]
    C00 = cnf_matrix[0][0]
    C11 = cnf_matrix[1][1]
    C =C01+C00+C10+C11
    cost = (1*(C01)/((C11+C10)))+((1.5*C10)/((C11+C01)))
#    cost = 1
#    recall =cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
#    acc = (C00+C11)/C
#    COST = (1/cost)*recall*acc
    COST=cost
#    print(COST)
    mg.append(COST)


# In[49]:


mingan = np.array(mingan)
log = np.array(log)
plt.plot(log,mg)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 20,
}
plt.ylabel('cost',font2)
plt.xlabel(' threshold value',font2)
fig = plt.gcf( )
fig.set_size_inches(8, 6)
#plt.figure(figsize=(8,6))
plt.savefig('cost.jpg', dpi=300)
plt.show( )


# In[50]:


thresholds = [0.350,0.355,0.360,0.365,0.370,0.375,0.380,0.385]
plt.figure(figsize=(10,10))

m = 1
for i in thresholds:
    y_test_predictions_high_recall = y_probabilities_rf > i
    
    plt.subplot(4,4,m)
    m += 1
    
    cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

#    print (i,"Recall:{}".format(cnf_matrix[0,0]/(cnf_matrix[0,1]+cnf_matrix[0,0])))
 #   acc = rfc.score(X_test,y_test)
 #   print(acc)
   
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i)
#    plt.savefig("laser", dpi=300)
    


# ## Train the deep neural network

# ### Use S1, S2 descriptors for training

# In[51]:


df = S1


# In[73]:


## The basic framework of neural network
X = df.iloc[:,2:].values
Y = df.iloc[:,1].values
X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.25) 

sc = StandardScaler()
X_transform = sc.fit_transform(X_train)
sc2 = StandardScaler()
# y scale to [0,1]

mms = MinMaxScaler()
Y_transform = mms.fit_transform(y_train.reshape(-1, 1) )

class ProgressBar(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    self.draw_progress_bar(epoch + 1, EPOCHS)

  def draw_progress_bar(self, cur, total, bar_len=50):
    cur_len = int(cur / total * bar_len)
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('epoch')
  plt.ylabel('metric - MSE')
  plt.plot(hist['epoch'], hist['mse'], label='tr')
  plt.plot(hist['epoch'], hist['val_mse'], label = 'va')
  plt.grid(True)
  plt.legend()
  
  plt.figure()
  plt.xlabel('epoch')
  plt.ylabel('metric - MAE')
  plt.plot(hist['epoch'], hist['mae'], label='tr')
  plt.plot(hist['epoch'], hist['val_mae'], label = 'va')
  plt.grid(True)
  plt.legend()
    
logdir = '.\callbacks'
output_model_file = os.path.join(logdir,"study.h5")
tensorboard =  keras.callbacks.TensorBoard(logdir)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-5,patience=50)

callbacks=[ProgressBar()
           ,early_stop]

EPOCHS = 50000

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(X.shape[1],)) ) 

model.add(tf.keras.layers.Dense(128, activation="relu") )
model.add(tf.keras.layers.Dense(256, activation="relu") )
model.add(tf.keras.layers.Dense(128, activation="relu") )
model.add(tf.keras.layers.Dense(128, activation="relu") )
model.add(tf.keras.layers.Dense(1, activation="relu") ) 
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001)
             ,loss = 'mse' 
             ,metrics=[['mae', 'mse']]
             )


# In[74]:


model.summary()


# In[75]:


history = model.fit(
  X_transform,Y_transform,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  batch_size= 32, validation_freq= 1 ,callbacks=callbacks)


# In[76]:


plot_history(history)


# In[77]:


y_pred = model.predict(sc.transform(X_test))
y_pred= mms.inverse_transform(y_pred)
y_test = y_test.reshape(-1,1)
mae = y_pred-y_test
print("mae",np.abs(mae).reshape(1,-1).mean())

from matplotlib import pyplot as plt
plt.scatter(y_test,y_pred)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
print("R2",r2_score(y_test,y_pred))
plt.title('DNN')
fig = plt.gcf( )
fig.set_size_inches(5.5, 10.5)

ll = np.linspace(0,130,1200)
ll_x = ll
ll_y = ll
plt.plot(ll_x,ll_y)
plt.show( )


# ###  Parameter optimization (using additional library optunaas for a longer time)
import pyforest
import warnings
warnings.filterwarnings("ignore")
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


df = pd.read_csv("S1.csv")

print(df.shape)

from sklearn.model_selection import train_test_split
X = df.iloc[:,2:].values
Y = df.iloc[:,1].values


X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_transform = sc.fit_transform(X_train)
sc2 = StandardScaler()
# y scale to [0,1]
from sklearn.preprocessing import MinMaxScaler
#mms = MinMaxScaler()
#Y_transform = mms.fit_transform(y_train.reshape(-1, 1) )
Y_transform = y_train

Y_transform.shape


import sys
class ProgressBar(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):

    self.draw_progress_bar(epoch + 1, EPOCHS)

  def draw_progress_bar(self, cur, total, bar_len=50):
    cur_len = int(cur / total * bar_len)
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=50)
callbacks=[ProgressBar() ,early_stop]

EPOCHS = 10000


def objective(trial):
    kwargs = {}
    jilu = []
    # Categorical parameter
    optimizer = trial.suggest_categorical('optimizer', ["RMSprop", "Adam", "SGD"])
    if optimizer == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    # Int parameter
    sj_n = [1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16]
    sj_n[0] = trial.suggest_int('sj_n[0]', 32, 256)
    sj_n[1] = trial.suggest_int('sj_n[1]', 32, 256)
    sj_n[2] = trial.suggest_int('sj_n[2]', 32, 256)
    sj_n[3] = trial.suggest_int('sj_n[3]', 32, 256)
    sj_n[4] = trial.suggest_int('sj_n[4]', 32, 256)
    sj_n[5] = trial.suggest_int('sj_n[5]', 32, 256)
    sj_n[6] = trial.suggest_int('sj_n[6]', 32, 256)
    sj_n[7] = trial.suggest_int('sj_n[7]', 32, 256)
    sj_n[8] = trial.suggest_int('sj_n[8]', 32, 256)
    sj_n[9] = trial.suggest_int('sj_n[9]', 32, 256)
    sj_n[10] = trial.suggest_int('sj_n[10]', 32, 256)    
    sj_n[11] = trial.suggest_int('sj_n[11]', 32, 256)    
    sj_n[12] = trial.suggest_int('sj_n[12]', 32, 256)    
    sj_n[13] = trial.suggest_int('sj_n[13]', 32, 256)   
    sj_n[14] = trial.suggest_int('sj_n[14]', 32, 256)   
    sj_n[15] = trial.suggest_int('sj_n[15]', 32, 256)     
    
    
    
    
    
    
    # Uniform parameter
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.9)

    # Loguniform parameter
    #    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    num_layers = trial.suggest_int('num_layers', 3, 15)
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(X.shape[1],)))
    test_n = [1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16]
    test_n[0] = trial.suggest_int('test_n[0]', 1, 2)
    test_n[1] = trial.suggest_int('test_n[1]', 1, 2)
    test_n[2] = trial.suggest_int('test_n[2]', 1, 2)
    test_n[3] = trial.suggest_int('test_n[3]', 1, 2)
    test_n[4] = trial.suggest_int('test_n[4]', 1, 2)
    test_n[5] = trial.suggest_int('test_n[5]', 1, 2)
    test_n[6] = trial.suggest_int('test_n[6]', 1, 2)
    test_n[7] = trial.suggest_int('test_n[7]', 1, 2)
    test_n[8] = trial.suggest_int('test_n[8]', 1, 2)
    test_n[9] = trial.suggest_int('test_n[9]', 1, 2)
    test_n[10] = trial.suggest_int('test_n[10]', 1, 2)
    test_n[11] = trial.suggest_int('test_n[11]', 1, 2)
    test_n[12] = trial.suggest_int('test_n[12]', 1, 2)
    test_n[13] = trial.suggest_int('test_n[13]', 1, 2)
    test_n[14] = trial.suggest_int('test_n[14]', 1, 2)
    test_n[15] = trial.suggest_int('test_n[15]', 1, 2)
    
    
    for i in range(num_layers):
        model.add(layers.Dense(sj_n[i], activation='relu'))
        if test_n[i] == 1:
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='relu'))

    model.compile(optimizer=optimizer, loss='mse', metrics=[['mae', 'mse']])
    history = model.fit(X_transform, Y_transform, epochs=EPOCHS, validation_split=0.2, verbose=0, batch_size=32,
                        validation_freq=1, callbacks=callbacks)
    y_pred = model.predict(sc.transform(X_test))
    global y_test

    y_test = y_test.reshape(-1, 1)

    mae = y_pred - y_test


    import numpy as np
    from sklearn.metrics import r2_score
    global r2
    r2 = r2_score(y_test, y_pred)

    MAE = np.abs(mae).reshape(1, -1).mean()
    import time
    
    ts = str(time.time())
    saved_model_path = "./saved999_models/{}".format(str(r2)) + "R2" +"|||||||MAE"+ str(
        np.abs(mae).reshape(1, -1).mean()) + "TM" + ts
    if r2 >= 0.91:
        tf.keras.experimental.export_saved_model(model, saved_model_path)
        model.save('optmodel.h5')
   
    print("R2",r2)
    print("mae",str(np.abs(mae).reshape(1, -1).mean()))
    return r2


import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10000)
print(study.best_params)
print(study.best_value)
# ###  The trained model, take DNN under S1 descriptor as an example

# In[78]:


save_model = tf.keras.experimental.load_from_saved_model("./S1model")
#save_model.predict(X).shape
save_model.summary()


# ### Residual analysis

# In[79]:


## take S2model for a example
model = tf.keras.experimental.load_from_saved_model("./S2model")
df =S2
from sklearn.model_selection import train_test_split
X = df.iloc[:,2:].values
Y = df.iloc[:,1].values
X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.1) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_transform = sc.fit_transform(X_train)
sc2 = StandardScaler()

from sklearn.preprocessing import MinMaxScaler

y_pred = model.predict(sc.transform(X_test))

from matplotlib import pyplot as plt
plt.scatter(y_test,y_pred)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
print(r2_score(y_test,y_pred))
from sklearn.metrics import mean_squared_error, mean_absolute_error


np.sqrt(mean_squared_error(y_test,y_pred))


print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))


mean_absolute_error(y_test,y_pred)

print("MAE",mean_absolute_error(y_test,y_pred))

font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 12,
                }
#plt.title('Deeplearning')
fig = plt.gcf( )
fig.set_size_inches(5.5, 10.5)
plt.xlabel("Test Tc(K)",font2)

plt.ylabel("Pred Tc(K)",font2)

ll = np.linspace(0,130,1200)
ll_x = ll
ll_y = ll
plt.plot(ll_x,ll_y)
#plt.savefig('DNN回归.jpg', dpi=300)
plt.show( )


# In[80]:


error = y_pred.T - y_test.T
error = pd.DataFrame(error)
fig = plt.figure(figsize = (10,6))
#ax1 = fig.add_subplot(2,1,1)  # 创建子图1
#ax1.scatter(s.index, s.values)
#plt.grid()
# 绘制数据分布图
font3 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 20,
                }
ax = fig.add_subplot(1,1,1)  # 创建子图2
#s.hist(bins=30,alpha = 0.5,ax = ax)
sns.distplot(error,ax = ax,bins = 50)
plt.legend(["Residual with S2"],fontsize =15)
plt.xlabel('Residual(K)',font3)
plt.ylabel('Density',font3)
ax.tick_params(axis='y',labelsize=15)
ax.tick_params(axis='x',labelsize=15)
#plt.title('Pred of VH')
#plt.savefig('误差分析.jpg', dpi=300)
plt.grid()
#plt.savefig("S2 residual", dpi=300,bbox_inches ="tight")


# ##  Virtual high throughput Settings

# ###  Virtual high-throughput sample generation (the following code is virtual sample generation for Ba distribution in HgCaBaPbCuO compounds)
### It should be noted that the sample distribution generated by virtual high flux conforms to the distribution characteristics of ###   training data as much as possible
vh = []
import numpy
import pandas as pd
x = numpy.arange(1,300,6)
x= x/1000
x = list(x)
print(x)
for i in x:
    print(i)
    vh = []
    data =[ {'Hg':[[0,0.26,0.01]]}, {'Pb':[[0.09, 0.34, 0.01]]}, {'Ba':[[i, i, i]]}, {'Ca':[[0, 0.2748, 0.048], [0.27, 1.152, 0.152]]}, {'Cu':[[0.1,0.6,0.01]]}, {'O':[[1,1,1]] }]
    def findall(pos,sum,outcome):
        if pos==len(data)-1:
            if 1/sum >=0.7 and 1/sum<=2.4:
                outcome+="O"
                print(outcome)
                vh.append(outcome)
        #    <----------------------------------output
        else:
                for (key,val) in data[pos].items():
                    for vl in val:
                        st = vl[0]
                        if st==0:
                            st=st+vl[2]
                        en = vl[1]
                        bu = vl[2]
                        while(st<=en):
                            findall(pos+1,sum+round(st,6), outcome+str(key)+str(round(st,6)))
                            st+=bu
    findall(0, 0,'')

    test=pd.DataFrame(vh)
    print(test)
    test.to_csv('./'+str(i)+'Ba.csv',encoding='gbk',index=0,header=0)
# ###  DNN prediction of virtual samples
see it in DNN-VH.ipynb
# ## Deep neural network combined with manifold learning

# In[83]:


import tensorflow as tf
saved_model_path = "path_to\modell"
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()
model = new_model
a = []
for i in model.get_weights():

    a.append(i[0])

from tensorflow.keras import Sequential
model2 = tf.keras.Sequential()
print(len(model.layers))
print(len(model.layers[:-1]))
for layer in model.layers[:-1]:
    model2.add(layer)

model2.summary()
d = []
for i in model2.get_weights():
    d.append(i[0])
print("-----------------")


# In[84]:


df = pd.read_csv("S1.csv")
df


# In[86]:


y_pred = model2.predict(sc.transform(X))
len(y_pred)


# In[87]:


y_pred = pd.DataFrame(y_pred)
newdf = pd.concat([df,y_pred],axis=1)
newdf


# ### PCA

# In[88]:


from sklearn.decomposition import PCA
ax = plt.axes(projection='3d')
x = y_pred
pca = PCA(n_components=3)
pca.fit(x)
x_pca = pca.transform(x)
newdf = np.array(newdf)
close = newdf[:,1]
volume = newdf[:,1]
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
#plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.2 ,c =close)
ax.scatter3D(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=close)
plt.show()
plt.show()


# In[121]:


font4 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 25,
                }
from sklearn.decomposition import PCA
#ax = plt.axes(projection='3d')
x = y_pred
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
newdf = np.array(newdf)
close = newdf[:,1]
volume = newdf[:,1]
plt.figure(figsize=(20,10))
plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.2 ,c =close,label="PCA")
#ax.scatter3D(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=close)
plt.xlabel('reduce x axis',font4)
plt.ylabel('reduce y axis',font4)
plt.legend(loc='upper right',fontsize= 25)
plt.xticks([])
plt.yticks([])
plt.savefig("reallPCA.png",dpi=300,bbox_inches="tight")
plt.show()

for i in range(2,120,1):
    newdf = np.array(newdf)
    close = newdf[:,i]
    print("elem Number",i-1)

    import seaborn as sns
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,5))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.2 ,c =close )
    plt.show()
# ### T-SNE

# In[127]:


from sklearn.manifold import TSNE
#ax = plt.axes(projection='3d')
x = y_pred
X_embedded = TSNE(n_components=2).fit_transform(x)
#X_embedded
plt.figure(figsize=(20,10))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.2 ,c =close,label = "t-SNE")
#ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=close)
plt.xlabel('reduce x axis',font4)
plt.ylabel('reduce y axis',font4)
plt.legend(loc='upper right',fontsize= 25)
plt.xticks([])
plt.yticks([])
plt.savefig("reallT-SNE.png",dpi=300,bbox_inches="tight")
plt.show()
plt.show()


# ### Isomap

# In[129]:


from sklearn import manifold
ax = plt.axes(projection='3d')
x = y_pred
X_embedded =  manifold.Isomap(n_components=3).fit_transform(x)
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.2 ,c =close,label="Isomap")
ax.scatter3D(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=close)
plt.xlabel('reduce x axis',font4)
plt.ylabel('reduce y axis',font4)
plt.legend(loc='upper right',fontsize= 25)
plt.xticks([])
plt.yticks([])
plt.savefig("reallIsomap.png",dpi=300,bbox_inches="tight")
plt.show()


# ### MDS

# In[130]:


from sklearn import manifold

x = y_pred
X_embedded =  manifold.MDS(n_components = 2, max_iter=100, n_init=1).fit_transform(x)
plt.figure(figsize=(20,10))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.2 ,c =close,label="MDS")
plt.xlabel('reduce x axis',font4)
plt.ylabel('reduce y axis',font4)
plt.legend(loc='upper right',fontsize= 25)
plt.xticks([])
plt.yticks([])
plt.savefig("reallMDS.png",dpi=300,bbox_inches="tight")
plt.show()


#  END
