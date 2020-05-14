# Librerias
import numpy as nump 
import pandas as pand 
import seaborn as seans
import re
from sys import exit
from matplotlib import pyplot as plot
from matplotlib import style
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics  import roc_auc_score 


#Recogemos los ficheros
test_df = pand.read_csv("PracticaIndividual/test.csv")
train_df = pand.read_csv("PracticaIndividual/train.csv")

#Mostramos la información
#train_df.info()

#Describe las variables
#print(train_df.describe())

#Describe las variables
#print(train_df.head(20))

#Se realiza la suma y se ordenan los valores de forma descendente cuando son "Null"
total = train_df.isnull().sum().sort_values(ascending=False)

#Sacamos los porcentajes
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
#redondeamos el porcentaje que vamos a mostrar
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

#mostramos los valores que tienen campos con tipo Null en dos  columnas(La suma de Null por columna y % sobre el global de filas)
missing_data = pand.concat([total, percent_2], axis=1, keys=['Total', '%'])

#Muestra la suma de valores NULL y el porcentaje sobre el Total
#print(missing_data.head(11))

#nos muestra todas las cabeceras del fichero
#print(train_df.columns.values)

#dataNew = [train_df, test_df]
dataNew =  pand.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)


#Cambiar los valores de Sexo a Int
sex = {"male": 3, "female": 0}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(sex)
    
#Agregamos al DataSet el sexo por la clase del pasajero para diferenciarlos

data = [train_df, test_df]
for dataset in data:
    dataset['Sex_Class']= dataset['Sex']+ dataset['Pclass']

#print(dataset['Sex_Class'].describe())

#MOSTRAR LAS MEDIAS AGRUPANDO POR EL GRUPO SEXCLASS
#print(dataset.groupby('Sex_Class').mean())

#Mostrar la relación entre la Edad y el Sexo
#AS = seans.factorplot(y="Age", x="Sex_Class", data = dataset, kind="box")
#PA = seans.factorplot(data = dataNew , x = 'Sex_Class' , y = 'Age', kind = 'box')


#Relación entre Pclass y edad
'''
facet = seans.FacetGrid(dataNew, hue="Sex_Class", aspect=4)
facet.map(seans.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plot.show()
'''


#Corrección v2 de la EDAD 

def AgeImpute(df):
    Age = df[0]
    Sex_Class = df[1]
    
    if pand.isnull(Age):
        if Sex_Class == 1: return 41
        elif Sex_Class == 2: return 24
        elif Sex_Class == 3: return 23
        elif Sex_Class == 4: return 40
        elif Sex_Class == 5: return 30
        else: return 24
    else:
        return Age

# Edad corregida
train_df['Age'] = train_df[['Age' , 'Sex_Class']].apply(AgeImpute, axis = 1)
test_df['Age'] = test_df[['Age' , 'Sex_Class']].apply(AgeImpute, axis = 1)

#Trabajaremos ahora para sustituir los valores Null de la variable EDAD (V1)
'''
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # calcula números aleatorios entre la media, std y is_null
    rand_age = nump.random.randint(mean - std, mean + std, size = is_null)
    # llena los valores de NaN en la columna de Edad con los valores aleatorios generados
    age_slice = dataset["Age"].copy()
    age_slice[nump.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
print(train_df["Age"].isnull().sum())
'''
print(train_df.info())
print(test_df.info())

#Trabajaremos ahora para sustituir los valores Null de la variable Embarked
#print(train_df['Embarked'].describe())
common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

#Nos deshacemos de cabina y lo volvemos un valor numerico(Cubierta-Deck) a partir de Primera letra del campo Cabin
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
    
# Nos desacemos de la variable cabina al trabajar con el nuevo campo "Cubierta"
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

#Eliminamos el numero de pasajero solo en el fichero de entrenamiento
train_df = train_df.drop (['PassengerId'], axis = 1)

#observamos un conteo de la información del campo número de ticket
#print(train_df['Ticket'].describe())

#Eliminamos el numero de ticket
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

#Generaremos una suma de "familiares" en vez de trabajar por separado con ellas 
#y otra variable 1,0 que nos permita saber si alguien está solo
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()

#Convertimos la tarifa pagada en un numero entero
data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

#Convertimos la variable de sexto en un valor numerico
sex = {3: 0, 0: 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(sex)


#Convertimos la variable de sexto en un valor numerico
data = [train_df, test_df]
abrev = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # Exportar titulos
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # Los titulos más comunes los unificamos en un grupo númerico
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # Convertir a valor numérico
    dataset['Title'] = dataset['Title'].map(abrev)
    # Los que no tengan ningún titulo recogen el valor 0
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

#Convertimos la puerta de embarque en un número entero
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

#Convertimos a Int la edad y hacemos grupos de edad de un tamaño similar
data = [train_df, test_df]
total = train_df.isnull().sum().sort_values(ascending=False)
print(test_df.info())
print(total)
test_df.head()

for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 24), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 28), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 32), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 37), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 37) & (dataset['Age'] <= 45), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 7


#contamos los grupos de edad según franjas de edad de un rango total(101 a 122)
#print(train_df['Age'].value_counts())

#Buscamos que nos ayude la funcion qcut a formar lo grupos aproximados
#print(pand.qcut(train_df['Fare'], q=4))

#Agrupamos el precio que han pagado por el ticket cada persona
data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#print(train_df['Fare'].value_counts())

#Agregamos al DataSet la edad por la clase del pasajero para diferenciarlos
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

#Generemos otro subgrupo en el DataSet que contenga el precio que paga cada persona
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)


#Mostrar la info tratada
#print(train_df.head(20))
#train_df.info()


#####VISUALIZACIÓN PARA ANÁLISIS DE DATOS#####

#visualización de datos con la libreria Seaborn
#print(seans.barplot(x='Pclass', y='Survived', data=train_df))

#Visualización relación clase y grupo de edad
#grid = seans.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plot.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();

#Mostramos los datos de los supervivientes por sexo
'''
survived = 'survived'
not_survived = 'not survived'
fig, axes = plot.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']==0]
men = train_df[train_df['Sex']==1]
ax = seans.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = seans.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = seans.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = seans.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
'''

#Mostramos visualmente los datos de superviviencia embarcando en cada puerto
'''
FacetGrid = seans.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(seans.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
'''

#Comprobamos la correlación entre el número de familiares
#axes = seans.factorplot('relatives','Survived',data=train_df, aspect = 2.5, )

#Mostrar los nulos y otra información sobre el DATASET Tratado
'''
print('Datos despues de tratarlos:')
print(pand.isnull(train_df).sum())
print(pand.isnull(test_df).sum())
print(train_df.shape)
print(test_df.shape)
print(test_df.head())
print(train_df.head())
'''


####BORRAMOS ATRIBUTOS PARA MEJORAR NUESTRO MODELO#####
# Estos atributos tiene poca importancia en la probabilidad de supervivencia
train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)
train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)

##### APLICACIÓN ALGORITMOS Y ENTRENAMIENTO #####

#Separamos la columna superviviente en otra matriz de datos y la de test sin el ID de Pasajero
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

#Información de las variables
#print(train_df.info())
#print(test_df.info())

#Algoritmo Stochastic Gradient Descent (SGD)
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Stochastic Gradient Descent (SGD):',acc_sgd)

#Algoritmo Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Random Forest:',acc_random_forest)

#Algoritmo Regresión Logistica
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Regresión Logistica:',acc_log)

#Algoritmo K Nearest Neighbor KNN
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#print('Algoritmo K Nearest Neighbor KNN:',acc_log)


#### Algoritmo Gaussian Naive Bayes
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train) 
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Gaussian Naive Bayes:',acc_gaussian)

#Algoritmo Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Perceptron:',acc_perceptron)

#Algoritmo Linear Support Vector Machine(Algoritmo maquinas de Soporte Lineal):
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Linear Support Vector Machine:',acc_linear_svc)

#Support Vector Machines (Algoritmo maquinas de Soporte Lineal)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#print('Precisión Soporte de Vectores:',acc_svc)

#Algoritmo Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#print('Algoritmo Decision Tree:',acc_decision_tree)



#Mostramos cual de los algoritmos es el más fiable en una matriz de datos

results = pand.DataFrame({
    'Model': ['Support Vector Machines', 'Linear Support Vector Machines', 'KNN', 'Regresión Logistica', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))


#Afinar el algoritmo Random Forest porque es de los que mejor tasa de acierto tiene

rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
#print("Scores:", scores)
#print("Mean:", scores.mean())
#print("Standard Deviation:", scores.std())


#Mostrar la importancia de cara atributo sobre la probabilidad de sobrevivir
#importances = pand.DataFrame({'feature':X_train.columns,'importance':nump.round(random_forest.feature_importances_,3)})
#importances = importances.sort_values('importance',ascending=False).set_index('feature')
#print(importances.head(15))

#Visualizar graficamente la importancia de los atributos del Dataset
#importances.plot.bar()


#####AFINAR EL MODELO#####

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
#clf.fit(X_train, Y_train)
#print(clf.bestparams)


#Nuevos parametros de Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
#print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


#####CALCULAR PRECICIÓN UTILIZANDO LOS MODELOS#####

#Matriz de Precisión
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
#print(confusion_matrix(Y_train, predictions))

#Precisión del modelo
#print("Precision:", precision_score(Y_train, predictions))
#print("Recall:",recall_score(Y_train, predictions))

#Puntuación F
print("Preción F",f1_score(Y_train, predictions))


#Visualizar curva de PRECISIÓN

y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]
'''
precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plot.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plot.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plot.xlabel("threshold", fontsize=19)
    plot.legend(loc="upper right", fontsize=19)
    plot.ylim([0, 1])

plot.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
#plot.show()
'''

#Otra opción de visualizar la precisión
'''
def plot_precision_vs_recall(precision, recall):
    plot.plot(recall, precision, "g--", linewidth=2.5)
    plot.ylabel("recall", fontsize=19)
    plot.xlabel("precision", fontsize=19)
    plot.axis([0, 1.5, 0, 1.5])

plot.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plot.show()
'''

#Curva ROC AUC
#calcular la tasa de verdaderos positivos y la tasa de falsos positivos
'''
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
#plotting entre cada uno de ellos
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plot.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plot.plot([0, 1], [0, 1], 'r', linewidth=4)
    plot.axis([0, 1, 0, 1])
    plot.xlabel('False Positive Rate (FPR)', fontsize=16)
    plot.ylabel('True Positive Rate (TPR)', fontsize=16)

plot.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plot.show()
'''

#Puntuación de ROC AUC
r_a_score = roc_auc_score (Y_train, y_scores) 
print ("ROC-AUC-Score:", r_a_score)


#GENERAR EL NUEVO FICHERO DE PREDICCION

#Crear un DataFrame con las identificaciones de los pasajeros y nuestra predicción sobre si sobrevivieron o no.
submission = pand.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_prediction})

filename = 'PracticaIndividual/gender_submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


#EXPORTACIÓN DEL DATASET FINAL EN OTRO FICHERO CSV
#train_df.to_csv(r'PracticaIndividual/export_TRAIN_DEFINITIVO.csv', index = False, header=True)
#test_df.to_csv(r'PracticaIndividual/export_TEST_DEFINITIVO.csv', index = False, header=True)