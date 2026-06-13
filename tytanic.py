import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


tytanic_data = pd.read_csv('train.csv')
clear_data = tytanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#poprzez print(clear_data.info()) można zobaczyć, że w kolumnie Age oraz Embarked są braki danych
age_mean = tytanic_data['Age'].mean()
clear_data['Age'] = clear_data['Age'].fillna(age_mean)
clear_data['Embarked'] = clear_data['Embarked'].fillna(1)#tak szczerze brakowalo tylko 2 wartości wiec mozna je zmyślić recznie
clear_data['Embarked'] = clear_data['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3, 1: 1})
clear_data['Sex'] = clear_data['Sex'].replace({'male': 1, 'female': 2})
#sns.heatmap(clear_data.corr(), annot=True, cmap='coolwarm')
#plt.show() #widac ze wiek nie ma dużego wypływu na przeżycie jednak uważam to za totalny cap
clear_data['Younger_than_10'] = clear_data['Age'].apply(lambda x: 1 if x < 10 else 0)
clear_data['Younger_than_20'] = clear_data['Age'].apply(lambda x: 1 if x < 20 and x >= 10 else 0)
clear_data['Younger_than_40'] = clear_data['Age'].apply(lambda x: 1 if x < 40 and x >= 20 else 0)
clear_data['Younger_than_60'] = clear_data['Age'].apply(lambda x: 1 if x < 60 and x >= 40 else 0)
clear_data['Older_than_60'] = clear_data['Age'].apply(lambda x: 1 if x > 60 else 0)
#agecorr = clear_data[['Survived', 'Younger_than_10', 'Younger_than_20', 'Younger_than_60', 'Older_than_60']]
#sns.heatmap(agecorr.corr(), annot=True, cmap='coolwarm') 
#widac ze dzieci do 10 lat maja 10x mocnejsza korelacje z przezyciem niz osoby w wieku 10-20
#również widac ze osoby po 60 są bardziej sklonne do zgonu niz przeżycia, jednak tutaj korelcja jest słaba
#plt.show()
from sklearn.model_selection import train_test_split
X = clear_data.drop('Survived', axis=1)
y = clear_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# model.fit(X_train, y_train)
# moje_predykcje = model.predict(X_test)
# skutecznosc = accuracy_score(y_test, moje_predykcje)
# print(f"Skuteczność Lasu Losowego: {skutecznosc * 100:.2f}%")

# dla modelu RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42) 
# skuteczność wynosi około 81.56% co jest całkiem dobrym wynikiem jak na ten zbiór danych.
# jednak planuje teraz przetestować inne modele i ewentalnie spróbować wykorzytsać dane z imion lub kabin, które wcześniej usunąłem.

#from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [50, 100, 150, 200], 
#     'max_depth': [3, 4, 5, 6, 7, 8],     
#     'random_state': [42]                 
# }
# rf_model = RandomForestClassifier()
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
# print("Rozpoczynam strojenie. Procesor pracuje...")
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# print(f"Najlepsze parametry znalezione przez maszynę: {grid_search.best_params_}")
# najlepsze_predykcje = best_model.predict(X_test)
# nowa_skutecznosc = accuracy_score(y_test, najlepsze_predykcje)

# print(f"Skuteczność ZOPTYMALIZOWANEGO Lasu Losowego: {nowa_skutecznosc * 100:.2f}%")

#skutecznosc na poziomie około 82,68% jest nieco lepsza
#jednak i tak podejme próbe wykorzystania danych z imion lub kabin
#print(tytanic_data[['Name', 'Cabin']].head(100))
#dość szybko idzie zauważyć że tytułu ograniczaja się do Mrs oraz Mr co jest już wyznaczone przez kolumne z płcią
#kabiny są w zdecydowanej większości Nan, a litera kabiny i tak wskazuje na klase co ponownie jest już zawarte w danych

final_model = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=42)
final_model.fit(X, y)

tytanic_test = pd.read_csv('test.csv')
#jako że nie umiem tworzyć pipelinów to po prostu powtórze te same kroki co wcześniej, ale na zbiorze testowym
clear_test = tytanic_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
clear_test['Age'] = clear_test['Age'].fillna(age_mean)
clear_test['Embarked'] = clear_test['Embarked'].fillna(1)
clear_test['Embarked'] = clear_test['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3, 1: 1})
clear_test['Sex'] = clear_test['Sex'].replace({'male': 1, 'female': 2})

clear_test['Younger_than_10'] = clear_test['Age'].apply(lambda x: 1 if x < 10 else 0)
clear_test['Younger_than_20'] = clear_test['Age'].apply(lambda x: 1 if x < 20 and x >= 10 else 0)
clear_test['Younger_than_40'] = clear_test['Age'].apply(lambda x: 1 if x < 40 and x >= 20 else 0)
clear_test['Younger_than_60'] = clear_test['Age'].apply(lambda x: 1 if x < 60 and x >= 40 else 0)
clear_test['Older_than_60'] = clear_test['Age'].apply(lambda x: 1 if x > 60 else 0)

#clear_test.info()
#okazuje się że w zbiorze testowym również brakuje danych w kolumnie Fare
fare_mean = clear_test['Fare'].mean()
clear_test['Fare'] = clear_test['Fare'].fillna(fare_mean)

final_predictions = final_model.predict(clear_test)

submission = pd.DataFrame({
    'PassengerId': tytanic_test['PassengerId'],
    'Survived': final_predictions
})
submission.to_csv('submission.csv', index=False)
print("Koniec")
#po opublikowaniu na keggle moj wynik to 77,99%