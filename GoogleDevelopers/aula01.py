# Usando os dados Iris.

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# Dados para treinamento.
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# Dados para testarem o classificador.
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Classificador.
classificador = tree.DecisionTreeClassifier()

# Treinando.
classificador.fit(train_data, train_target)

print( test_target )
print( classificador.predict( test_data ) )


'''
print( iris.feature_names ) # Caracteristicas.
print( iris.target_names ) # Nome dos tipos de flores.
'''
