
from sklearn import tree

''' Treino. '''

# 0 - Irregular
# 1 - Suave
features = [[140, 1],[130, 1],[150, 0],[170, 0]]

# 0 - Maça
# 1 - Laranja
labels = [0, 0, 1, 1]

# Classificador.
classificador = tree.DecisionTreeClassifier()

# Encontrar padrões nos dados.
classificador = classificador.fit(features, labels)

# Para essa entrada a saída deveria ser [1], pois é uma laranja, por causa do peso.
print( classificador.predict([[150, 0]]) )