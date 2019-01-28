from sklearn.naive_bayes import GaussianNB

''' Calcular a precisão de um classificador no momento. '''
def getAccuracy(features_train, labels_train, features_test, labels_test):
    
    # Criando o classificador.
    clf = GaussianNB()

    # Treinando.
    clf.fit(features_train, labels_train)

    pred = clf.predict(features_test)

    accuracy = clf.score(features_test, labels_test)
    return accuracy