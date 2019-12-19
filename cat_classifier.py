"""
Vamos a crear un sencillo clasificador de imagenes que nos dirá si una imagen
dada es un gato (y = 1) o no lo es (y = 0)
Para dicho propósito construiremos un modelo de red neuronal compuesto por una
única neurona.
Este software fue desarrollado para completar el curso de Deep Learnig impartido
por deeplearning.ai™
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#%matplotlib inline

# Cargamos los datos
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

#Las imágenes, compuestas por una matriz tridimensional cuya tercera dimensión
#consta de tamaño = 3, correspondiente a las tres capas del RGB
#por motivos de eficiencia aglutinaremos todos los pixeles en un vector columna
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

#Estandarizamos el dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#Una vez los datos están listos, vamos a crear las funciones de ayuda
#Funcion de activacion
def sigmoid(z):
    ''' Calcula la funcion sigmoide de z

        Args:
            z(number or numpy array)

        Returns:
            s(number or np.array) funcion sigmoide para z
    '''
    s = (1+np.exp(-z))**-1
    return s

def initialize_with_zeros(dim):
    ''' Crea los vectores necesarios para w y biass b
        Sabemos que no es bueno inicializar con 0, porque todas las neuronas
        estarían calculando la misma función, pero este caso es una única neurona
        así que no pasa nada

        Args:
            dim(number): dimension de los vectores a inicializar

        Returns:
            w(vector): array con los pesos W inicializados

            b(scalar): corresponde al biass
    '''
    w = np.zeros((dim,1))
    b = 0
    return w, b

#FORWARD y BACKPROPAGATION
#GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    ''' Funcion que computa la funcion de coste y el calculo del gradiente

        Args:
            w(array): vector con pesos, shape = (num_px*num_px*3, 1)

            b(scalar): biass

            X(array): los datos, shape = (num_px*num_px*3, numero de ejemplares)

            Y(number-boolean): 0 si no es gato, 1 si es gato

        Return:
            cost(float): resultado funcion coste (probabilidad)

            dw(array): gradiente de la funcion perdida con respecto parametro w
                      shape = w.shape
            db(array): gradiente de la funcion perdida con respecto parametro b
                      shape = b.shape

    '''
    m = X.shape[1]

    #Forward propagation
    A = sigmoid(np.dot(w.T,X)+b)
    cost = np.sum(np.multiply(Y,np.log(A))+np.multiply((1-Y),np.log(1-A)))/(-m)

    #Backward propagation
    dw = np.dot(X,(A-Y).T) / m
    db = np.sum(A-Y) / m

    grads = {"dw" : dw,
             "db": db}
    cost = np.squeeze(cost)

    return grads, cost

#Funcion gradiente: optimizacion
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    ''' Optimiza w y b usando el algoritmo de descenso del gradiente

        Args:
            w(array): vector con pesos, shape = (num_px*num_px*3, 1)

            b(scalar): biass

            X(array): los datos, shape = (num_px*num_px*3, numero de ejemplares)

            Y(number-boolean): 0 si no es gato, 1 si es gato

            num_iterations(int): numero de iteraciones para el bucle de optimizacion

            learning_rate(float): para actualizar pesos segun alg. descendo grad

            print_cost(boolean): para ir imprimiendo los costes cada 100 pasos

        Returns:
            params(dictionary): contiene pesos w y biass b

            grads(dictionary): contiene el gradiente de los pesos w y biass b
                              respecto funcion de perdida

            costs(list): lista con todos los costes computados durante optimizacion
                        usado para imprimer la curva de aprendizaje
    '''
    costs = []

    for i in range(num_iterations):
        #Cost y calculo del gradiente
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        #actualizacion
        w = w - learning_rate * dw
        b = b - learning_rate * db

        #Vamos guardando costs para luego plotear grafica y demas
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Coste despues de iteracion %i: %f" %(i, cost))

    params = {"w" : w,
              "b" : b}

    grads = {"dw" : dw,
             "db" : db}

    return params, grads, costs

#Funcion predicción
def predict(w, b, X):
    ''' Predice usando la regresión lógica para los parámetros w y b si X es un gato

        Args:
            w(array): vector con pesos, shape = (num_px*num_px*3, 1)

            b(scalar): biass

            X(array): los datos, shape = (num_px*num_px*3, numero de ejemplares)

        Returns:
            Y_prediction(numpy array): con todas las predicciones (0/1) para
                                       los ejemplares en X
    '''
    m = X.shape[1]

    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    #Computa vector A con predicciones de si es gato o no
    A = sigmoid(np.dot(w.T, X) + b)

    #Convertir probabilidades en decisiones en funcion si supera 0.5
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

#Agrupamos todas las funciones y creamos nuestro modelo
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    ''' Construccion del modelo de regresión lógica

        Args:
            X_train(np.array): set de entrenamiento shape = (num_px*num_px*3, m_train)

            Y_train(np.array): etiquetas para el set de entrenamiento shape = (1, m_train)

            X_test(np.array): set de test shape = (num_px*num_px*3, m_test)

            Y_test(np.array): etiquetas para el set de test shape = (1, m_test)

            num_iterations(int): numero de iteraciones para el bucle de optimizacion

            learning_rate(float): para actualizar pesos segun alg. descendo grad

            print_cost(boolean): para ir imprimiendo los costes cada 100 pasos

        Returns:
            d(dictionary): diccionario con informacion sobre le modelo
    '''

    #Inicializacion de parámetros
    w, b = initialize_with_zeros(X_train.shape[0])

    #Descenso del gradiente
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    #Recuperacion de parametros w y b
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    #Imprimir train/test errors
    print("Precision de entrenamiento: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("Precicion de test: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

#Y corremos el modelo!
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.01, print_cost = True)

#Ejemplo para una foto en concreto dentro del set
for index in [20, 21, 22, 23, 24, 25, 30]:
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    if d["Y_prediction_test"][0,index] == 1:
        gato = 'es un gato'
    else:
        gato = 'no es un gato'
    print ("y = " + str(test_set_y[0,index]) + ", la prediccion es que " + gato)
    plt.title("y = " + str(test_set_y[0,index]) + ", la prediccion es que " + gato)
    plt.show()
