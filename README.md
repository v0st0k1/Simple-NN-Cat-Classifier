
# Cat Classifier

Modelo de regresión lógica compuesto por una red neuronal de una sola neurona, equivalente a un modelo de regresión clásico.
Los resultados obtenidos son:

Precision de entrenamiento: 99.52153110047847 %

Precicion de test: 70.0 %

Como datos adicionales decir que se ha inicializado la matriz de pesos $W$ a ceros, práctica no recomendada para cuando se tenga más de una neurona por capa. Se ha usado como función de activación en la capa de salida (y única) la función sigmoide, ya que se trata de un problema de clasificación. Para el backpropagation usamos el algoritmo de descenso del gradiente.

Para un ejemplar $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoide(z^{(i)})\tag{2}$$ 
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

El coste se calcula para todos los ejemplares:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$
