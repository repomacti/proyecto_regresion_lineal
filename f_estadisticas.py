import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def media(X):
    """
    Calcula la media del conjunto de datos 'X'.

    Parameters
    ----------
    X: np.array
    Arreglo con el conjunto de datos.

    Returns
    -------
    Media del conjunto de datos.
    """
    suma = 0
    for xi in X:
        suma += xi
    
    return suma / len(X)

def varianza(X):
    """
    Calcula la varianza del conjunto de datos 'X'.

    Parameters
    ----------
    X: np.array
    Arreglo con el conjunto de datos.

    Returns
    -------
    Varianza del conjunto de datos.
    """
    xm = media(X)
    suma = 0
    for xi in X:
        suma += (xi - xm)**2
    
    return suma / (len(X)-1)

def covarianza(X, Y):
    """
    Calcula la covarianza entre los conjuntos de datos 'X' y 'Y'.

    Parameters
    ----------
    X: np.array
    Arreglo con el primer conjunto de datos.

    Y: np.array
    Arreglo con el segundo conjunto de datos.

    Returns
    -------
    Covarianza entre las variables 'x' y 'y'.
    """
    xm = media(X)
    ym = media(Y)
    suma = 0.0  
    for xi, yi in zip(X, Y):
        suma += (xi - xm) * (yi - ym)
    
    return suma / (len(X)-1)
    
def reglin(X, Y, xlabel = "$x$", ylabel = "$y$"):
    """
    Calcula la regresión lineal simple entre los conjuntos de datos 'X' y 'Y'.

    Parameters
    ----------
    X: np.array
    Arreglo con el primer conjunto de datos (variable independiente).

    Y: np.array
    Arreglo con el segundo conjunto de datos (variable dependiente).

    xlabel: str
    Etiqueta para el eje 'x'.

    ylabel: str
    Etiqueta para el eje 'y'.
    
    Returns
    -------
    Coeficientes de la regresión lineal (B0 y B1).
    Coeficiente de determinación (R2).
    Arreglos 'x' y 'y' con los valores de la recta aproximada.
    """
    xm = media(X)
    ym = media(Y)
    s2x = varianza(X)
    s2y = varianza(Y)
    Sxy = covarianza(X, Y)
    B1 = Sxy / s2x
    B0 = ym - B1 * xm
    R2 = Sxy**2 / (s2x * s2y)

    xoff = (max(X) - min(X))*0.20
    x = np.linspace(min(X)-xoff, max(X)+xoff, 50)
    y = B0 + B1 * x

    plot_reglin(x, y, X, Y, B0, B1, R2, xlabel, ylabel)
    
    return B0, B1, R2, x, y

def elimina_outliers(X, Y, q1 = 0.25, q3 = 0.75, silent = True):
    """
    Elimina valores atípicos de un conjunto de datos usando los
    rangos intercuartílicos (IQR). Estos rangos se calculan con 
    base en los datos de 'Y' (la variable dependiente). Los 
    valores de la variable independiente 'X' se eliminan de acuerdo
    con los eliminados en 'Y'.
    Algoritmo:
    IQR = Q3 - Q1
    Límite inferior = Q1 - 1.5 * IQR
    Límite superior = Q3 + 1.5 * IQR

    Parameters
    ----------
    X: np.array
    Arreglo con el primer conjunto de datos (variable independiente).

    Y: np.array
    Arreglo con el segundo conjunto de datos (variable dependiente).

    q1: float 
    Primer cuartil del conjunto 'Y'. Valor por omisión = 0.25.

    q3: float 
    Tercer cuartil del conjunto 'Y'. Valor por omisión = 0.75.

    silent: Bool
    Bandera para imprimir o no algunos resultados.
    
    Returns
    -------
    Arreglos sin valores atípicos: 'Xno' y 'Yno'
    """    
    Q1 = np.quantile(Y, q1)
    Q3 = np.quantile(Y, q3)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Determinación de índices a eliminar
    inf = np.where(Y < limite_inferior)
    sup = np.where(Y > limite_superior)

    if not silent:
        print(f"Lim. inferior = {limite_inferior}")
        print(f"Lim. superior = {limite_superior}")

    # Conjuntamos los índices en un conjunto para eliminar las repeticiones.
    idx = set()
    for i in inf[0]:
        idx.add(i)
    for i in sup[0]:
        idx.add(i)
    idx = list(idx)

    # Eliminamos los índices con outliers
    Xno = np.delete(X, idx)
    Yno = np.delete(Y, idx)

    return Xno, Yno

def elimina_outliers_manual(X, Y, vmin, vmax, silent=True):
    """
    Elimina valores atípicos de un conjunto de datos de manera
    manual. Se deben proporcionar los valores mínimo y máximo,
    del conjunto 'Y', a partir de los cuales se eliminan valores 
    atípicos.

    Parameters
    ----------
    X: np.array
    Arreglo con el primer conjunto de datos (variable independiente).

    Y: np.array
    Arreglo con el segundo conjunto de datos (variable dependiente).

    vmin: float 
    Valor mínimo de 'Y'; por debajo de este número se eliminan todos 
    los valores.

    vmax: float 
    Valor máximo de 'Y'; por arriba de este número se eliminan todos 
    los valores.

    silent: Bool
    Bandera para imprimir o no algunos resultados.
    
    Returns
    -------
    Arreglos sin valores atípicos: 'Xno' y 'Yno'
    """ 

    # Determinación de índices a eliminar
    inf = np.where(Y < vmin)
    sup = np.where(Y > vmax)

    if not silent:
        print(f"Lim. inferior = {vmin}")
        print(f"Lim. superior = {vmax}")

    # Conjuntamos los índices en un conjunto para eliminar las repeticiones.
    idx = set()
    for i in inf[0]:
        idx.add(i)
    for i in sup[0]:
        idx.add(i)
    idx = list(idx)

    # Eliminamos los índices con outliers
    Xno = np.delete(X, idx)
    Yno = np.delete(Y, idx)

    return Xno, Yno
    
def plot_boxplots(X, Y):
    """
    Dibuja diagramas de caja (box plot) para representar gráficamente
    las series de datos numéricos 'X' y 'Y' a través de sus cuartiles.
    El ojetivo es observar si hay valores atípicos.
    Grafica también los datos 'X' vs 'Y' para observar su distribución.

    Parameters
    ----------
    X: np.array
    Arreglo con el primer conjunto de datos (variable independiente).

    Y: np.array
    Arreglo con el segundo conjunto de datos (variable dependiente).
    """
    fig, ax = plt.subplots(1,3,figsize=(6,3))
    ax[0].boxplot(X)
    ax[0].set_title(f"x")
    ax[1].boxplot(Y)
    ax[1].set_title(f"y")
    ax[2].scatter(X, Y, marker='.', fc='orange', ec='red')
    ax[2].set_title("Datos")
    ax[2].set_xlabel("$x$")
    ax[2].set_ylabel("$y$")
    plt.tight_layout()
    plt.show()

def plot_reglin(x, y, X, Y, B0, B1, R2, xlabel = "$x$", ylabel = "$y$"):
    """
    Grafica la distribución de puntos de los conjuntos 'X' y 'Y'
    junto con la línea recta producto de la regresión lineal.

    Parameters
    ----------
    x, y: np.array
    Arreglos con los valores aproximados usnado la regresión lineal.
    Se usan para graficar la línea recta.
    
    X: np.array
    Arreglo con el primer conjunto de datos (variable independiente).

    Y: np.array
    Arreglo con el segundo conjunto de datos (variable dependiente).

    B0, B1: float
    Coeficientes de la regresión lineal.

    R2: float
    Coeficiente de determinación.

    xlabel: str
    Etiqueta para el eje 'x'.

    ylabel: str
    Etiqueta para el eje 'y'.
    """
    fig = plt.figure(figsize=(6,3))
    plt.plot(x, y, color='black', label=f"B0 = {B0:5.2f}, B1 = {B1:5.2f}", zorder=10)
    plt.scatter(X, Y, marker = 'o', s=40, 
                fc='orange', ec='red', alpha = 0.75,  zorder=5, label="Datos")
    plt.title(f"Regresión lineal R2 = {R2:6.4f}", fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid()
    plt.legend(loc="best", fontsize=8)
    plt.show()

def histograma(X, bins=20, title=""):
    """
    Dibuja un histograma de los datos 'X'.

    Parameters
    ----------
    X: np.array
    Arreglo con el conjunto de datos.

    bins: int
    Número de barras.

    title: str
    Título del histograma.
    """
    n, bins, patches = plt.hist(X, bins=20, fc='turquoise', ec = "seagreen",alpha=0.75)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.xticks(bin_centers, fontsize=8, rotation=90)
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.show()