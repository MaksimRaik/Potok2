import numpy as np

Ea =

def f1( ro, u, p, Z ):

    return - ro / u * ( - Q * u * ( gamma - 1.0 ) / c ** 2 / ( 1.0 - u ** 2 / c ** 2 ) ) * ( - A * ro * Z / u * np.exp( - ro * Ea / p / mu ) )
