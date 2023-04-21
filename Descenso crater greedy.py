# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:25:50 2021

@author: netob
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import struct
import copy
import math
import time
import random

def puedeNavegar(fila, columna):
    if fila==0 and columna==0:
        if abs(image_data[fila + 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    elif fila==0 and columna==discr.shape[1]-1:
        if abs(image_data[fila + 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    elif fila==discr.shape[0]-1 and columna==0:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    elif fila==discr.shape[0]-1 and columna==columna==discr.shape[1]-1:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    elif fila == 0:
        if abs(image_data[fila][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    elif columna == 0:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < .5:
            return .5
        else:
            return 0
    elif fila == discr.shape[0]-1:
        if abs(image_data[fila][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    elif columna == discr.shape[1]-1:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0
    else:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < .5 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < .5:
            return 1
        else:
            return 0


# Archivo de datos
data_file = open("cdata.bin", "rb")

# Cantidad de renglones de la imagen (INT32, 4 bytes)
data = data_file.read(4)
n_rows = int.from_bytes(data, byteorder='little')
print('Rows:', n_rows)

# Cantidad de columnas de la imagen (INT32, 4 bytes)
data = data_file.read(4)
n_columns = int.from_bytes(data, byteorder='little')
print('Columns:', n_columns)

# Escala de la imagen (metros/pixel) (FLOAT64, 8 bytes)
data = data_file.read(8)
scale = struct.unpack('d', data)
scale = scale[0]
print('Scale:', scale, 'meters/pixel')

# Datos de la imagen (arreglo de números códificados en float64, 8 bytes por cada pixel)
image_size = n_rows * n_columns
data = data_file.read(8*image_size) 

# Transforma los datos de la imagen en un arreglo de numpy
image_data = np.frombuffer(data)
image_data = image_data.reshape((n_rows, n_columns))

# Superfice en 2D
cmap = copy.copy(plt.cm.get_cmap('autumn'))
cmap.set_under(color='black')   

ls = LightSource(315, 45)
rgb = ls.shade(image_data, cmap=cmap, vmin = 0, vmax = image_data.max(), vert_exag=2, blend_mode='hsv')

fig, ax = plt.subplots()

im = ax.imshow(rgb, cmap=cmap, vmin = 0, vmax = image_data.max(), 
                extent =[0, scale*n_columns, 0, scale*n_rows], 
                interpolation ='nearest', origin ='upper')

cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Altura (m)')

plt.title('Superficie de Marte')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.show()
discr = image_data.copy()


for j in range(len(image_data)):
    for i in range(len(image_data[j])):
        discr[j][i] = puedeNavegar(j, i)
plt.imshow(discr)

#%%
#J ES Y, ES P0
#I ES X, ES P1
class Greedys(object):
    """ Class that represents n-queens placed on a chess board. """
    
    def __init__(self, p):        
        """ 
            This constructor initializes the board with n queens.                         

            n : The number of rows and columns of the chess.
            randomize : True indicates that the queen positions are choosen randomly.
                        False indicates that the queen are placed on the first row.
        """
        self.initial_state= p
        self.j=p[0]
        self.i=p[1]
        
        
    
    def cost(self):
        """ This method calculates the cost of this solution (the number of queens that are not safe). """
        #i,j,k = state
        
        return image_data[self.j][self.i]

    def moves(self,state):
        """ This method returns a list of possible moves given the current placements. """
        i, j, k= state
        actions = []
        if self.mapa[self.j - 1][self.i] == 1:
            actions.append('Arriba')
        if self.mapa[self.j + 1][self.i] == 1:
            actions.append('Abajo')
        if self.mapa[self.j][self.i - 1] == 1:
            actions.append('Izquierda')
        if self.mapa[self.j][self.i + 1] == 1:
            actions.append('Derecha')
    
        return actions

    def neighbor(self):
        
        #i, j, k= state
        neighbors=[]
        
        if discr[self.j + 1][self.i]==1:
            neighbors.append(Greedys((self.j + 1,self.i)))
        
        if discr[self.j - 1][self.i]==1:
            neighbors.append(Greedys((self.j - 1,self.i)))
                
        if discr[self.j][self.i + 1]==1:
            neighbors.append(Greedys((self.j,self.i + 1)))
            
        if discr[self.j][self.i - 1]==1:
            neighbors.append(Greedys((self.j,self.i - 1)))
            
        return neighbors
                                       
#------------------------------------------------------------------------------------------------------------------
#   Program
#------------------------------------------------------------------------------------------------------------------
'''random.seed(time.time()*1000)'''

#P SE ESCRIBE YX
p = (2000, 1500)
greedy = Greedys(p)        # Initialize search
cost = greedy.cost()             # Initial cost    
step = 0;                       # Step count


alpha = 0.995; # Coefficient of the exponential temperature schedule        

t0 = 1;         # Initial temperature
t = t0    
b = np.min(image_data)


while (t > 0.005) and (cost > b):

    
    # Calculate temperature
    #t = t0 * math.pow(alpha, step)
    #step += 1
        
    # Get random neighbor
    neighbor = greedy.neighbor()
    #new_cost = neighbor.cost()

    costos = []
    # Test neighbor
    for n in range(len(neighbor)):
        costo_n = neighbor[n].cost()
        costos.append(costo_n)
    
    new_cost = min(costos)
    idxmin = costos.index(new_cost)
    if new_cost < cost:
        greedy = neighbor[idxmin]
        cost = new_cost
    else:
        break
    '''else:
        # Calculate probability of accepting the neighbor
        p = math.exp(-(new_cost - cost)/t)
        if p >= random.random():
            board = neighbor
            cost = new_cost'''

#    print("Iteration: ", step, "    Cost: ", cost, "    Temperature: ", t)

#P[1] ES X
#P[0] ES Y
# I ES X, J ES Y
#ImaGEDATA[Y][X]

print('Ubicación incial:', p[1], p[0])
print('Ubicación final:', greedy.i, greedy.j)
print('Altura inicial;', image_data[p[0]][p[1]])
print('Altura final:', image_data[greedy.j][greedy.i])
print("--------Solution-----------")
#board.show_board()     