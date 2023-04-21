import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import struct
import copy
import math
from simpleai.search import astar, SearchProblem

import winsound

class Exploracion(SearchProblem):
    
    def __init__(self, initial_state, mapa):
        self.mapa = mapa
        self.initial_state = initial_state
        SearchProblem.__init__(self, self.initial_state)
        
    def actions(self, state):
        i, j, k, coord_alien = state
        actions = []
        if self.mapa[j - 1][i] == 1:
            actions.append('Arriba')
        if self.mapa[j + 1][i] == 1:
            actions.append('Abajo')
        if self.mapa[j][i - 1] == 1:
            actions.append('Izquierda')
        if self.mapa[j][i + 1] == 1:
            actions.append('Derecha')
            
        return actions
    
    def result(self, state, action):
        i, j, k, coord_alien = state
        ia, ja = coord_alien
        
        if action == 'Arriba':
            j -= 1
        if action == 'Abajo':
            j += 1
        if action == 'Izquierda':
            i -= 1
        if action == 'Derecha':
            i += 1       
        k+=1
        dist_alien = math.sqrt((ia - i)**2 + (ja - j)**2)
        a = []
        b = []
        
        if k % 3 == 0:
            if self.mapa[ja][ia - 1] == 1:
                d1 = math.sqrt((ja - j)**2 + (ia - 1 - i)**2)
                a.append((d1,"Izquierda"))
            if self.mapa[ja][ia + 1] == 1:
                d2 = math.sqrt((ja - j)**2 + (ia + 1 - i)**2)
                a.append((d2,"Derecha"))
            if self.mapa[ja + 1][ia] == 1:
                d3 = math.sqrt((ja + 1 - j)**2 + (ia  - i)**2)
                a.append((d3,"Abajo"))
            if self.mapa[ja - 1][ia] == 1:
                d4 = math.sqrt((ja - 1 - j)**2 + (ia  - i)**2)
                a.append((d4,"Arriba"))
                
            for m in range(len(a)):
                b.append(a[m][0])
            max_dist = max(b)
            max_index = b.index(max_dist)
            if max_dist > dist_alien:
                if a[max_index][1] == "Izquierda":
                    nuevo_alien = (ia - 1,ja)
                    coord_alien = nuevo_alien
                if a[max_index][1] == "Derecha":
                    nuevo_alien = (ia + 1,ja)
                    coord_alien = nuevo_alien
                if a[max_index][1] == "Arriba":
                    nuevo_alien = (ia,ja - 1)
                    coord_alien = nuevo_alien
                if a[max_index][1] == "Abajo":
                    nuevo_alien = (ia,ja + 1)
                    coord_alien = nuevo_alien   
        new_state = (i,j,k,coord_alien)
        return new_state
    
    def is_goal(self, state):
        """ 
            This method evaluates whether the specified state is the goal state.

            state : The game state to test.
        """
        i, j, k, coord_alien = state
        coord = (i,j)
        return coord == coord_alien
    
    
    
    def cost(self, state, action, state2):
        """ 
            This method receives two states and an action, and returns
            the cost of applying the action from the first state to the
            second state

            state : The initial game state.
            action : The action used to generate state2.
            state2 : The game state obtained after applying the specfied action.
        """
        return 1
    
    def heuristic(self, state):
        columna_actual, fila_actual, k, coord_alien = state
        columna_objetivo, fila_objetivo = coord_alien
        #distancia = abs(fila_objetivo - fila_actual) + abs(columna_objetivo - columna_actual)  #distancia L1
        distancia = math.sqrt((fila_objetivo - fila_actual)**2+(columna_objetivo - columna_actual)**2)   #distancia L2
        return distancia

def puedeNavegar(fila, columna):
    if fila == 0:
        if abs(image_data[fila][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < 1:
            return 1
        else:
            return 0
    elif columna == 0:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < 1:
            return 1
        else:
            return 0
    elif fila == discr.shape[0] - 1:
        if abs(image_data[fila][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < 1:
            return 1
        else:
            return 0
    elif columna == discr.shape[1] - 1:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < 1:
            return 1
        else:
            return 0
    else:
        if abs(image_data[fila - 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna - 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila][columna + 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila - 1][columna + 1] - image_data[fila][columna]) < 1 and \
        abs(image_data[fila + 1][columna + 1] - image_data[fila][columna]) < 1:
            return 1
        else:
            return 0
                          
    
    
# Archivo de datos
data_file = open("data.bin", "rb")

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

# Submuestrea la imagen original para reducir su tamaño (también es posible escalara la imagen)
sub_rate = 10
scale = sub_rate*scale
image_data = image_data[0::sub_rate, 0::sub_rate]
n_rows, n_columns = image_data.shape

print('Sub-sampling:', sub_rate)
print('New scale:', scale, 'meters/pixel')

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


for i in range(len(image_data)):
    for j in range(len(image_data[i])):
        if image_data[i][j] != -1:
            discr[i][j] = puedeNavegar(i, j)
objetivo = (469, 1094) 
estado_inicial = (449, 1094, 0, objetivo)

result = astar(Exploracion(estado_inicial, discr), graph_search = True)
plt.imshow(discr)
for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial configuration')
    elif i == len(result.path()) - 1:
        print('Después de moverse ', action, '. Goal achieved!')
    else:
        print('Después de moverse ', action)
    print(state)
    
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 4000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)