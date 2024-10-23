from re import L
from scipy.interpolate import griddata
import numpy as np
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

SIZE = 64

#get first row of data from each weather station
dataset_path = "data/Set6"
root = Path(dataset_path)
nodes = [f for f in root.iterdir() if f.is_dir()]
num_nodes = len(nodes)
data = torch.empty(size= (num_nodes, 1, 8), dtype=torch.float)
points = torch.empty(size= (num_nodes, 2), dtype=torch.float)
for i,node in enumerate(nodes):
    df = pd.read_csv(node / "data.csv")
    df['DTG'] = pd.to_datetime(df['DTG'])
    df = df[df['DTG'].dt.year == 2019]
    #drop dtg column
    df = df.drop(columns=['DTG'])[:1]
    data[i] = torch.tensor(df.values)
    location = pd.read_csv(node / "metadata.csv")
    points[i] = torch.tensor(location[['LATITUDE','LONGITUDE']].values[0])

#normalize points to be between 0 and 1 for x and y separately
columns = list(df.columns.values)

min_long, max_long = 3.1688, 7.4312 #min and max long and lat of Netherlands precipitation maps
min_lat, max_lat = 50.82512, 53.4748
#swap lat and long
#points = points[:,[1,0]]

grid_x = np.linspace(min_long, max_long, SIZE)
grid_y = np.linspace(min_lat, max_lat, SIZE)
# print(grid_x.shape)
# print(points.shape)
kriged_maps = []

def makeKrigeMap(nodes):
    '''Create a Krige map from a set of nodes'''
    maps = np.empty([8, SIZE, SIZE])
    for var_num in range(8):
        var_name = columns[var_num]
        #print("NODES: " + str(nodes))
        variable = nodes[:,var_num]
        #print("VAR: " + str(variable))
        #print(variable)
        OK = OrdinaryKriging( #x equals longitude, y = latitude
            points[:,1], 
            points[:,0], 
            variable, 
            coordinates_type="geographic", 
            variogram_model="spherical",
            pseudo_inv=True)
        z, _ = OK.execute('grid', grid_x, grid_y)
        # plt.imshow(z)
        # plt.colorbar()
        # plt.gca().invert_yaxis()
        # plt.savefig('data/interpolation/' + var_name + '.png')
        # plt.close()
        #plt.show()
        maps[var_num] = z

    return maps

if __name__ == "__main__":
    nodes = data
    makeKrigeMap(nodes)