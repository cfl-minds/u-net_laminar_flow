import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from time import time
import pickle
import matplotlib.pyplot as plt


def read_scalar(champ, fichier):
    start = fichier[fichier.x.str.contains(champ)].index[0]
    end = start + N
    df = fichier.loc[start + 1:end, ].reset_index(drop=True)
    df.columns = [champ]
    try:
        df = df.astype('float64')
    except ValueError:
        pass
    return df


def flow_field(vtu):
    Pression = read_scalar('Pression', vtu)
    # BordNoeud = read_scalar('BordNoeud', vtu)
    AppartientEntree6 = read_scalar('AppartientEntree6', vtu)
    # AppartientTop = read_scalar('AppartientTop', vtu)
    # AppartientOutlet = read_scalar('AppartientOutlet', vtu)
    # AppartientBottom = read_scalar('AppartientBottom', vtu)
    # AppartientObjet = read_scalar('AppartientObjet', vtu)
    Vitesse = read_scalar('Vitesse', vtu)
    Vitesse = pd.DataFrame(np.delete(np.asarray(Vitesse.Vitesse.str.split(' ').tolist()), -1, 1).astype(np.float))
    Vitesse.columns = ['U', 'V', 'W']
    Vitesse = Vitesse.astype('float64')
    return Pression, AppartientEntree6, Vitesse


start = time()

### définir un maillage structuré uniforme

dx = 0.01
dy = 0.01
x = 0.5 * dx + np.arange(-5, 5, dx)
y = 0.5 * dy + np.arange(-5, 10, dy)

grid = np.zeros((len(x) * len(y), 2))

for i in range(len(x)):
    for j in range(len(y)):
        grid[i * len(y) + j] = np.array([y[j], x[i]])

for index in range(10):
    ##lire le maillage d'un vtu
    vtu = pd.read_csv('D:/DS0/vtus/shape_{}.vtu'.format(index))
    N = int(vtu.iloc[1][0][27:32])  ## number of nodes
    vtu.columns = ['x']

    noeuds = pd.DataFrame()
    noeuds[['x', 'y']] = vtu.iloc[4:4 + N].x.str.split(' ', expand=True)[[0, 1]]
    noeuds = noeuds.astype(np.float64)

    #### lire champs de solution
    Pression, AppartientEntree6, Vitesse = flow_field(vtu)
    shape = np.asarray(AppartientEntree6.AppartientEntree6)
    shape = 1.0 - shape
    p = np.asarray(Pression.Pression)
    u = shape * np.asarray(Vitesse.U)
    v = shape * np.asarray(Vitesse.V)

    ###interpoler les données non-structurées sur le maillage structuré
    data = np.array([shape, u, v, p]).T
    data1 = griddata(noeuds.values, data, grid)

    ###Représenter les champs structurés par une image
    image = np.zeros((len(x), len(y), 4))
    for k in range(grid.shape[0]):

        i = len(x) - 1 - (k // len(y))
        j = k % len(y)
        image[i, j] = data1[k]

    sshape = image[:, :, 0]
    uu = image[:, :, 1] * sshape
    vv = image[:, :, 2] * sshape
    pp = image[:, :, 3] * sshape
    velocity = (uu ** 2 + vv ** 2) ** (1 / 2)

    with open('D:/DataUnet/u/shape_{}.pickle'.format(index), 'wb') as handle1:
        pickle.dump(uu, handle1, protocol=pickle.HIGHEST_PROTOCOL)
    with open('D:/DataUnet/v/shape_{}.pickle'.format(index), 'wb') as handle2:
        pickle.dump(vv, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    with open('D:/DataUnet/p/shape_{}.pickle'.format(index), 'wb') as handle3:
        pickle.dump(pp, handle3, protocol=pickle.HIGHEST_PROTOCOL)
    with open('D:/DataUnet/velocities/shape_{}.pickle'.format(index), 'wb') as handle4:
        pickle.dump(velocity, handle4, protocol=pickle.HIGHEST_PROTOCOL)
    with open('D:/DataUnet/shapes/shape_{}.pickle'.format(index), 'wb') as handle5:
        pickle.dump(sshape, handle5, protocol=pickle.HIGHEST_PROTOCOL)

end = time()
print(end - start)