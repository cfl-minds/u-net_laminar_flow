import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from time import time


dx = 0.01
dy = 0.01
N = 12000##nombre de formes aléatoires
dir_vtu = 'D:/DS0/vtus/'
dir_matrices = 'D:/DataUnet/'


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
x = 0.5 * dx + np.arange(-5, 5, dx)
y = 0.5 * dy + np.arange(-5, 10, dy)

grid = np.zeros((len(x) * len(y), 2))

for i in range(len(x)):
    for j in range(len(y)):
        grid[i * len(y) + j] = np.array([y[j], x[i]])

toutes_formes = np.zeros((len(x), len(y), 1, 10))
toutes_imgs = np.zeros((len(x), len(y), 3, 10))
start = time()
for index in range(N):
    ##lire le maillage d'un vtu
    vtu = pd.read_csv(dir_vtu + 'shape_{}.vtu'.format(index))
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
    for k in range(grid.shape[0]):
        i = len(x) - 1 - (k // len(y))
        j = k % len(y)
        toutes_formes[i, j, 0, index] = data1[k, 0]
        toutes_imgs[i, j, :, index] = data1[k, 1:]

bas = toutes_imgs.min()
haut = toutes_imgs.max()

for index in range(10):
    velocity = (toutes_imgs[:, :, 0, index] ** 2 + toutes_imgs[:, :, 1, index] ** 2) ** (1 / 2)
    velocity = np.ma.masked_where(toutes_formes[:, :, 0, index] < 0.9, velocity)
    uu = np.ma.masked_where(toutes_formes[:, :, 0, index] < 0.9, toutes_imgs[:, :, 0, index])
    vv = np.ma.masked_where(toutes_formes[:, :, 0, index] < 0.9, toutes_imgs[:, :, 1, index])
    pp = np.ma.masked_where(toutes_formes[:, :, 0, index] < 0.9, toutes_imgs[:, :, 2, index])

    cmap = plt.cm.coolwarm
    cmap.set_bad(color='black')

    plt.imsave(dir_matrices + 'shapes/shape_{}.png'.format(index), toutes_formes[:, :, 0, index], cmap='gray')
    plt.imsave(dir_matrices + 'u/shape_{}.png'.format(index), uu, cmap=cmap, vmin=bas, vmax=haut)
    plt.imsave(dir_matrices + 'v/shape_{}.png'.format(index), vv, cmap=cmap, vmin=bas, vmax=haut)
    plt.imsave(dir_matrices + 'p/shape_{}.png'.format(index), pp, cmap=cmap, vmin=bas, vmax=haut)
    plt.imsave(dir_matrices + 'velocities/shape_{}.png'.format(index), velocity, cmap=cmap, vmin=bas, vmax=haut)

end = time()
print(end - start)