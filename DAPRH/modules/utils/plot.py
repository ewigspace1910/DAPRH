import matplotlib.pyplot as plt
import os
import cv2
from .osutils import mkdir_if_missing


def plot_clusters(cdict:dict, save_path:str, columns=5, epoch=0):
    """
    cdict : (dict) {cluster1 :{path1, path2,...}, cluster2: {path1, path2,...}, ....}
    save_path: (str) folder to save figures
    columns: number image in cluster, maximum is 15
    """
    mkdir_if_missing(save_path)

    # showing image
    # create figure

    # setting values to rows and column variables
    rows = len(cdict.keys())
    columns = min(columns, 15)
    fig = plt.figure(figsize=(columns, 2*rows))
    # reading images
    for i, k in enumerate(cdict.keys()):
        for j in range(columns):
            if isinstance(cdict[k][j],str):
                cdict[k][j] = (cdict[k][j], i)
            
            ax_i = fig.add_subplot(rows, columns, i*columns+j+1)
            Image1 = cv2.cvtColor(cv2.imread(cdict[k][j][0]), cv2.COLOR_BGR2RGB)
            plt.imshow(Image1)
            plt.axis('off')
            ax_i.set_title(f"ID:{cdict[k][j][1]}")
    path = os.path.join(save_path,f"cluster-{epoch}.png")
    fig.savefig(path)
    return path


