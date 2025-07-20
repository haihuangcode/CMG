import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class Draw_Heatmap(nn.Module):
    def __init__(self):
        super(Draw_Heatmap, self).__init__()

    def forward(self, m1, m2, m3, video_id, modality,epoch):
        matrix1 = m1.cpu().numpy()
        matrix2 = m2.cpu().numpy()
        matrix3 = m3.cpu().numpy()

        matrices = [matrix1, matrix2, matrix3]

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 13))

        for i, matrix in enumerate(matrices):
            im = axs[i].imshow(matrix, cmap='viridis_r', interpolation='nearest')
            for j in range(matrix.shape[0]):
                for k in range(matrix.shape[1]):
                    text = axs[i].text(k, j, np.round(float(matrix[j, k]), 2), ha="center", va="center", color="w",fontsize=8,weight=3)

        axs[0].set_title('pred', y=-0.16)
        axs[1].set_title('label', y=-0.16)
        axs[2].set_title('result', y=-0.16)
        plt.text(0,-25,s=video_id,fontsize=20,weight=6,color="k")

        fig.colorbar(im, ax=axs)

        if(modality=="va"):
            plt.savefig('heatmap/va/epoch_' f'{epoch}_' f'{video_id}''.png',dpi=200,bbox_inches = 'tight')
        elif(modality=="av"):
            plt.savefig('heatmap/av/epoch_' f'{epoch}_' f'{video_id}''.png',dpi=200,bbox_inches = 'tight')

