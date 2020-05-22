import matplotlib.pyplot as plt
import numpy as np




def imshow(img):
    '''Functions to show an image'''

    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()