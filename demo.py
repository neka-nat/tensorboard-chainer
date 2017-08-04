import math
import chainer
import cupy
import numpy as np
from datetime import datetime
from tensorboard import SummaryWriter

def make_grid(tensor, padding=2):
    """
    Given a 4D mini-batch Tensor of shape (B x C x H x W), makes a grid of images
    """
    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = int(nmaps**0.5)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = np.ones((3, height*ymaps, width*xmaps))
    k = 0
    sy = 1 + padding // 2
    for y in range(ymaps): 
        sx = 1 + padding // 2
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, sy:sy+height-padding, sx:sx+width-padding] = tensor[k]
            sx += width
            k = k + 1
        sy += height
    return grid

vgg = chainer.links.VGG16Layers()
writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]
for n_iter in range(100):
    M_global = cupy.random.rand(1) # value to keep
    writer.add_scalar('M_global', M_global[0], n_iter)
    x = np.random.rand(32, 3, 64, 64) # output from network
    if n_iter%10==0:
        x = make_grid(x)
        writer.add_image('Image', x, n_iter)
        x = np.zeros(sample_rate*2)
        for i in range(x.shape[0]):
            x[i] = np.cos(freqs[n_iter//10]*np.pi*float(i)/float(sample_rate)) # sound amplitude should in [-1, 1]
        writer.add_audio('Audio', x, n_iter)
        for name, param in vgg.namedparams():
            writer.add_histogram(name, chainer.cuda.to_cpu(param.data), n_iter)
        writer.add_text('Text', 'text logged at step:'+str(n_iter), n_iter)
        writer.add_text('another Text', 'another text logged at step:'+str(n_iter), n_iter)
        
writer.close()
