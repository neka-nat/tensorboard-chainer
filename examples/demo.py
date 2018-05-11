import math
import chainer
import numpy as np
from datetime import datetime
from tb_chainer import utils, SummaryWriter

vgg = chainer.links.VGG16Layers()
writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]
for n_iter in range(100):
    M_global = np.random.rand(1) # value to keep
    writer.add_scalar('M_global', M_global[0], n_iter)
    x = np.random.rand(32, 3, 64, 64) # output from network
    if n_iter % 10 == 0:
        x = utils.make_grid(x)
        writer.add_image('Image', x, n_iter)
        x = np.zeros(sample_rate*2)
        for i in range(x.shape[0]):
            x[i] = np.cos(freqs[n_iter//10] * np.pi * float(i) / float(sample_rate)) # sound amplitude should in [-1, 1]
        writer.add_audio('Audio', x, n_iter)
        for name, param in vgg.namedparams():
            writer.add_histogram(name, chainer.cuda.to_cpu(param.data), n_iter)
        writer.add_text('Text', 'text logged at step:'+str(n_iter), n_iter)
        writer.add_text('another Text', 'another text logged at step:'+str(n_iter), n_iter)

video = np.random.rand(16, 3, 16, 64, 64) # (batchsize, channel, time, height, width)
writer.add_video('Video', vid_tensor=video)

writer.close()
