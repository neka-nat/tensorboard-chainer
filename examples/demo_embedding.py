from tb_chainer.embedding import add_embedding
import keyword
import numpy as np
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = np.random.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0
    
add_embedding(np.random.randn(100, 5), save_path='embedding1', metadata=meta, label_img=label_img)
add_embedding(np.random.randn(100, 5), save_path='embedding2', label_img=label_img)
add_embedding(np.random.randn(100, 5), save_path='embedding3', metadata=meta)

#tensorboard --logdir embedding1
