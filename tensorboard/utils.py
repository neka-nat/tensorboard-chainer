import math
import numpy as np
irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrows (int, optional): Number of rows in grid. Final grid size is
            (B / nrow, nrow). Default is 8.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each(bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value(float, optional): Value for the padded pixels.
    """
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensorlist = tensor
        numImages = len(tensorlist)
        size = tuple([numImages] + tensorlist[0].shape)
        tensor = np.zeros(size)
        for i in irange(numImages):
            tensor[i] = tensorlist[i].copy()

    assert tensor.ndim < 5, "'tensor.ndim' must be less than 5. the given 'tensor.ndim' is %d." % tensor.ndim

    if tensor.ndim == 1:
        tensor = tensor.reshape((1, tensor.shape[0]))
    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.reshape((1, tensor.shape[0], tensor.shape[1]))
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = np.concatenate((tensor, tensor, tensor), 0)
        return tensor
    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), 1)

    if normalize is True:
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img = img.clip(min=min, max=max)
            img = (img - min) / (max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = np.ones((3, height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2)) * pad_value
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid[:, (y * height + 1 + padding // 2):((y + 1) * height + 1 + padding // 2 - padding), (x * width + 1 + padding // 2):((x + 1) * width + 1 + padding // 2 - padding)] = tensor[k]
            k += 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = (grid * 255).clip(0, 255).astype(np.uint8).transpose((1, 2, 0))
    im = Image.fromarray(ndarr)
    im.save(filename)
