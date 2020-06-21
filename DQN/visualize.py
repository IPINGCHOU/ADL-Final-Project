import math

import torch
from torch.nn import functional as F


def gaussian_kernel(sigma=3):
    """
    Compute a Gaussian kernel
    :param sigma:  scale of the Normal
    :return: kernel of shape [5 sigma]^2
    """
    kernel_size = int(5 * sigma)

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


def init_masks(masks, k):
    """
    draw a circle of radius 1 every `k` steps and changing mask every step

    :param masks: blank masks of shape [B x C x H x W]
    :param k: interval between each circle
    :return: marked masks
    """
    q = 0
    for i in range(masks.shape[2]):
        for j in range(masks.shape[3]):
            u = k // 2
            masks[q, :, u + (i * k), u + (j * k) - 1:u + (j * k) + 2] = 1
            masks[q, :, u + (i * k) - 1:u + (i * k) + 2, u + (j * k)] = 1
            q += 1
    return masks


def generate_mask(size, k, sigma, downsample=1, contrast=1):
    """
    Generate Gaussian masks with center every k pixels

    :param size: image size (H x W)
    :param k: step size
    :param sigma: scale of the gaussian filter
    :param downsample: downsampling steps
    :param contrast: contrast for the final mask values (mask = torch.clamp(contrast * mask, 0, 1))
    :return: masks of shape [N x C x H x W]
    """

    # compute downscaled masks
    m = (size[0] // k, size[1] // k)
    mask = torch.zeros((m[0] * m[1], 1, size[0] // downsample, size[1] // downsample), dtype=torch.long)
    mask = init_masks(mask, k // downsample)
    mask = torch.tensor(mask, dtype=torch.float)
    kernel = gaussian_kernel(sigma=sigma / downsample)[None, None, :, :]
    convolved = F.conv2d(mask, kernel, groups=1, padding=(kernel.shape[-1] - 1) // 2)

    # upscale masks
    mask = F.interpolate(convolved, size, mode='bilinear')

    # normalize
    m, _ = mask.view(mask.size(0), -1).max(1)
    m = m[:, None, None, None]
    m = torch.where(m > 0, m, torch.tensor(1.0))
    mask = mask / m

    if contrast > 1:
        mask = torch.clamp(contrast * mask, 0, 1)

    return mask


def blur_image(img, sigma, downsample):
    """
    Blur input inage using a Gaussian filter

    :param img: input image
    :param sigma: sale of the Gaussian filter
    :param downsample: down sampling
    :return: blurred image
    """
    original_shape = img.shape[2:]
    channels = img.shape[1]
    size = (img.shape[2] // downsample, img.shape[3] // downsample)
    img = F.interpolate(img, size, mode='nearest')
    kernel = gaussian_kernel(sigma=sigma // downsample)[None, None, :, :].repeat(channels, 1, 1, 1)
    img = F.conv2d(img, kernel, groups=channels, padding=(kernel.shape[-1] - 1) // 2)
    img = F.interpolate(img, original_shape, mode='bilinear')
    return img


def generate_candidate_frames(img, spacing=6, sigma=9, blur_sigma=15, contrast=1.2, downsample=8, use_mode=False):
    """
    Generate the candidates frames used to compute saliencies (I')
    using masks representing Gaussian perturbations with spaced every `spacing steps
    `
    output_i = mask_i * blurred_img + (1-mask_i) * img

    :param img: input image
    :param spacing: space between each perturbation
    :param sigma: scale of the perturbations
    :param blur_sigma: scale of the gaussian noise to blur the image
    :param contrast: contrast for the final mask values (mask = torch.clamp(contrast * mask, 0, 1))
    :param downsample: downsampling size
    :param use_mode: use color mode of the input image instead of the blurred version (mode = background color)
    :return: candidate frames to compute saliency
    """

    masks = generate_mask(img.shape[2:], spacing, sigma, downsample=downsample, contrast=contrast).cuda()

    if use_mode:
        #print(img.size())
        mode, _ = torch.mode(img, -1)#.transpose(1, 0).view(4, -1)
        #print(mode.size())
        blurred_img = mode.unsqueeze(3).expand_as(img)
    else:
        blurred_img = blur_image(img.float(), blur_sigma, downsample)

    outputs = (masks) * blurred_img + (1 - masks) * img.float()

    return outputs, masks


def saliency_map(state, policy, output_shape, spacing=6, sigma=6, blur_sigma=9, contrast=1.5, downsample=6,
                 use_mode=True):
    """
    Compute the saliency map using perturbations in the input state.

    original implementation and paper: https://github.com/greydanus/visualize_atari

    :param state: input state of shape [B x C x H x W]
    :param policy: policy network
    :param output_shape: (H W)
    :param spacing: spacing between each perturbation
    :param sigma: scale of the perturbations
    :param blur_sigma: scale of the Gaussian blur applied to the state
    :param contrast: additional parameter to control the roughness of the perturbations
    :param downsample: compute heavy operations on the downsampled state
    :param use_mode: use the mode of the state (i.e. background color) instead of the blurred image
    :return: Saliency map, Saliency signs
    """
    state = state.float()

    # generate frames with perturbations
    candidates, masks = generate_candidate_frames(state, spacing=spacing, sigma=sigma, blur_sigma=blur_sigma,
                                                  contrast=contrast, downsample=downsample, use_mode=use_mode)

    # compute Value for the state and the perturbed states
    frames = torch.cat([state, candidates], 0)
    values = policy.forward(frames).sum(1)
    L, Ls = values[:1], values[1:]

    # reshape into an image
    m = int(math.sqrt(Ls.shape[0]))
    Ls = Ls.view(m, m).transpose(1, 0)

    # compute saliency sign and values
    Sign = (Ls > L.squeeze()).float()
    S = 0.5 * (Ls - L.squeeze()) ** 2

    # upsample
    S = F.interpolate(S[None, None, :, :], output_shape, mode='nearest')
    Sign = F.interpolate(Sign[None, None, :, :], output_shape, mode='nearest')

    return S, Sign
