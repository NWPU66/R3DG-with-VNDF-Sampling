from typing import Any
import torch
import numpy as np
import matplotlib
from matplotlib import pyplot
from .graphics_utils import rgb_to_srgb


def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps["turbo"]
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1)
    )
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0.0, 1.0)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))


def show_image_form_data(data: Any, change_channel: bool = False):
    def _show_image_form_data(data: torch.Tensor, change_channel: bool = False):
        if data.max() > 1:
            data = rgb_to_srgb(data)
        if change_channel:
            img = data.detach().cpu().permute(1, 2, 0)
        else:
            img = data.detach().cpu()
        pyplot.imshow(img)  # 显示图片
        pyplot.axis("off")  # 不显示坐标轴
        pyplot.show()
        pyplot.waitforbuttonpress(0)  # 等待用户按下任意键

    if isinstance(data, dict):
        for _, v in data.items():
            _show_image_form_data(v, change_channel)
    elif isinstance(data, torch.Tensor):
        _show_image_form_data(data, change_channel)
    else:
        raise TypeError("Unsupported data type: {}".format(type(data)))
