import numpy as np
from PIL import Image

nets = ["caffenet", "googlenet", "vggf", "vgg16", "vgg19"]


def load(nets):
    res = []
    for net in nets:
        data_path = "perturbations/perturbation_%s.npy" % net
        imgs = np.load(data_path, allow_pickle=True, encoding="latin1")
        # print(imgs.shape)
        img = np.transpose(imgs[0], (0, 1, 2))
        im = Image.fromarray(np.uint8(img))
        im.save("imgs/%s.jpg" % net)
        res.append(im)
    return res


def connet(imgs, rate=1):
    n = len(imgs)
    im = imgs[0]
    width = int(im.size[0] * rate)
    height = int(im.size[1] * rate)
    # im = im.resize((width, height), Image.ANTIALIAS)
    interval = int(0.05 * width)
    toImage = Image.new("RGB", (n * width + interval * (n - 1), height), "white")
    # 构造图片的宽和高，如果图片不能填充完全会出现黑色区域
    for i in range(n):
        im = imgs[i]
        im = im.resize((width, height), Image.ANTIALIAS)
        toImage.paste(im, (i * (width + interval), 0))
    toImage.save("imgs/result.jpg")


if __name__ == "__main__":
    connet(load(nets))
