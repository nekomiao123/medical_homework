import torch
import numpy as np
import math
from glob import glob
from medpy import metric
from dataset import read_h5
from PIL import Image
from model import Model
import wandb

if __name__ == '__main__':

    # wandb.init(
    #     project='xm_assignment1',
    #     entity='nekokiku',
    #     name='VNet_Test',
    #     save_code=True,
    # )

    model = Model() # load your model here
    model.cuda()
    model.load_state_dict(torch.load(f'./checkpoints/VNet_Dice/epoch_5000.pth'))
    dice = 0.0
    jc = 0.0
    asd = 0.0
    hd = 0.0

    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4

    path_list = glob('./datas/test/*.h5')
    model.eval()
    index = 0
    for path in path_list:
        index += 1
        image, label = read_h5(path)

        w, h, d = image.shape
        sx = math.ceil((w - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((h - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((d - patch_size[2]) / stride_z) + 1

        scores = np.zeros((2, ) + image.shape).astype(np.float32)
        counts = np.zeros(image.shape).astype(np.float32)
        
        # inference all windows (patches)
        for x in range(0, sx):
            xs = min(stride_xy * x, w - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, h - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, d - patch_size[2])

                    # extract one patch for model inference
                    test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    with torch.no_grad():
                        test_patch = torch.from_numpy(test_patch).float().cuda() # if use cuda
                        test_patch = test_patch.unsqueeze(0).unsqueeze(0) # [1, 1, w, h, d]
                        out = model(test_patch)
                        out = torch.softmax(out, dim=1)
                        out = out.cpu().data.numpy() # [1, 2, w, h, d]
                    # record the predicted scores
                    scores[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out[0, ...]
                    counts[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
                    # print(scores.shape, counts.shape)
                    # im = Image.fromarray(im)
                    # im = im.convert('L')
                    # im.save("./image/1.jpg")

        scores = scores / np.expand_dims(counts, axis=0)
        predictions = np.argmax(scores, axis = 0) # final prediction: [w, h, d]
        # temp_img = image[:,:,30]
        # temp = predictions[:,:,30]
        # temp_label = label[:,:,30]

        # images = wandb.Image(temp_img, caption="image")
        # prediction = wandb.Image(temp, caption="prediction")
        # labels = wandb.Image(temp_label, caption="label")
        # wandb.log({"image": images, "prediction": prediction, "label": labels})
        
        # print(temp.shape)
        # for i in range(temp.shape[0]):
        #     for j in range(temp.shape[1]):
        #         if temp[i][j] != 0:
        #             temp[i][j] = 255
        #         if temp_label[i][j] != 0:
        #             temp_label[i][j] = 255

        # im_im = Image.fromarray(temp_img).convert('RGB')
        # pathp = "./image/" + str(index) + "_im.png"
        # im_im.save(pathp)

        # im = Image.fromarray(np.uint8(temp)).convert('L')
        # pathp = "./image/" + str(index) + ".png"
        # im.save(pathp)

        # im_l = Image.fromarray(np.uint8(temp_label)).convert('L')
        # pathl = "./image/" + str(index) + "_label.png"
        # im_l.save(pathl)

        dice += metric.binary.dc(predictions == 1, label == 1)
        jc += metric.binary.jc(predictions == 1, label == 1)
        asd += metric.binary.asd(predictions == 1, label == 1)
        hd += metric.binary.hd95(predictions == 1, label == 1)

    dice /= len(path_list)
    jc /= len(path_list)
    asd /= len(path_list)
    hd /= len(path_list)
    print(f'dice: {dice}')
    print(f'jc: {jc}')
    print(f'asd: {asd}')
    print(f'hd: {hd}')