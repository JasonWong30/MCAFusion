from PIL import Image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

import time
import imageio

import torchvision.transforms as transforms

from Networks.net import MODEL as net



model = net(in_channel=2)

model_path = "models/model_25.pth"
use_gpu = torch.cuda.is_available()


if use_gpu:

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))

else:
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)


def fusion():


    for i in range(3):
        tic = time.time()
        i = i+1

        # 记得改轮数啊---------------------------------------------------------------------------------------------------------------------------------------------

        path1 = './source images/PET-MRI/Y_PET/'+str(i)+'.png'
        path2 = './source images/PET-MRI/MRI/'+str(i)+'.png'
        #
        # path1 = './source images/CT-MRI/CT/'+str(i)+'.png'
        # path2 = './source images/CT-MRI/MRI/'+str(i)+'.png'
        #
        # path1 = './source images/SPECT-MRI/Y_SPECT/'+str(i)+'.png'
        # path2 = './source images/SPECT-MRI/MRI/'+str(i)+'.png'

        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')

        img1_org = img1
        img2_org = img2

        tran = transforms.ToTensor()

        img1_org = tran(img1_org)
        img2_org = tran(img2_org)

        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img

        model.eval()
        out = model(input_img)

        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        imageio.imwrite('./fusion_result/res_PET-MRI/{}.png'.format(i),result)
        # imageio.imwrite('./fusion_result/res_CT-MRI/{}.png'.format(i), result)
        # imageio.imwrite('./fusion_result/res_SPECT-MRI/{}.png'.format(i), result)


        toc = time.time()
        print('end  {} {}'.format(i // 10, i % 10), ', time:{}'.format(toc - tic))



if __name__ == '__main__':
    fusion()
