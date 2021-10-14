import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import sys
sys.path.append('../')

from config import ecssd_path, dutomron_path, pascals_path, hkuis_path, dutste_path, xpie_path
#dutomron_path, hkuis_path, dutste_path, xpie_path

from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from fpn_ea_oa_net import FPN

torch.manual_seed(2019)

# set which gpu to use
torch.cuda.set_device(0)

ckpt_path = './ckpt'

args = {
    'resize': [320, 320],
    'max_iter': 6000,
    'save_interval': 3000,  # your snapshot filename (exclude extension name)
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    
    transforms.Resize(args['resize']),
    transforms.ToTensor(),
    transforms.Normalize([0.4923, 0.4634, 0.3974], [0.2737, 0.2659, 0.2836])
])
to_pil = transforms.ToPILImage()

to_test = {'ecssd_path': ecssd_path}
save_path = 'result'

def main():
    net = FPN().cuda()
    
    net.load_state_dict(torch.load(os.path.join(ckpt_path, 'stage3.pth')))
    net.eval()

    results = {}

    with torch.no_grad():

        for name, root in to_test.items():

            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                check_mkdir(save_path)

            img_list = [os.path.splitext(f)[0] for f in os.listdir(root+'/image') if f.endswith('.jpg')]

            idx_i = 0
            #import time
            #start = time.time()
            for idx, img_name in enumerate(img_list):
                #start = time.time()
                idx_i += 1
                if idx_i % 100 == 0:
                    print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))

                #start = time.time()
                img = Image.open(os.path.join(root + '/image', img_name + '.jpg')).convert('RGB')
                img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                prediction = net(img_var)
                prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))

                if args['save_results']:
                    import cv2
                    prediction = cv2.resize(prediction, (np.array(img).shape[1], np.array(img).shape[0]))
                    Image.fromarray(prediction).save(os.path.join(save_path + '/' + img_name + '.png'))

if __name__ == '__main__':
    main()
