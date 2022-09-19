import os
import random

import torch
from torch.autograd import Variable
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset import val_Dataset
from net import model_TransSR

import numpy as np

from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))

def val(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/default.txt')
    config_dict = opt._spec(kwargs)

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    save_output_folder = '../val_output/%s/%s/' % (opt.path_key, str(opt.net_idx))
    os.makedirs(save_output_folder, exist_ok=True)

    # stage 3
    save_model_list = sorted(os.listdir(save_model_folder))
    use_model = [each for each in save_model_list if each.endswith('pkl')][0]
    use_model_path = save_model_folder + use_model
    config_dict = non_model.update_kwargs(use_model_path, kwargs)
    opt._spec(config_dict)
    print('load config done')

    # stage 4 Dataloader Setting
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    GLOBAL_SEED = 2022
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ###### GPU ######
    data_gpu = opt.gpu_idx
    torch.cuda.set_device(data_gpu)

    save_model_path = save_model_folder + use_model
    save_dict = torch.load(save_model_path, map_location=torch.device('cpu'))
    config_dict = save_dict['config_dict']
    config_dict.pop('path_img')
    config_dict['mode'] = 'test'
    opt._spec(config_dict)

    # val set
    val_list = [each.split('.')[0] for each in sorted(os.listdir(opt.path_img + 'val/1mm/'))]
    val_set = val_Dataset(val_list)
    val_data_num = len(val_set.img_list)
    val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
                                num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
    print('load val data done, num =', val_data_num)

    load_net = save_dict['net']
    load_model_dict = load_net.state_dict()

    net = model_TransSR.TVSRN()
    net.load_state_dict(load_model_dict, strict=False)

    del save_dict
    net = net.cuda()
    net = net.eval()

    with torch.no_grad():
        pid_list = []
        psnr_list = []
        ssim_list = []

        for i, return_list in tqdm(enumerate(val_batch)):
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]

            pid_list.append(case_name)

            x = x.squeeze().data.numpy()
            y = y.squeeze().data.numpy()

            y_pre = np.zeros_like(y)
            pos_list = pos_list.data.numpy()[0]

            for pos_idx, pos in enumerate(pos_list):
                tmp_x = x[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                tmp_x = torch.from_numpy(tmp_x)
                tmp_x = tmp_x.unsqueeze(0).unsqueeze(0)
                im = Variable(tmp_x.type(torch.FloatTensor).cuda())
                tmp_y_pre = net(im)
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                D = y_for_psnr.shape[0]
                pos_z_s = 5 * tmp_pos_z + 3
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x

                y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s + opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

            del tmp_y_pre, im

            y_pre = y_pre[5:-5]
            y = y[5:-5]

            save_name_pre = save_output_folder + '%s_pre.nii.gz' % case_name
            output_pre = sitk.GetImageFromArray(y_pre)
            sitk.WriteImage(output_pre, save_name_pre)

            psnr = non_model.cal_psnr(y_pre, y)
            psnr_list.append(psnr)
            
            pid_ssim_list = []
            for z_idx, z_layer in enumerate(y_pre):
                mask_layer = y[z_idx]

                tmp_ssim = non_model.cal_ssim(mask_layer, z_layer, cuda_use=data_gpu)
                pid_ssim_list.append(tmp_ssim)

            ssim_list.append(np.mean(pid_ssim_list))

        print(np.mean(psnr_list))
        print(np.mean(ssim_list))

if __name__ == '__main__':
    import fire

    fire.Fire()
    

