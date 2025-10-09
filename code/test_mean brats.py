# ensemble
import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
#设置可见的显卡号码环境变量  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume_mean
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_brats import brats
import h5py
from icecream import ic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import io
from PIL import Image

def inference(args, multimask_output, db_config, net, test_save_path=None):
    with open(args.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    writer = SummaryWriter(log_dir=os.path.join("output/test/tensorboard_logs", args.exp))

    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    first_list = np.zeros([len(image_list), 4])
    second_list = np.zeros([len(image_list), 4])
    third_list = np.zeros([len(image_list), 4])
    count = 0

    for case in tqdm(image_list):
        h5f = h5py.File(args.root_path + "/test_data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        metrics, outputs = test_single_volume_mean(case, image, label, net, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case, z_spacing=db_config['z_spacing'])
        if 'uncertainty_map' in outputs:
            slice_map = outputs['uncertainty_map']  # shape [H, W], torch.Tensor
            slice_map = slice_map * 20.0
            slice_map = torch.clamp(slice_map, 0.0, 1.0)  # 放大增强，限制到[0, 1]

            # 如果图像太“黑”，跳过不写入
            if slice_map.max() < 0.1:
                continue

            # ==== (可选) 叠加原图 ====
            # 假设你能从 test_single_volume_mean 中也返回原图的对应 slice（比如 'image_slice'），否则先注释这段
            # image_slice = outputs['image_slice']  # shape [H, W]
            # image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)

            # ==== matplotlib 画图 ====
            fig, ax = plt.subplots(figsize=(6, 6))  # 提高清晰度

            # 背景图（原图灰度）
            # ax.imshow(image_slice.numpy(), cmap='gray')  # 若想叠加原图请取消注释上面两行

            # 主热图
            ax.imshow(slice_map.numpy(), cmap='inferno', alpha=0.9)

            # 添加轮廓线（不确定性 > 0.3）
            ax.contour(slice_map.numpy(), levels=[0.3], colors='cyan', linewidths=0.8)

            # 标记最大不确定性位置
            y, x = torch.nonzero(slice_map == slice_map.max(), as_tuple=False)[0].tolist()
            ax.plot(x, y, 'ro', markersize=4)  # 红点标注最大值

            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)

            heatmap_img = TF.to_tensor(Image.open(buf))  # shape [3, H, W]
            writer.add_image(f"PrettyUncertaintyMap/{case}", heatmap_img, global_step=count)


        first_metric = metrics[0]
        second_metric = metrics[1]
        third_metric = metrics[2]

        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

        first_list[count] = first_metric
        second_list[count] = second_metric
        third_list[count] = third_metric
        count += 1

    # 计算平均指标
    avg_metric1 = np.nanmean(first_list, axis=0)
    avg_metric2 = np.nanmean(second_list, axis=0)
    avg_metric3 = np.nanmean(third_list, axis=0)
    avg_all = (avg_metric1 + avg_metric2 + avg_metric3) / 3

    # ✅ 保存为 CSV 文件（路径按实验名划分文件夹）
    save_dir = os.path.join("output/test/save_excel", args.exp)
    os.makedirs(save_dir, exist_ok=True)
    write_csv = os.path.join(save_dir, f"{args.exp}_test_mean.csv")

    df = pd.DataFrame({
        '1-dice': first_list[:, 0],
        '1-hd95': first_list[:, 1],
        '1-asd': first_list[:, 2],
        '1-jc': first_list[:, 3],
        '2-dice': second_list[:, 0],
        '2-hd95': second_list[:, 1],
        '2-asd': second_list[:, 2],
        '2-jc': second_list[:, 3],
        '3-dice': third_list[:, 0],
        '3-hd95': third_list[:, 1],
        '3-asd': third_list[:, 2],
        '3-jc': third_list[:, 3]
    })

    # 添加平均行
    avg_row = {
        '1-dice': avg_metric1[0], '1-hd95': avg_metric1[1], '1-asd': avg_metric1[2], '1-jc': avg_metric1[3],
        '2-dice': avg_metric2[0], '2-hd95': avg_metric2[1], '2-asd': avg_metric2[2], '2-jc': avg_metric2[3],
        '3-dice': avg_metric3[0], '3-hd95': avg_metric3[1], '3-asd': avg_metric3[2], '3-jc': avg_metric3[3]
    }
    avg_df = pd.DataFrame([avg_row])
    df = pd.concat([df, avg_df], ignore_index=True)

    df.to_csv(write_csv, index=False)

    print("Average per class:")
    print("1:", avg_metric1)
    print("2:", avg_metric2)
    print("3:", avg_metric3)
    print("Overall mean:", avg_all)
    logging.info("Testing Finished!")
    
    # 保存最终结果到 TXT 文件
    summary_path = os.path.join(save_dir, f"{args.exp}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Average per class:\n")
        f.write(f"Class 1 (RV): Dice={avg_metric1[0]:.4f}, HD95={avg_metric1[1]:.4f}, ASD={avg_metric1[2]:.4f}, JC={avg_metric1[3]:.4f}\n")
        f.write(f"Class 2 (Myo): Dice={avg_metric2[0]:.4f}, HD95={avg_metric2[1]:.4f}, ASD={avg_metric2[2]:.4f}, JC={avg_metric2[3]:.4f}\n")
        f.write(f"Class 3 (LV): Dice={avg_metric3[0]:.4f}, HD95={avg_metric3[1]:.4f}, ASD={avg_metric3[2]:.4f}, JC={avg_metric3[3]:.4f}\n\n")
        f.write(f"Overall mean:\n")
        f.write(f"Dice={avg_all[0]:.4f}, HD95={avg_all[1]:.4f}, ASD={avg_all[2]:.4f}, JC={avg_all[3]:.4f}\n")
    writer.close()
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                    default='data/brats_5mm', help='Name of Experiment')
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--dataset', type=str, default='brats', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='output/test/')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=384, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1337, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', default=True,help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='ckpt/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    
    parser.add_argument('--lora_ckpt', type=str, default='output/sam/results_ssl/brats_512_pretrain_vit_b_dualmask_same_prompt_class_random_large_epo1000_bs12_lr0.0001_s1337_20_labeled_brats_5mm_0.35_0.7*0.3_T0.1/best_model.pth', help='The checkpoint from LoRA')
    parser.add_argument('--exp', type=str, default='brats_20_label_0.7*0.3')
    
    parser.add_argument('--vit_name', type=str, default='vit_b_dualmask_same_prompt_class_random_large', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_prompt')
    parser.add_argument('--promptmode', type=str, default='point',help='prompt')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'brats': {
            'Dataset': brats,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions', args.exp)  # 按实验名分类保存
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path=test_save_path)
    

