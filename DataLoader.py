
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random


class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num
        self.data_path = data_path

        # json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        # dataset = json.load(json_file)
        _label_json = os.environ.get("SAMMED2D_LABEL_JSON", f'/mnt/hdd2/task2/sam-med2d/label2image_{mode}.json')
        dataset = json.load(open(_label_json, "r"))
        print(f"Loaded dataset with {len(dataset)} samples from {self.data_path} for mode {mode}.")
        
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
        self.pixel_mean = [94.01123382560912, 57.77812151883644, 53.55980543966791] # [123.675, 116.28, 103.53]
        self.pixel_std = [79.134414081972, 60.63022484441235, 57.946300608015605] # [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(os.path.join(f'{self.data_path}/images', self.image_paths[index]))
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(os.path.join(f'{self.data_path}/images', self.image_paths[index]))

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask, max_pixel = 0)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [94.01123382560912, 57.77812151883644, 53.55980543966791] # [123.675, 116.28, 103.53]
        self.pixel_std = [79.134414081972, 60.63022484441235, 57.946300608015605] # [58.395, 57.12, 57.375]

        # dataset = json.load(open(os.path.join(data_dir, f'/mnt/hdd2/task2/sam-med2d/image2label_{mode}.json'), "r"))
        dataset = json.load(open(f'/mnt/hdd2/task2/sam-med2d/image2label_{mode}.json', "r"))
        print(f"Loaded dataset with {len(dataset)} samples from /mnt/hdd2/task2/sam-med2d/image2label_{mode}.json for mode {mode}.")
        
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        
        # 使用random.choices进行有放回抽样，即使实际mask数量不足mask_num，也能通过重复采样补足数量
        # 实际mask数量 > mask_num时，随机选择子集
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            # 应用相同的数据增强（保证image-mask对齐）
            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        # 将多个mask堆叠为 [mask_num, H, W] 张量
        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        
        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1) # mask张量 [mask_num, 1, H, W] 
        image_input["boxes"] = boxes # 边界框 [mask_num, 4]
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        

        # 可视化代码---------------------------------------------
        # visualize = True  # 配置开关，是否进行可视化
        # if visualize:  # 添加配置开关
        #     save_path = "/mnt/hdd2/task2/sam-med2d/plots"
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
                
        #     # 修复：将均值和标准差转换为张量
        #     pixel_mean_tensor = torch.tensor(self.pixel_mean, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
        #     pixel_std_tensor = torch.tensor(self.pixel_std, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
            
        #     # 反归一化还原图像
        #     img_vis_tensor = image_tensor * pixel_std_tensor + pixel_mean_tensor
        #     img_vis = img_vis_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
        #     # 确保颜色通道顺序正确 (OpenCV使用BGR)
        #     img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            
        #     # 保存原始图像
        #     cv2.imwrite(f"{save_path}/{index}_original.png", img_vis)
            
        #     # 为每个mask单独生成可视化图像
        #     for i, (m, box_tensor, points, labels) in enumerate(zip(
        #         masks_list, boxes_list, point_coords_list, point_labels_list
        #     )):
        #         # 保存纯mask图
        #         mask_np = m.cpu().numpy().astype(np.uint8) * 255
        #         cv2.imwrite(f"{save_path}/{index}_mask_{i}.png", mask_np)
                
        #         # 创建叠加图像副本
        #         combined_img = img_vis.copy()
                
        #         # 处理边界框
        #         if box_tensor.dim() == 2 and box_tensor.shape[0] == 1 and box_tensor.shape[1] == 4:
        #             box = box_tensor[0].cpu().numpy().astype(int)
        #         elif box_tensor.dim() == 1 and box_tensor.numel() == 4:
        #             box = box_tensor.cpu().numpy().astype(int)
        #         else:
        #             print(f"Warning: Invalid box shape {box_tensor.shape} for mask {i}. Skipping box.")
        #             box = None
                
        #         # 绘制边界框 (绿色)
        #         if box is not None:
        #             cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # 绿色
                
        #         # 绘制点 (前景点蓝色，背景点红色)
        #         points = points.cpu().numpy().astype(int) if points.device != 'cpu' else points.numpy().astype(int)
        #         labels = labels.cpu().numpy() if labels.device != 'cpu' else labels.numpy()
                
        #         for pt, label in zip(points, labels):
        #             if label == 1:  # 前景点
        #                 cv2.circle(combined_img, tuple(pt), 5, (255, 0, 0), -1)  # 蓝色
        #             else:  # 背景点
        #                 cv2.circle(combined_img, tuple(pt), 5, (0, 0, 255), -1)  # 红色
                
        #         # 添加半透明mask (红色)
        #         mask_color = np.zeros_like(combined_img)
        #         mask_color[:, :] = (0, 0, 255)  # 红色 (BGR格式)
        #         mask_area = mask_np[:, :, np.newaxis] > 0  # 创建布尔掩码
        #         combined_img = np.where(mask_area, 
        #                                cv2.addWeighted(combined_img, 0.7, mask_color, 0.3, 0),
        #                                combined_img)
                
        #         # 保存叠加后的图像
        #         cv2.imwrite(f"{save_path}/{index}_combined_{i}.png", combined_img)


    
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
        
    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

