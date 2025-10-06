import json
import torch
from PIL import Image
import torchvision.transforms.functional as F


# import numpy as np
# from torchvision import transforms

# def save_tensor_as_png(tensor, filename):
#     """
#     Save a tensor as PNG image
#     """
#     # Convert tensor to PIL Image
#     to_pil = transforms.ToPILImage()
    
#     # Handle different tensor formats
#     if tensor.dim() == 4:  # [B, C, H, W] - batch
#         # Save first image in batch
#         img = to_pil(tensor[0])
#     elif tensor.dim() == 3:  # [C, H, W] - single image
#         img = to_pil(tensor)
#     elif tensor.dim() == 2:  # [H, W] - grayscale
#         img = to_pil(tensor.unsqueeze(0))  # Add channel dimension
#     else:
#         raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
#     # Save image
#     img.save(filename)
#     print(f"Saved: {filename}")
    
    
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"] 
        if output_img.endswith(".avif"):
            mask_path = self.data[img_id]["target_image"].replace("images" ,"masks").replace(".avif" ,".png")
        else:
            mask_path = self.data[img_id]["target_image"].replace("images-2fps" ,"masks")
            
        # print(f"input_img : {input_img}")
        # print(f"output_img : {output_img}")
        # print(f'mask : {mask_path}')
        try:
            input_img = Image.open(input_img)
            output_img = Image.open(output_img)
            mask_img =  Image.open(mask_path)

        except Exception as e:
            print("Error loading image:", input_img, output_img,e)
            # print(what_is_this)
            
            return self.__getitem__(idx + 1)
        

        img_t = F.to_tensor(input_img)        
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        mask_t = F.to_tensor(mask_img)
        mask_t = F.resize(mask_t, self.image_size)
        mask_t = mask_t.expand_as(output_t)
        mask_t = (mask_t>0.5).float()
        output_t[mask_t==0]=1.0


        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids
        return out
