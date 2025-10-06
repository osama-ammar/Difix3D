import os
import time

import cv2
import numpy as np
import torch
import torch_tensorrt
import torchvision.transforms.functional as F
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import load_image
from src.model import Difix
from PIL import Image
from src.pipeline_difix import DifixPipeline


def run_pipeline(pipe, input_image, _ref_image=None):

    output_image = pipe(
        image=input_image,
        ref_image=_ref_image,
        num_inference_steps=1,
        timesteps=[199],
    ).images[0]

    return output_image


def run_folder():
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    base_dir = "data/render/"
    img_paths = os.listdir(base_dir)

    for i in img_paths:
        input_image = load_image(f"{base_dir}{i}")
        output_image = run_pipeline(pipe, input_image)
        # Save or display the output image as needed
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image = output_image * 255
        output_image = output_image.astype(np.uint8)
        cv2.imwrite(f"{base_dir}/output_difix_{i}", output_image)


def test_pipeline():
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    base_dir = "data/woman/iteration_29000/"
    input_image = load_image(f"{base_dir}render.jpg")
    ref_image = load_image(f"{base_dir}gt_image.jpg")
    # input_image = input_image.resize((512, 512))

    for _ in range(1):
        output_image = run_pipeline(pipe, input_image, ref_image)

    # Convert PIL image to OpenCV format
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)

    output_image_no = run_pipeline(pipe, input_image, None)

    cv2.imshow("Output Image", output_image)
    output_image_no = cv2.cvtColor(output_image_no, cv2.COLOR_RGB2BGR)
    output_image_no = output_image_no * 255
    output_image_no = output_image_no.astype(np.uint8)
    cv2.imwrite(f"{base_dir}/output_difix.png", output_image_no)
    cv2.imshow("Output Image No Ref", output_image_no)

    diff = cv2.absdiff(output_image, output_image_no)
    diff = diff.max(axis=2).astype(np.uint8)
    print(diff.max())
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)


@torch.no_grad()
def test_model():
    net_difix = Difix(
        lora_rank_vae=4, timestep=199, mv_unet=False, pretrained_path="difix.ckpt"
    )

    net_difix = net_difix.to("cuda")
    net_difix.eval()
    net_difix.set_eval()
    net_difix = net_difix.half()
    # net_difix = torch.compile(net_difix)
    print("Model loaded")

    img = Image.open("data/woman/iteration_28000/render.jpg")
    ref = Image.open("data/woman/iteration_28000/gt_image.jpg")
    print(img.size)
    # img_t = F.to_tensor(img)
    # img_t = F.resize(img_t, (512, 512))
    # img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
    # test_input = img_t.unsqueeze(0).unsqueeze(0).to("cuda")
    img = img.resize((512, 512))
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    test_input = image_processor.preprocess(img)
    test_ref = image_processor.preprocess(ref)
    test_input = test_input.unsqueeze(0).to("cuda")
    test_ref = test_ref.unsqueeze(0).to("cuda")

    test_input = test_input.half()
    test_ref = test_ref.half()

    # test_input = torch.randn(1, 1, 3, 512, 512, device="cuda")

    def run_model():
        return net_difix(torch.concat([test_input, test_ref], dim=0))

    # for _ in range(10):
    #     run_model()

    for _ in range(1):
        output = run_model()

    # Convert output to numpy and show
    # output = output.squeeze().cpu().numpy()
    output = output.squeeze(0)
    print(output.shape)

    output = image_processor.postprocess(
        output[0], output_type="np", do_denormalize=[True]
    )

    output = output[0]

    # # Denormalize
    # output = (output * 0.5) + 0.5

    # output = np.transpose(output, (1, 2, 0))
    # output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

    

if __name__ == "__main__":
    # test_pipeline()
    # test_model()
    run_folder()
