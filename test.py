from src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image



model_path= "/home/osama/Difix3D/outputs/difix/train/checkpoints/model_1101.pkl"

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("/data1/hs_denoising/NeRSemble/gsplat_results_f/ava_TCE/renders/val_step4999_cam401875.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")