

model_path="/home/osama/Difix3D/outputs/difix/train/checkpoints/model_1101.pkl"
input_image_path="/data1/hs_denoising/NeRSemble/gsplat_results_f/ava_TCE/renders"


echo "Starting Diffix training..."

python src/inference_difix.py \
    --input_image "${input_image_path}"  \
    --prompt "remove degradation" \
    --output_dir "outputs/difix/test_inference_difix_weights" \
    # --timestep 199