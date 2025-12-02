TORCH_CUDA_ARCH_LIST="8.9" python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo.yaml examples/diffusion/configs/svdquant/int4.yaml \
    --save-model /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251202_2007 --skip-eval true

echo $?

# nohup z_image_scripts/z_image_turbo_quantize.sh > z_image_scripts/z_image_turbo_quantize_20251202_2007.log 2>&1 &