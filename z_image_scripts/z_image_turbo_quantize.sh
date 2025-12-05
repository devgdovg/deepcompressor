TORCH_CUDA_ARCH_LIST="9.0" python3 -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/z-image-turbo-rank128-skip-refiners.yaml examples/diffusion/configs/svdquant/int4.yaml \
    --save-model /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251204_1646 --copy-on-save true --skip-eval true

echo $?

# nohup z_image_scripts/z_image_turbo_quantize.sh > z_image_scripts/z_image_turbo_quantize_20251204_1646.log 2>&1 &