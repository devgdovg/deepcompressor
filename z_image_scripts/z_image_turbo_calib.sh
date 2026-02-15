TORCH_CUDA_ARCH_LIST="9.0" python3 -m deepcompressor.app.diffusion.dataset.collect.calib \
    examples/diffusion/configs/model/z-image-customized-rank256.yaml examples/diffusion/configs/collect/qdiff.yaml

echo $?

# nohup z_image_scripts/z_image_turbo_calib.sh > z_image_scripts/z_image_turbo_calib_20251217_1931.log 2>&1 &