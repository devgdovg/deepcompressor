echo "-quant-path /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251204_0743"

TORCH_CUDA_ARCH_LIST="9.0" python -m deepcompressor.backend.nunchaku.convert \
  --quant-path /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251204_0743 \
  --output-root /data/dongd/dc_converted_model/Z_IMAGE_TURBO_20251204_0743_r128 \
  --model-name z-image-turbo \


echo $?

# nohup z_image_scripts/z_image_turbo_convert.sh > z_image_scripts/z_image_turbo_convert_20251205_2244.log 2>&1 &