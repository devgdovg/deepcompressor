echo "-quant-path /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251217_2034"

TORCH_CUDA_ARCH_LIST="9.0" python -m deepcompressor.backend.nunchaku.convert \
  --quant-path /data/dongd/dc_saved_model/Z_IMAGE_TURBO_20251217_2034 \
  --output-root /data/dongd/dc_converted_model/Z_IMAGE_CUSTOM_20251217_2034_r256_int4 \
  --model-name z-image-customized \
  # --float-point \


echo $?

# nohup z_image_scripts/z_image_turbo_convert.sh > z_image_scripts/z_image_turbo_convert_20251218_0750.log 2>&1 &