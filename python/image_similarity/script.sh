#!/bin/bash

# python image_similarity.py \
#     --mse --rmse --psnr \
#     --img_a "./test_images/test_lower_res_image_1.JPG" \
#     --img_b "./test_images/test_lower_res_image_2.JPG"

python image_similarity.py \
    --all \
    --verbose \
    --img_a "./test_images/test_low_res_image_1.JPG" \
    --img_b "./test_images/test_low_res_image_2.JPG"
