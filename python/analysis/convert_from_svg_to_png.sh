#!/bin/bash

dpi=400

inkscape svg/astronomy/ergas_astronomy_compound.svg -d $dpi -e png/ergas_astronomy_compound.png;
inkscape svg/astronomy/mse_astronomy_compound.svg -d $dpi -e png/mse_astronomy_compound.png;
inkscape svg/astronomy/psnr_astronomy_compound.svg -d $dpi -e png/psnr_astronomy_compound.png;
inkscape svg/astronomy/rmse_astronomy_compound.svg -d $dpi -e png/rmse_astronomy_compound.png;

inkscape svg/mri/ergas_mri_compound.svg -d $dpi -e png/ergas_mri_compound.png;
inkscape svg/mri/mse_mri_compound.svg -d $dpi -e png/mse_mri_compound.png;
inkscape svg/mri/psnr_mri_compound.svg -d $dpi -e png/psnr_mri_compound.png;
inkscape svg/mri/rmse_mri_compound.svg -d $dpi -e png/rmse_mri_compound.png;