#!/bin/bash
usage() {
  echo "Usage: ${0} [-g|--gpu] [-c|--case]  [-lr|--learning_rate]  [-lr_geo|--learning_rate_geo]"  1>&2
  exit 1
}
while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -c|--case)
      CASE=${2}
      shift 2
      ;;
    -g|--gpu)
      GPU=${2}
      shift 2
      ;;
    -lr|--learning_rate)
      LR=${2}
      shift 2
      ;;
    -lr_geo|--learning_rate_geo)
      LR_GEO=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done

CUDA_VISIBLE_DEVICES=${GPU} python exp_runner_blending.py --conf ./confs/udf_dtu_blending_ft.conf \
--case ${CASE} --threshold 0.005 --resolution 128  --is_continue --is_finetune