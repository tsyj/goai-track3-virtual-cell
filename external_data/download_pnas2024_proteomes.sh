#!/bin/bash
# 诊断实验 87 使用的公开数据（最终模型不使用）：PNAS 2024 942 株酵母蛋白质组，CC-BY-4.0
# Zenodo 10.5281/zenodo.10567083 → data_upload.zip（121,876,xxx 字节）
set -e; mkdir -p data/external/pnas2024 && cd data/external/pnas2024
curl -L -o data_upload.zip 'https://zenodo.org/records/10567083/files/data_upload.zip?download=1'
sha256sum data_upload.zip; unzip -qo data_upload.zip -d data_upload
ls data_upload/data_upload/DIA-NN_1.8/SCmedia_MBR_CommonPeptide_Approach/*DetectionThreshold30_genes_ORF*.tsv
