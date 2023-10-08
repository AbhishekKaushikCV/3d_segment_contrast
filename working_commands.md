# Working Commands

- ## Inference on SemanticKITTI:
```
python3 inference_vis.py --checkpoint segment_contrast_1p0 --data-dir /workspace/dataset/SemanticKITTI --sparse-model MinkUNet --dataset-name SemanticKITTI --best epoch14 --log-dir /workspace/semseg_1p0 --use-intensity --sparse-resolution 0.05 --use-cuda |& tee inference_segcontrast_1p0.txt
``` 

- ## Inference on SemanticSlamantic:
```
 python3 inference_target.py --checkpoint segment_contrast_1p0 --data-dir /workspace/dataset/SLAMANTIC/ --sparse-model MinkUNet --dataset-name SemanticSlamantic --best epoch14 --log-dir /workspace/semseg_1p0 --use-intensity --sparse-resolution 0.05 --use-cuda --inference-dir /workspace/inference/ --write-pcd |& tee inference_segcontrast_1p0_slamantic.txt
```