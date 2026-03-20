# MLSM-within-UNet-Dense-Skip-Connections-for-Mapping-of-Retrogressive-Thaw-Slumps
MLSM is designed as a lightweight plug-and-play aggregation block that can be embedded into the dense skip pathways of a UNet++ backbone to improve non-local feature modeling, structural consistency, and boundary delineation in RTS segmentation.


# MLSM for RTS Mapping

This repository provides the complete executable implementation of the **MLSM (Multi-Level Self-Modulation)** plug-and-play module proposed in our ISPRS 2026 Congress paper:

**Self-Modulation Aggregation within Dense Skip Connections for Mapping of Retrogressive Thaw Slumps**

The released code includes the standalone MLSM module and a simple test demo for quick verification and integration. As described in the paper, MLSM is designed as a lightweight plug-and-play aggregation block that can be embedded into the dense skip pathways of a UNet++ backbone to improve non-local feature modeling, structural consistency, and boundary delineation in RTS segmentation.

#Contents

- mlsm.py: standalone executable implementation of the MLSM module  
- unetplusplus_mlsm.py: UNet++ backbone with MLSM integrated into dense skip connections  
- mlsm_demo.py: built-in test demo for module checking and shape validation  

# Notes

This repository mainly provides the **MLSM plug-and-play module** and its core network implementation.  
All data preprocessing procedures and the training/testing sample partition strategy used in this study are consistent with those adopted for the comparative methods, and the corresponding files can be found in the **AmRTSNet_v51** file under this GitHub account.
# Citation

If this code is helpful for your research, please cite our paper and clearly acknowledge the source.

Paper title:
Self-Modulation Aggregation within Dense Skip Connections for Mapping of Retrogressive Thaw Slumps

A BibTeX entry can be added here after the final publication information is available.

# Contact

If you have any questions, suggestions, or find any issues in the code, please feel free to contact us by email.

Thank you very much for your interest and feedback.
