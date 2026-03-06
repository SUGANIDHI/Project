# StripUnetMCSA - Detailed Model Architecture

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Encoder: ResNet50](#encoder-resnet50)
4. [Decoder: Dual-Branch Architecture](#decoder-dual-branch-architecture)
5. [MCSA: Multi-Context Spatial Attention](#mcsa-multi-context-spatial-attention)
6. [Data Flow](#data-flow)
7. [Technical Specifications](#technical-specifications)
8. [Implementation Details](#implementation-details)

---

## Overview

**StripUnetMCSA** is a state-of-the-art semantic segmentation model designed specifically for road extraction from satellite and aerial imagery. It combines the proven U-Net encoder-decoder architecture with advanced attention mechanisms to achieve high-precision road segmentation.

### Key Innovations
- **Dual-Branch Decoder**: Parallel processing paths for capturing both local details and broader context
- **Multi-Context Spatial Attention (MCSA)**: Novel attention mechanism with multi-scale receptive fields
- **Skip Connections**: U-Net style feature fusion for precise spatial localization
- **ResNet50 Backbone**: Pre-trained ImageNet encoder for robust feature extraction

### Performance Metrics
| Metric | Value |
|--------|-------|
| Validation F1 Score | 0.778 |
| IoU (Intersection over Union) | 0.644 |
| Total Parameters | 40.4 Million |
| Training Epochs | 25 |

---

## Architecture Components

```

                        StripUnetMCSA Architecture                        

                                                                         
  Input Image (3HW)                                                    
                                                                        
                                                                        
                            
             ENCODER (ResNet50)                                        
                                          
     e1   e2   e3   e4   e5                             
     64     256    512   1024   2048                           
                                          
                            
                                                                     
          Skip Connections                                             
                                                                     
                            
             DECODER (Dual-Branch)                                      
                                                                        
    dec4  MCSA4  dec3  MCSA3  dec2  ...                           
    512           256           128                                  
                            
                                                                        
                                                                        
  Output Mask (1HW)                                                    
                                                                         

```

---

## Encoder: ResNet50

The encoder is based on **ResNet50**, a proven convolutional neural network architecture pre-trained on ImageNet with 1000+ classes. This provides robust, generalizable feature extraction capabilities.

### Encoder Stages

| Stage | Layer | Output Channels | Spatial Resolution | Description |
|-------|-------|-----------------|-------------------|-------------|
| Initial | conv1 + bn1 + relu | 64 | H/2  W/2 | 77 convolution, stride 2 |
| e1 | maxpool | 64 | H/4  W/4 | 33 max pooling, stride 2 |
| e2 | layer1 | 256 | H/4  W/4 | 3 Bottleneck blocks |
| e3 | layer2 | 512 | H/8  W/8 | 4 Bottleneck blocks |
| e4 | layer3 | 1024 | H/16  W/16 | 6 Bottleneck blocks |
| e5 | layer4 | 2048 | H/32  W/32 | 3 Bottleneck blocks (bottleneck) |

### ResNet50 Bottleneck Block
```
Input (C channels)
    
    
                               (identity shortcut)
                              
                    
 11 Conv   C/4 channels    
 BatchNorm                    
 ReLU                         
                    
                               
                    
 33 Conv   C/4 channels    
 BatchNorm                    
 ReLU                         
                    
                               
                    
 11 Conv   C channels      
 BatchNorm                    
                    
                               
                               
   (+) 
     
     
   ReLU
     
Output (C channels)
```

---

## Decoder: Dual-Branch Architecture

The decoder features a novel **Dual-Branch** design that processes features through two parallel pathways before fusion.

### DualBranchDecoderBlock Structure

```
Input (in_channels)
         
    
             
             
 
Dilated Standard
Branch  Branch  
 
             
             
 
33 Conv 33 Conv
BatchNorm BatchNorm
  ReLU      ReLU   
 
             
    
         
    Concatenate
         
         
    
    11 Conv   Fusion Layer
    
         
Output (out_channels  2)
```

### Decoder Stages with Channel Progression

| Stage | Input | Skip Connection | Total Input | Output | MCSA Applied |
|-------|-------|-----------------|-------------|--------|--------------|
| dec4 | e5 (2048) | - | 2048 | 512 |  |
| dec3 | d4_up (512) | e4 (1024) | 1536 | 256 |  |
| dec2 | d3_up (256) | e3 (512) | 768 | 128 |  |
| dec1 | d2_up (128) | e2 (256) | 384 | 64 |  |
| dec0 | d1_up (64) | e1 (64) | 128 | 32 |  |
| final | d0_up (32) | - | 32 | 1 |  |

### Why Dual-Branch?
1. **Multi-Scale Context**: Dilated branch captures broader spatial context
2. **Fine Details**: Standard branch preserves local spatial information
3. **Feature Fusion**: 11 convolution learns optimal combination of both paths
4. **Robustness**: Parallel paths provide redundancy and improved gradient flow

---

## MCSA: Multi-Context Spatial Attention

The **Multi-Context Spatial Attention (MCSA)** module is the key innovation that enables the model to focus on road-relevant features while suppressing background noise.

### MCSA Architecture

```
Input Feature Map (C channels)
              
    
                               
                               
   
11 Conv 33 Conv 33 Conv 33 Conv
  C/4      C/4      C/4      C/4   
         dilation dilation dilation
            =1       =2       =3   
   
                               
    
                       
    Concatenate (C channels)
         
         
    
    11 Conv   1 channel
     Sigmoid 
    
         
    Attention Map (1HW)
         
         
   Element-wise  Input
         
    Output (C channels)
```

### MCSA Components

| Component | Kernel Size | Dilation | Receptive Field | Purpose |
|-----------|-------------|----------|-----------------|---------|
| conv1x1 | 11 | 1 | 11 | Point-wise features |
| conv3x3 | 33 | 1 | 33 | Local spatial features |
| conv3x3_d2 | 33 | 2 | 55 | Medium-range context |
| conv3x3_d3 | 33 | 3 | 77 | Long-range context |

### Attention Mechanism
1. **Multi-Scale Feature Extraction**: Four parallel convolutions with different receptive fields
2. **Feature Aggregation**: Concatenation preserves all multi-scale information
3. **Spatial Attention**: 11 convolution + Sigmoid generates pixel-wise attention weights
4. **Feature Modulation**: Input features are multiplied by attention weights

---

## Data Flow

### Complete Forward Pass

```python
# 1. ENCODER
x = conv1(input)           # 3  64, H/2
x = bn1(x)
x = relu(x)
e1 = maxpool(x)            # 64, H/4

e2 = layer1(e1)            # 64  256, H/4
e3 = layer2(e2)            # 256  512, H/8
e4 = layer3(e3)            # 512  1024, H/16
e5 = layer4(e4)            # 1024  2048, H/32 (bottleneck)

# 2. DECODER
d4 = dec4(e5)              # 2048  512
d4 = mcsa4(d4)             # Attention refinement
d4_up = upsample(d4, e4)   # Upsample to e4 size

d3 = dec3(concat(d4_up, e4))  # 1536  256
d3 = mcsa3(d3)
d3_up = upsample(d3, e3)

d2 = dec2(concat(d3_up, e3))  # 768  128
d2 = mcsa2(d2)
d2_up = upsample(d2, e2)

d1 = dec1(concat(d2_up, e2))  # 384  64
d1 = mcsa1(d1)
d1_up = upsample(d1, e1)

d0 = dec0(concat(d1_up, e1))  # 128  32
d0_up = upsample(d0, 4x)      # Final 4 upsampling

# 3. OUTPUT
output = final(d0_up)         # 32  1 (binary mask)
```

### Feature Map Sizes (for 512512 input)

| Layer | Spatial Size | Channels |
|-------|--------------|----------|
| Input | 512512 | 3 |
| e1 | 128128 | 64 |
| e2 | 128128 | 256 |
| e3 | 6464 | 512 |
| e4 | 3232 | 1024 |
| e5 | 1616 | 2048 |
| d4 | 1616 | 512 |
| d3 | 3232 | 256 |
| d2 | 6464 | 128 |
| d1 | 128128 | 64 |
| d0 | 128128 | 32 |
| Output | 512512 | 1 |

---

## Technical Specifications

### Model Configuration

```python
StripUnetMCSA(
    in_channels=3,          # RGB input
    out_channels=1,         # Binary segmentation
    pretrained=True         # ImageNet pre-training
)
```

### Layer Count
| Component | Layers |
|-----------|--------|
| Encoder (ResNet50) | 53 layers |
| Decoder Blocks (5) | 30 layers |
| MCSA Modules (4) | 24 layers |
| Final Layer | 1 layer |
| **Total** | **~108 layers** |

### Parameter Distribution
| Component | Parameters | Percentage |
|-----------|------------|------------|
| Encoder | 23.5M | 58.2% |
| Decoder | 14.8M | 36.6% |
| MCSA | 2.0M | 5.0% |
| Final | 0.1M | 0.2% |
| **Total** | **40.4M** | **100%** |

### Memory Requirements
| Phase | Memory (GPU) | Memory (CPU) |
|-------|--------------|--------------|
| Model Loading | ~160 MB | ~160 MB |
| Inference (512512) | ~1.5 GB | ~1.5 GB |
| Batch Inference (4) | ~4 GB | N/A |

---

## Implementation Details

### Loss Function
- **Binary Cross-Entropy with Logits** for training stability
- **Dice Loss** component for handling class imbalance (roads vs background)

### Optimization
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 8-16 (depending on GPU memory)

### Data Augmentation
- Random horizontal/vertical flips
- Random rotation (15)
- Color jittering
- Random scaling (0.8-1.2)

### Inference Pipeline
1. **Preprocessing**: RGB conversion, ImageNet normalization
2. **Tiling**: Large images split into 512512 overlapping tiles
3. **Forward Pass**: Each tile processed through the model
4. **Post-processing**: 
   - Sigmoid activation
   - Tile stitching with overlap blending
   - Binarization at threshold 0.5

### Checkpoint Format
```python
# Checkpoint structure
{
    'encoder.conv1.weight': Tensor,
    'encoder.bn1.weight': Tensor,
    'encoder.layer1.0.conv1.weight': Tensor,
    # ... ~412 total keys
    'dec4.dilated.0.weight': Tensor,
    'mcsa4.conv1x1.weight': Tensor,
    'final.weight': Tensor,
}
```

---

## Code Reference

The full implementation is available in:
- **Model Definition**: [`backend/model_loader.py`](backend/model_loader.py)
- **Inference**: [`backend/inference.py`](backend/inference.py)
- **Configuration**: [`backend/config.py`](backend/config.py)

---

*StripUnetMCSA - Advancing Road Segmentation with Dual-Branch Attention*
