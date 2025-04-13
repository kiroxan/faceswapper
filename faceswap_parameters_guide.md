# Face Swap Quality Parameters Guide

This guide explains the key parameters that influence the quality of face swapping results in both the standard InsightFace method and the ACE_Plus portrait enhancement model.

## Standard InsightFace Method

InsightFace provides a fast and reliable face swapping solution with fewer parameters but still offers good quality results for most use cases.

### Key Parameters:

1. **Input Image Quality**
   - **Resolution**: Higher resolution images (at least 512×512) provide better details.
   - **Lighting**: Consistent lighting between source and target faces improves blending.
   - **Face Angle**: Similar face angles between source and target yield the most natural results.

2. **Mask Integration** (optional)
   - **Purpose**: Defines blend regions between the swapped face and the original image.
   - **Quality Impact**: Precise masks can significantly improve edge transitions and blending.
   - **Usage**: White areas in the mask indicate where the swapped face should be fully applied.

3. **GFPGAN Enhancement**
   - **Purpose**: Additional face restoration to improve details and quality.
   - **When Useful**: Particularly effective for lower quality or lower resolution inputs.
   - **Limitations**: May occasionally over-smooth face details or change facial characteristics.

## ACE_Plus Portrait Model

The ACE_Plus portrait model leverages Stable Diffusion with a specialized LoRA model to generate higher quality results, especially for portrait photos. It offers more parameters for fine-tuning the outcome.

### Key Parameters:

1. **LoRA Strength** (`lora_strength`)
   - **Range**: 0.0 to 1.0 (default: 0.7)
   - **Effect**: Controls how strongly the ACE_Plus portrait model influences the result.
   - **Higher Values** (0.7-1.0): More idealized, higher quality faces but potentially less resemblance to the source face.
   - **Lower Values** (0.3-0.6): Better preservation of the original face characteristics but fewer enhancements.
   - **Recommendation**: Start with 0.7 and adjust based on results.

2. **Guidance Scale** (`guidance_scale`)
   - **Range**: 1.0 to 15.0 (default: 7.5)
   - **Effect**: Controls how closely the result follows the text prompt.
   - **Higher Values** (8.0-12.0): More precise adherence to the prompt description but potentially more artifacts.
   - **Lower Values** (4.0-7.0): More natural images but less adherence to specific prompt details.
   - **Recommendation**: Values between 6.0-8.0 provide a good balance for portrait generation.

3. **Number of Inference Steps** (`num_inference_steps`)
   - **Range**: 20 to 50 (default: 30)
   - **Effect**: Controls the refinement of the generated image.
   - **Higher Values** (40-50): More detailed and refined results but longer processing time.
   - **Lower Values** (20-25): Faster processing but potentially less detailed.
   - **Recommendation**: 30 steps provides a good balance between quality and speed.

4. **Prompt Engineering** (`prompt` and `negative_prompt`)
   - **Positive Prompt**: Guides the model toward desired characteristics.
     - Example: `"a portrait photo of person, highly detailed face, clear eyes, perfect face"`
   - **Negative Prompt**: Steers the model away from undesired characteristics.
     - Example: `"blurry, low quality, disfigured face, bad eyes, bad nose, bad ears, bad mouth, bad teeth"`
   - **Quality Impact**: Well-crafted prompts significantly improve results.
   - **Recommendation**: Be specific about desired facial features and portrait style.

5. **Seed** (`seed`)
   - **Purpose**: Controls the randomness in the generation process.
   - **Effect**: Same seed with identical parameters produces the same result.
   - **Usage**: Set a specific seed to reproduce good results or make small adjustments.

6. **Initial Face Swap** (`use_initial_face_swap`)
   - **Purpose**: Determines whether to use InsightFace for initial face swapping before applying ACE_Plus.
   - **Default**: `true` (recommended)
   - **Quality Impact**: Using the initial face swap typically produces better identity preservation.

## Best Practices for Optimal Results

### For Standard InsightFace:

1. **Face Selection**: When multiple faces are detected, the first face is used. Position the primary face prominently.
2. **Resolution Matching**: Try to use source and target images with similar resolutions.
3. **Face Angles**: Match the head pose and facial expression between source and target.

### For ACE_Plus Portrait:

1. **Parameter Combinations**:
   - For identity preservation: Lower LoRA strength (0.4-0.6), moderate guidance scale (6.0-7.0)
   - For ideal portrait quality: Higher LoRA strength (0.7-0.9), higher guidance scale (7.5-9.0)

2. **Prompt Engineering Examples**:
   - For professional headshots: `"professional headshot of person, studio lighting, neutral background, sharp features, clear eyes"`
   - For artistic portraits: `"artistic portrait of person, dramatic lighting, detailed features, cinematic"`

3. **Hardware Considerations**:
   - ACE_Plus requires significantly more GPU memory than standard InsightFace
   - At least 6GB VRAM recommended for reasonable processing time
   - Processing time: 30-60 seconds on average vs. 1-3 seconds for standard InsightFace

## Comparison of Parameter Influence

| Parameter | InsightFace Impact | ACE_Plus Impact |
|-----------|-------------------|-----------------|
| Input Resolution | High | Medium (enhanced by model) |
| Face Angle | High | Medium-High |
| Lighting | High | Low-Medium (corrected by model) |
| LoRA Strength | N/A | Very High |
| Guidance Scale | N/A | High |
| Inference Steps | N/A | Medium-High |
| Prompts | N/A | Very High |
| Seed | N/A | Medium |

## Example Parameter Sets for Common Use Cases

### Professional Headshot Enhancement
```
use_ace: true
lora_strength: 0.8
guidance_scale: 7.5
num_inference_steps: 35
prompt: "professional corporate headshot, neutral background, studio lighting, sharp features"
```

### Natural Portrait Preservation
```
use_ace: true
lora_strength: 0.5
guidance_scale: 6.0
num_inference_steps: 30
prompt: "natural portrait of person, realistic skin texture, detailed features"
```

### Artistic Portrait Transformation
```
use_ace: true
lora_strength: 0.9
guidance_scale: 8.5
num_inference_steps: 40
prompt: "artistic portrait photo, dramatic lighting, detailed face, cinematic look"
``` 