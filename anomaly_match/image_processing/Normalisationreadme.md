[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Image Normalisation
Normalisation is applied during image loading. This allows to retain high dynamic range information from images and work with uint8s internally. This is especially recommended to apply when working with fits files or other high dynamic range inputs.

For the available classes see [NormalisationMethod](NormalisationMethod.py) and their respective implementation in [normalise()](normalisation.py)

Currently (V1.1.0), when evaluating all files the top files are saved ino a numpy array with the selected normalisation.
When loading those top files with the "Load Top Files" button, they will be displayed with the normalisation applied during evaluation.

Conversion only: Behaviour depends on the image dtype
    uint8: No scaling
    uint16: rescaling to uint8 by dividing /256
    float32: rescaling to uint8 by a linear min/max stretch

Log: Applies a logstretch to an interval of
    minimum: minimum of array, 0 if negative
    maximum: maximum of array

ZScale:
    Applies a linear stretch between a minimum and maximum derived
    from a zscale.

When evaluating all files, the top files are saved as a NumPy array using the selected normalisation.
When loading these top files with the "Load Top Files" button, they are displayed with the same normalisation applied during evaluation.

## Normalisation Methods
### Conversion_Only:
"No" normalisation applied
 - Behaviour depends on the image data type:
    - uint8: No scaling is applied.
    - uint16: Rescaled to uint8 by dividing by 256 and rounding.
    - float32 & other: Rescaled to uint8 using a linear min/max stretch.

### Log:
Applies a logarithmic stretch.
- Minimum: The minimum value of the array (or 0 if negative).
- Maximum: The maximum value of the array.

### ZScale:
Applies a linear stretch.
 - The minimum and maximum are determined using the zscale algorithm.

### Asinh
Applies an asinh stretch 
 - First: clipping based on the set min/max, then clipping symmetrically by percentile each channel and then dividing with the asinh_scale cfg parameter, and normalising
 - Minimum and Maximum are determined based on the config parameters described below


## Channel Combination

When loading FITS files with multiple extensions, `cfg.normalisation.channel_combination` defines how extensions are linearly combined into output channels before normalisation is applied.

- **Type:** NumPy array of shape `(n_output_channels, n_extensions)`, or `None`
- **When needed:** When you have more FITS extensions than desired output channels (e.g. 4 extensions -> 3 RGB channels)
- **Default:** `None` — extensions map directly to channels one-to-one
- **Order of operations:** Channel combination is applied first, then normalisation

Each row of the array defines one output channel as a weighted sum of the input extensions. For example, with 4 extensions and 3 output channels:

```python
import numpy as np
cfg.normalisation.channel_combination = np.array([
    [1, 0, 0, 0],      # Channel 0 = extension 0
    [0, 0.5, 0.5, 0],  # Channel 1 = average of extensions 1 and 2
    [0, 0, 0, 1],      # Channel 2 = extension 3
])
```

## Normalisation settings (optional)
- cfg.normalisation.maximum_value (None, float, default: None)
    - set the upper clipping value for normalisation, overwrites other settings

- cfg.normalisation.minimum_value (None, float, default: None)
    - set the lower clipping value for normalisation, overwrites other settings

- cfg.normalisation.crop_for_maximum_value (None, Tuple, default: None) 
    - (a,b) where a and b are ints that determine the amount of pixels to crop each direction around the center to compute the max_value
    - intended to avoid that faint objects are incorrectly normalised when bright objects exist in the image outskirts

- cfg.normalisation.log_calculate_minimum_value (bool, defaul:False)
    - if set to true calculates minimum value for the log scaling or uses cfg.normalisation.minimum_value
    - if set to false minimum value is set to 0 or cfg.normalisation.minimum_value

- cfg.normalisation.asinh_clip (list of 3 floats/float, default:\[99.0,99.0,99.0\])
    - applies channel wise (RGB) upper limit clipping to the images based on the percentile
    - if set as a float applies the same clipping to all channels
    - must be 0< float <= 100

- cfg.normalisation.asinh_stretch (list of 3 floats, default:\[10.0,10.0,10.0\])
    - applies channel wise stretching by multiplication with the image before applying asinh
    - larger parameters lead to a more log like handling
    - lower parameters lead to a linear like stretching
