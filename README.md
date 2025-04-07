# Einops from Scratch Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange)
![Tests](https://img.shields.io/badge/Tests-85%25%20coverage-green)

## Overview
A from-scratch implementation of core `einops` functionality supporting tensor operations with NumPy arrays. Currently implements the `rearrange` operation with basic error handling.

## Current Implementation Status

| Feature               | Status  | Notes                      |
|-----------------------|---------|----------------------------|
| Basic Reshaping       | ✅      | Full support               |
| Axis Transposition    | ✅      | Full support               |
| Simple Axis Splitting | ✅      | Single-level only          |
| Axis Merging          | ✅      | Basic cases working        |
| Axis Repetition       | ⚠️      | Needs edge case fixes      |
| Ellipsis Handling     | ⚠️      | Partial support            |
| Nested Operations     | ❌      | Not yet implemented        |

## Test Results Summary

```text
🔁 Axis Repetition: 
  ✅ Basic cases
  ❌ Fails on axis matching ('b' not in list)

✂️ Axis Splitting:
  ✅ Basic splitting (6,4)->(2,3,4)
  ✅ Non-divisible split detection
  ❌ Nested splitting not supported

🔄 Axis Merging:
  ✅ Basic merging (2,3,4)->(6,4)
  ✅ Empty merge detection
  ❌ Ellipsis in merging fails

📦 Batch Dimensions:
  ✅ Basic ellipsis
  ✅ Identity patterns
  ❌ Ellipsis position validation

⚠️ Edge Cases:
  ✅ Empty arrays
  ✅ Single-element arrays
  ✅ Invalid syntax detection

## Installation
```sh
git clone https://github.com/Shubhf/SARVAM-.git
cd SARVAM-
pip install -e .
```

## Usage
```python
import numpy as np
from einops_scratch import rearrange

# Transposition
x = np.random.rand(3, 4)
rearrange(x, 'h w -> w h')  # (4, 3)

# Basic splitting
x = np.random.rand(6, 10)
rearrange(x, '(h w) c -> h w c', h=2)  # (2, 3, 10)

# Axis merging
x = np.random.rand(2, 3, 5)
rearrange(x, 'a b c -> (a b) c')  # (6, 5)
```

## Current Limitations
```python
# Not yet working:
# x = np.random.rand(3, 2, 5)
# rearrange(x, 'a 1 c -> a b c', b=4)  # Axis repetition edge cases

# x = np.random.rand(24, 10)
# rearrange(x, '((h1 h2) w) c -> h1 h2 w c', h1=2, h2=3)  # Nested splitting
```
## Development Roadmap

-Basic reshape/transpose operations

-Simple axis splitting/merging

-Full ellipsis support

-Nested operation support

-Advanced error messages

-Performance optimization


## Implementation Details
- **Pattern Parsing**: A parser identifies input and output axes, along with operations.
- **Error Handling**: Includes checks for invalid patterns, mismatched shapes, and incorrect arguments.
- **Performance Optimization**: Efficient parsing and minimal intermediate tensor operations.

Test coverage includes:

-Basic functionality

-Error conditions

-Edge cases

-Failure modes

## Submission
- The implementation is provided in a Google Colab notebook.
- Unit tests are included in separate cells.

## License
This project is licensed under the MIT License.

## Contact
For queries, reach out to `shubhgarg265@gmail.com`.


