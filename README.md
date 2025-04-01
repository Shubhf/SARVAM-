# Einops from Scratch

## Overview
This project is an implementation of core `einops` functionality from scratch, specifically the `rearrange` operation. The implementation supports reshaping, transposition, splitting, merging, and repeating of axes using NumPy arrays.

## Features
- **Reshaping**: Modify tensor dimensions according to a specified pattern.
- **Transposition**: Reorder tensor axes efficiently.
- **Splitting Axes**: Break an axis into multiple smaller axes.
- **Merging Axes**: Combine multiple axes into a single dimension.
- **Repeating Axes**: Duplicate tensor elements along a specified axis.
- **Ellipsis Handling**: Support for batch dimensions.

## Installation
```sh
# Clone the repository
git clone https://github.com/your-repo/einops-scratch.git

# Navigate to the project directory
cd einops-scratch

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
import numpy as np
from your_module import rearrange

# Transpose
x = np.random.rand(3, 4)
result = rearrange(x, 'h w -> w h')

# Split an axis
x = np.random.rand(12, 10)
result = rearrange(x, '(h w) c -> h w c', h=3)

# Merge axes
x = np.random.rand(3, 4, 5)
result = rearrange(x, 'a b c -> (a b) c')

# Repeat an axis
x = np.random.rand(3, 1, 5)
result = rearrange(x, 'a 1 c -> a b c', b=4)

# Handle batch dimensions
x = np.random.rand(2, 3, 4, 5)
result = rearrange(x, '... h w -> ... (h w)')
```

## Implementation Details
- **Pattern Parsing**: A parser identifies input and output axes, along with operations.
- **Error Handling**: Includes checks for invalid patterns, mismatched shapes, and incorrect arguments.
- **Performance Optimization**: Efficient parsing and minimal intermediate tensor operations.

## Testing
Unit tests are included to validate correctness across various use cases:
```sh
pytest tests/
```

## Submission
- The implementation is provided in a Google Colab notebook.
- Unit tests are included in separate cells.

## License
This project is licensed under the MIT License.

## Contact
For queries, reach out to `your-email@example.com`.


