
# Custom Object Detection Model (From Scratch)

A custom object detection model inspired by YOLOv8, implemented from scratch in Python (without PyTorch/TensorFlow) and accelerated on CPU using Numba Parallel.

## Key Features
- **From-Scratch Architecture:** Custom forward & backward passes for `Conv2D`, `BatchNorm`, `C2f`, `SiLU`, `Detect` head, `NMS`, and `MSE/BCE` losses.
- **CPU Optimization:** Parallelized tensor operations using `Numba (@njit / parallel)`.
- **Performance (Pure CPU — AMD Ryzen 7 7840H):**
  - Training: `3–5s / batch`
  - Inference: `8–15 FPS` (avg. `~12 FPS`, up to `15.8 FPS` on lightweight face detection)
- **Tooling:** Includes a custom-built dataset annotation tool and clean `train` / `detect` API.

## Demo Results
<img width="924" height="731" alt="image" src="https://github.com/user-attachments/assets/df8748c5-818c-48ae-8616-916b0b46f1bc" />


https://github.com/user-attachments/assets/d03e993c-b13d-45f7-9644-fa25d1ca137e


---
> **Note:** The core model engine is kept in a private repository. Full source code walkthrough and live demo are available upon request during technical interviews.
