> Have feedback to make this site better? I'd love to hear from you. Please fill out [this form](https://forms.gle/YD5UPKAnasYtCutbA).

LLM-Visualized is an interactive Three.js walkthrough of GPT-2 124M internals. The project follows a live 12-layer transformer forward pass and lets you inspect residual streams, attention heads, and MLP updates in both the 3D scene and the 2D matrix view.

The visualization uses real sampled GPT-2 activations to color vectors throughout the visualization.

The goal is to make the model's internal flow easier to understand by tying each visualization back to the same runtime state as the pass progresses.

## Keyboard Controls

### Global and 3D scene

- **Space** pauses or resumes the animation.
- **Arrow keys** orbit the 3D camera.
- **W / A / S / D** pan the 3D camera.
- **+ / -** zoom the 3D camera.
- **Page Up / Page Down** also zoom the 3D camera.
- **Escape** closes the current detail panel or exits focused panel states.

### 2D view and MHSA matrix view

- **Arrow keys** or **W / A / S / D** pan the active 2D viewport after you interact with it.
- **+ / -** zoom the active 2D viewport.
- The same movement keys are context-sensitive: when a 2D panel owns keyboard focus, they move that panel instead of the main 3D camera.

## Technical Overview

TODO

## Source of Inspiration

TODO
