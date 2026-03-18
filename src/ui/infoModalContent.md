> If you have any ideas or feedback to make the site better, I'd love to hear from you! You can send it through [this form](https://forms.gle/YD5UPKAnasYtCutbA), or as a DM on X.

This site is an interactive 3D and 2D visualization of GPT-2 (124M) performing forward passes.

GPT-2 (124M) is a 12-layer decoder-only Transformer trained to predict the next token on a corpus of internet text.

When most people refer to GPT-2, they are usually referring to the 1.5B parameter version rather than the smaller 124M version depicted on this site.

## Overview

The vectors, attention scores, and logit probabilities shown on this site were extracted from GPT-2 as it processed a prompt. The 3D view and 2D view are meant to show the same model state from different angles, so you can follow one forward pass as it moves through the network.

For all the vectors shown here, I saved a single value every 64 units. That means a 768-dimensional residual vector is represented with 12 colors, and a 64-dimensional vector is represented with one color. If you turn on Dev mode in Settings and click one of the vectors in the scene, you can see the extracted values in the sidebar.

The site was built with Three.js and is currently deployed on Vercel.

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

## Some Sources of Inspiration

Below is an incomplete list of projects, papers, and references that inspired this project in one way or another.

- [Andrej Karpathy's minGPT repo](https://github.com/karpathy/minGPT) and the [Zero to Hero series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- The diagrams in Anthropic's [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- The diagrams in [Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262)
- [Distill](https://distill.pub/)
- Brendan Bycroft's [3D GPT Visualization](https://bbycroft.net/llm)
- Polo Club's [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- AlphaXiv's [Illustrated Transformer in 3D](https://x.com/askalphaxiv/status/1983543092079640963?s=20)
- Ja-Bin Huang's [Deep Learning Video Visualizations](https://www.youtube.com/@jbhuang0604/videos)
- 3Blue1Brown's [Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Welch Labs' [deep learning videos](https://www.youtube.com/@WelchLabs/videos)
- Maarten Grootendorst's [Exploring Language Models](https://newsletter.maartengrootendorst.com/)
- Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- Khan Academy's black-and-colored visual style
- Jack Morris' [Transformer visualization thread](https://x.com/jxmnop/status/1757129193442173367?s=20)
- Will Depue's [WebGPT](https://github.com/0hq/WebGPT)
- Sasha Rush's [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- Jay Alammar's [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Aman Sanger's point that [UX design is a form of programming](https://youtu.be/BGgsoIgbT_Y?si=_oV22SkD5jNgDxZH&t=1167)
- roon's post that [latent space is best explored in person](https://x.com/tszzl/status/1964842112438161707)
- roon's essay [Text is the universal interface](https://scale.com/blog/text-universal-interface)
- Oriol Vinyals' note that vectors are ["the soul of a neural network"](https://x.com/OriolVinyalsML/status/1796970824353595510)
- Andrej Karpathy's short story [A Forward Pass](https://karpathy.github.io/2021/03/27/forward-pass/)

## Limitations and Next Steps

- Attention is still organized in a somewhat weird way. The z-axis is used for context, the y-axis is used for layer depth, and the x-axis is used for attention heads. That works for now, but I still want to find a better way to use the 3D space to depict the Transformer.
- I plan to add more prompts and generations once I get a better sense of steady-state traffic. I also plan to release my extraction script so people can visualize forward passes of their own prompts.
- There are still a lot of small things that do not need to be so laggy, plus interaction bugs that need to be fixed. A lot of the current work is still bug hunting and cleanup.
- If you run into any problems or have suggestions, please let me know through the form above!

## Acknowledgements

Thank you to all my AI buddies who helped me build this project. I could not have done it without you all.

Thank you to the people on the internet who have already been generous with their time and shared very detailed feedback on how to improve the site. I really appreciate it and will try my best to incorporate those ideas quickly.

Thank you to my friends and family who gave me a lot of ideas throughout development. This project would not be the same without their feedback.

And thank you for visiting the site! If you have any ideas or feedback to make it better, I'd love to hear from you!

I hope you enjoy the site 😊🫡
