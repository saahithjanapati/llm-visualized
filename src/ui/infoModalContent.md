> If you have any ideas or feedback to make the site better, I'd love to hear from you! You can send it through [this form](https://forms.gle/YD5UPKAnasYtCutbA), or as a [DM on X](https://x.com/stargaz3r42).

This site is an interactive 3D and 2D visualization of GPT-2 (124M) performing forward passes.

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (124M) is a 12-layer decoder-only [Transformer](https://arxiv.org/abs/1706.03762) trained to predict the next token on a corpus of internet text.

When most people refer to GPT-2, they are usually referring to the 1.5B parameter version rather than the smaller 124M version depicted on this site.

## Overview

The vectors, attention scores, and logit probabilities on this site were extracted from GPT-2 as it processed a prompt. The main goal is to help you follow one real forward pass as it moves through the model.

The 3D view and 2D view show the same model state in two different ways. The 3D view is meant to show the forward pass in motion. The 2D view is meant to make the same information readable in matrix form.

For the vectors shown here, I extracted one value every 64 dimensions and used those sampled values to color the vectors. For example, for a 768-dimensional vector, I extract the values at positions 0, 64, 128, and so on up to 704. Those 12 sampled values determine the shades used to color the vector. For a 64-dimensional vector, I just use the value at position 0. If you turn on Dev mode in Settings and click a vector in the scene, you can inspect the extracted values in the sidebar.

The 3D visualization was built with Three.js and the site is currently deployed on Vercel.

## Keyboard Controls

### Global and 3D scene

- Pause or resume: `Space`
- Orbit camera: `↑` `↓` `←` `→`
- Pan camera: `W` `A` `S` `D`
- Zoom camera: `+` `-`
- Close the active panel or exit a focused state: `Esc`

### 2D view and MHSA matrix view

- After clicking the 2D view, pan with `↑` `↓` `←` `→` or `W` `A` `S` `D`
- Zoom with `+` `-`

## Some Sources of Inspiration

Below is an incomplete list of sources that inspired this project in some way. If you like this project, you might like these too!

- [Andrej Karpathy's minGPT repo](https://github.com/karpathy/minGPT) and the [Zero to Hero series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- The diagrams in Anthropic's [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Oriol Vinyals' post that vectors are ["The soul of a neural network"](https://x.com/OriolVinyalsML/status/1796970824353595510)
- The diagrams in [Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262)
- [distill.pub](https://distill.pub/)
- Brendan Bycroft's [3D GPT Visualization](https://bbycroft.net/llm)
- Polo Club's [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- AlphaXiv's [Illustrated Transformer in 3D](https://x.com/askalphaxiv/status/1983543092079640963?s=20)
- Ja-Bin Huang's [Deep Learning Video Visualizations](https://www.youtube.com/@jbhuang0604/videos)
- 3Blue1Brown's [Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Welch Labs' [deep learning videos](https://www.youtube.com/@WelchLabs/videos)
- Maarten Grootendorst's [Exploring Language Models](https://newsletter.maartengrootendorst.com/)
- Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- Jay Alammar's [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Khan Academy's](https://www.khanacademy.org/) black-and-colored visual style
- Jack Morris' [Transformer visualization thread](https://x.com/jxmnop/status/1757129193442173367?s=20)
- Will Depue's [WebGPT](https://github.com/0hq/WebGPT)
- Aman Sanger's comment that [UX design is a form of programming](https://youtu.be/BGgsoIgbT_Y?si=_oV22SkD5jNgDxZH&t=1167)
- roon (@tszzl)'s post that [latent space is best explored in person](https://x.com/tszzl/status/1964842112438161707)
- roon (@tszzl)'s essay [Text is the universal interface](https://scale.com/blog/text-universal-interface)
- Sasha Rush's [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- Andrej Karpathy's short story [A Forward Pass](https://karpathy.github.io/2021/03/27/forward-pass/)

## Limitations and Next Steps

- In the 3D view, I think the x-axis has less semantic meaning than the y-axis, which is used for layers, and the z-axis, which is used for context dimension. The current organization of the attention heads is a little arbitrary, and I'm trying to think of other ways to represent this part of the model in the 3D setting.

- The way the vectors are colored right now is also somewhat arbitrary. Right now I just sample one value every 64 dimensions, but I could have chosen a different set of positions and gotten a different visual result. I think the vectors look nice, but it would be cool if the color scheme was a bit less arbitrary. One idea is to include a [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) visualization to show how the residual vectors change throughout the scene (thank you to Professor David Bau for the suggestion). Let me know if you have any other ideas!

- I plan to add more prompts/generations once I get a better sense of traffic volume. I also plan to release my extraction script so people can visualize forward passes of their own prompts.

- The `Have a question? Copy context to ask 🤖` button in the sidebars is also still pretty experimental. The generated prompts could definitely be made better and optimized.

- There is a lot of unnecessary lag and several interaction bugs that need to be fixed. A lot of the current work is optimization, bug hunting, and cleanup.

If you run into any problems or have suggestions, please let me know through the [form above](#project-info-feedback)!

## Acknowledgements

Thank you to all my AI buddies who helped me build this project. I could not have done it without you all.

Thank you to everyone on the internet who has already been generous with their time and shared detailed feedback on how to improve the site. I really appreciate it and will try my best to incorporate those ideas quickly.

Thank you to my friends and family who gave me a lot of ideas throughout development. This project would not be the same without their feedback.

And thank you for visiting the site! If you have any ideas or feedback to make it better, I'd love to hear from you (through [this form](https://forms.gle/YD5UPKAnasYtCutbA) or [DM on X](https://x.com/stargaz3r42)).

I hope you enjoy the site 😊🫡
