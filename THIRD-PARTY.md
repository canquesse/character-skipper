# Third-party components

This extension bundles the following third-party software and model weights.
All of them run **locally**; none of them contact a network at runtime.

## Libraries

| Component | Version | License | Source |
|---|---|---|---|
| `@vladmandic/human` (`libs/human.js`) | 3.3.6 | MIT | https://github.com/vladmandic/human |
| TensorFlow.js (bundled inside Human) | 4.22.0 | Apache-2.0 | https://github.com/tensorflow/tfjs |
| ONNX Runtime Web (`libs/ort/`) | 1.20.1 | MIT | https://github.com/microsoft/onnxruntime |

### Fonts (`libs/fonts/`)
| Font | License | Source |
|---|---|---|
| Space Grotesk | SIL Open Font License 1.1 | https://fonts.google.com/specimen/Space+Grotesk |
| Hanken Grotesk | SIL Open Font License 1.1 | https://fonts.google.com/specimen/Hanken+Grotesk |

Fonts are self-hosted (not loaded from Google's servers) so the extension makes
no external requests.

## Model weights (`models/`)

| Model | Purpose | License / terms | Source |
|---|---|---|---|
| `blazeface.*` | Face detection | MIT | https://github.com/vladmandic/human-models |
| `facemesh.*` | Facial landmarks (used for alignment) | MIT | https://github.com/vladmandic/human-models |
| `faceres.*` | Face embedding (fallback engine) | MIT | https://github.com/vladmandic/human-models |
| `w600k_mbf.onnx` | Face embedding (primary engine, ArcFace) | **See note below** | InsightFace `buffalo_s` model pack |

> **Note on `w600k_mbf.onnx`:** this file comes from the InsightFace model zoo.
> InsightFace states that its **pretrained models are provided for
> non-commercial research purposes only**. Review these terms before
> redistributing this file or shipping it in a commercial product.
> See https://github.com/deepinsight/insightface for the current terms.
