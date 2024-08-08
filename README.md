# D2Styler

Welcome to the official implementation of [D2Styler](https://arxiv.org/pdf/2408.03558v1), which has been accepted at the International Conference on Pattern Recognition (ICPR 2024).

## Overview

"D2Styler: Advancing Arbitrary Style Transfer with Discrete Diffusion Methods" introduces a novel framework for style transfer called D2Styler. Leveraging VQ-GANs and discrete diffusion, this method aims to improve the quality and stability of style transfer, addressing common issues like mode-collapse and over/under-stylization. By using Adaptive Instance Normalization (AdaIN) features, D2Styler facilitates effective style transfer between images. Experimental results show that D2Styler outperforms twelve existing methods on various metrics, producing high-quality, visually appealing images. The method uses images from the WikiArt and COCO datasets.
The model's architecture and its qualitative results are showcased below. The model will be available on HuggingFace 🤗, where you can download it for inference or fine-tuning.

## Model Architecture

![D2Styler Architecture](https://github.com/user-attachments/assets/673efff9-dad5-4872-97af-eab1e72ece7a)

## Results

![D2Styler Results](https://github.com/user-attachments/assets/37add96c-1b76-4e83-bd90-5b52228f5fa8)

## Installation

To get started with D2Styler, follow the steps below to install the necessary dependencies:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/D2Styler.git
    cd D2Styler
    ```

2. Install the dependencies:

    ```bash
    pip install -e ".[torch]"
    pip install -e .[all,dev,notebooks]
    ```


## Contributing

We welcome contributions to D2Styler! If you have any ideas for improvements or find any issues, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

For more details, please refer to our [paper](https://arxiv.org/pdf/2408.03558v1) and our repository on [HuggingFace](https://huggingface.co/).
