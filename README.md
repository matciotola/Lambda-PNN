# Unsupervised Deep Learning-based Pansharpening with Jointly-Enhanced Spectral and Spatial Fidelity

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.14403)
[![GitHub Stars](https://img.shields.io/github/stars/matciotola/Lambda-PNN?style=social)](https://github.com/matciotola/Lambda-PNN)
![Visitors](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2Fmatciotola%2FLambda-PNN.json\&style=flat\&label=hits\&color=blue)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/fca77356704048f6a47841f73e8c97db)](https://app.codacy.com/gh/matciotola/Lambda-PNN/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/github/matciotola/Lambda-PNN/graph/badge.svg?token=28AINVS2EK)](https://codecov.io/github/matciotola/Lambda-PNN)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-learning-based/pansharpening-on-worldview-3-pairmax)](https://paperswithcode.com/sota/pansharpening-on-worldview-3-pairmax?p=unsupervised-deep-learning-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-learning-based/pansharpening-on-worldview-3-adelaide)](https://paperswithcode.com/sota/pansharpening-on-worldview-3-adelaide?p=unsupervised-deep-learning-based)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-learning-based/pansharpening-on-worldview-2-pairmax)](https://paperswithcode.com/sota/pansharpening-on-worldview-2-pairmax?p=unsupervised-deep-learning-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-learning-based/pansharpening-on-worldview-2-washington)](https://paperswithcode.com/sota/pansharpening-on-worldview-2-washington?p=unsupervised-deep-learning-based)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-learning-based/pansharpening-on-geoeye-1-pairmax)](https://paperswithcode.com/sota/pansharpening-on-geoeye-1-pairmax?p=unsupervised-deep-learning-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-learning-based/pansharpening-on-geoeye-1-genoa)](https://paperswithcode.com/sota/pansharpening-on-geoeye-1-genoa?p=unsupervised-deep-learning-based)


[Unsupervised Deep Learning-based Pansharpening with Jointly-Enhanced Spectral and Spatial Fidelity](https://ieeexplore.ieee.org/document/10198408) ([ArXiv](https://arxiv.org/abs/2307.14403)) is
a deep learning method with a residual attention mechanism for Pansharpening, based on unsupervised and full-resolution framework training.
The proposed algorithm features a novel loss function that jointly promotes the spectral and spatial quality of the pansharpened data.
Experiments on a large variety of test images, performed in challenging scenarios,
demonstrate that the proposed method compares favourably with the state of the art both in terms of numerical results and visual output.

## Cite λ-PNN

If you use λ-PNN in your research, please use the following BibTeX entry.

    @article{Ciotola2023,
      author={Ciotola, Matteo and Poggi, Giovanni and Scarpa, Giuseppe},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Unsupervised Deep Learning-Based Pansharpening With Jointly Enhanced Spectral and Spatial Fidelity}, 
      year={2023},
      volume={61},
      number={},
      pages={1-17},
      doi={10.1109/TGRS.2023.3299356}
     }

## Team members

*   Matteo Ciotola (matteo.ciotola@unina.it);

*   Giovanni Poggi   (poggi@unina.it);

*   Giuseppe Scarpa  (giuseppe.scarpa@uniparthenope.it).

## License

Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/Lambda-PNN/LICENSE.txt)
(included in this package)

## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constraints:

*   Python 3.10.10
*   PyTorch 2.0.0
*   Cuda 11.7 or 11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the algorithm
*   Download the algorithm and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/matciotola/Lambda-PNN

*   Create the virtual environment with the `lambda_pnn_environment.yml`

<!---->

    conda env create -n lambda_pnn_env -f lambda_pnn_environment.yml

*   Activate the Conda Environment

<!---->

    conda activate lambda_pnn_env

*   Test it

<!---->

    python test.py -i example/WV3_example.mat -o ./Output_folder/ -s WV3 --coregistration --show_results 

## Usage

### Before start

The easiest way to test this algorithm is to create a `.mat` file. It must contain:

*   `I_MS_LR`: Original Multi-Spectral Stack in channel-last configuration (Dimensions: H x W x B);
*   `I_PAN`: Original Panchromatic band, without the third dimension (Dimensions: H x W).

It is possible to convert the GeoTIff images into the required format with the scripts provided in [`tiff_mat_conversion.py`](https://github.com/matciotola/Lambda-PNN/blob/master/tiff_mat_conversion.py):

    python tiff_mat_conversion.py -m Tiff2Mat -ms /path/to/ms.tif -pan /path/to/pan.tif  -o path/to/file.mat

Please refer to `--help` for more details.

### Testing

The easiest command to use the algorithm on full-resolution data:

    python test.py -i path/to/file.mat -s sensor_name

Several options are possible. Please refer to the parser help for more details:

    python test.py -h
