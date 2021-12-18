# TrOCR

## Introduction
This repository is created for test TrOCR-BASE model by using IAM dataset, CER(Character Error Rate).

 
| Model                          |  #Param   | Test set | Score          |
|--------------------------------|-----------|----------|----------------|
| TrOCR-Base                     | 334M       | IAM     | 3.42 (Cased CER)     |

## Test Environment
OS : Ubuntu 20.04.3 LTS
GPU : RTX2080
CPU : i7-8700K
RAM : 16GB

## Used Model and test dataset
|   Model  | Download |
| -------- | -------- |
| TrOCR-Base-IAM     | [trocr-base-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt) |

|   Test set  | Download |
| --------| -------- |
| IAM     | [IAM.tar.gz](https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz) |



## Citation
``` latex
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree. Portions of the source code are based on the [fairseq](https://github.com/pytorch/fairseq) project. [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information
For help or issues using TrOCR, please submit a GitHub issue.

For other communications related to TrOCR, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
