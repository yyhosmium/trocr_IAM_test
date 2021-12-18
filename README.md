# TrOCR

## Introduction
TrOCR is an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. 


 
| Model                          |  #Param   | Test set | Score          |
|--------------------------------|-----------|----------|----------------|
| TrOCR-Base                     | 334M       | IAM     | 3.42 (Cased CER)     |



## Fine-tuning and evaluation
|   Model  | Download |
| -------- | -------- |
| TrOCR-Base-IAM     | [trocr-base-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt) |

|   Test set  | Download |
| --------| -------- |
| IAM     | [IAM.tar.gz](https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz) |



## Citation
If you want to cite TrOCR in your research, please cite the following paper:
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
