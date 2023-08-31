# ProtoMixer
Released code for the paper ['Mixing Histopathology Prototypes into Robust Slide-Level Representations for Cancer Subtyping'](https://proceedings.mlr.press/v156/butke21a/butke21a.pdf)
by **Joshua Butke**, Noriaki Hashimoto, Ichiro Takeuchi, Hiroaki Miyoshi, Koichi Ohshima, and Jun Sakuma.

Accepted to and presented at MICCAI 2023 Workshop 14TH INTERNATIONAL CONFERENCE ON MACHINE LEARNING IN MEDICAL IMAGING [(MLMI 2023)](https://sites.google.com/view/mlmi2023/).

[![Follow me on Twitter](https://img.shields.io/twitter/follow/JoshuaButke?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=JoshuaButke)

## Overview
This repo contains the Python code of our experiments, however there is no data included. Still, this implementation might serve as a starting point for those interested in applying Attention-based Multiple Instance Learning to problems of cytopathology.

## Citation
Please cite our paper, if this work is of use to you or you use the code in your research:
```
    @inproceedings{butke2021end,
        title={End-to-end Multiple Instance Learning for
            Whole-Slide Cytopathology of Urothelial Carcinoma},
        author={Butke, Joshua and Frick, Tatjana and Roghmann, Florian
            and El-Mashtoly, Samir F and Gerwert, Klaus and Mosig, Axel},
        booktitle={MICCAI Workshop on Computational Pathology},
        pages={57--68},
        year={2021},
        organization={PMLR}
    }
```

## Requirements
Packages:
- Pytorch (>= 1.6.0)
- OpenCV (4.4.0)
- sklearn (0.23.0)
- matplotlib (3.3.0)

Hardware:
We used a cluster equipped with 4 NVIDIA V100 GPUs, which is reflected in `joshnet/custom_model.py` where blocks of layers are assigned to dedicated cards.


## Further Reading
I highly recommend to check out the original paper and implementation of **Ilse et al.** for Attention-based MIL, that can be found [here](https://github.com/AMLab-Amsterdam/AttentionDeepMIL), as well as the **Li et al.** paper ['Deep Instance-Level Hard Negative Mining Model for Histopathology Images'](https://arxiv.org/pdf/1906.09681.pdf). The second one introduced Hard Negative Mining that I adopted. However they never released any code, so I implemented their improvements as best as I could.

## Contact
If you have any questions you can contact me at joshua.butke@riken.jp, however we do not gurantee any support for this software.

### Acknowledgements
This work was supported by JST CREST JPMJCR21D3 and Grant-in-Aid for Scientific Research (A) 23H00483. J.B. was supported by the Gateway Fellowship program of Research School, Ruhr-University Bochum, Bochum, Germany.

