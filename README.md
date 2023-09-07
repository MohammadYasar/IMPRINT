# IMPRINT
Code for IMPRINT: Interactional Dynamics-aware Multi-agent Motion Prediction using Multimodal Context.

# Instruction

Our implementation requires download skeleton and rgb data for [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/). In addition, we suggest extracting rgb features separately for efficient training. We have provided a [feature_extract.py](./utils/feature_extract.py) script. Replace the path to video file and the path to store
extracted features.

## Training IMPRINT

```
./run_ntu.sh
```

If you find this code useful, and want to cite our paper, please use the following bibtex:

```
@article{yasar2023imprint,
  title={IMPRINT: Interactional Dynamics-aware Motion Prediction in Teams using Multimodal Context},
  author={Yasar, Mohammad Samin and Islam, Md Mofijul and Iqbal, Tariq},
  journal={ACM Transactions on Human-Robot Interaction},
  year={2023}
}
```


