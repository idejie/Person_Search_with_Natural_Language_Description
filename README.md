# Person Search with Natural Language Description

>PyTorch implementation for "Person Search with Natural Language Description"(CVPR2017)

**Torch Version: [[ShuangLI59/Person-Search-with-Natural-Language-Description]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)**

## 0.DataSet
 - Download:
**Details in [[ShuangLI59/Person-Search-with-Natural-Language-Description]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)**

- Preprocess
    - you can update the parameters for the preprocess in [utils/config.py](./utils/config.py), like `word_count_threshold` and set `action`value "process"
    - then run:
        ```shell
        python main.py
        ```
    - the script will produce the  vocabulary map in directory `vocab` and split dataset  in directory `data`

## 1.Train

## 2.Test

## 3.Visualization

## 4.TODO:
- [x] Preprocess
    - [x] create the vocabulary of the dataset
    - [x] encode the captions
- [] DataLoader
    - [x] CUDK-PEDES dataset
    - [ ] sample negative
- [ ]  Model: GNA-RNN
    - [x] Visual units
    - [x] Attention over visual units
    - [x] Word-level gates for visual units
    - [x] train
    - [ ] valid
    - [ ] test
- [ ] Web Visualization

## References
- [[ShuangLI59/Person-Search-with-Natural-Language-Description]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)

