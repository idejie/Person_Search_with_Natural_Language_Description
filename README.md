# Person Search with Natural Language Description

**Torch Version: [[ShuangLI59/Person-Search-with-Natural-Language-Description]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)**

## Dataset

**Details see [[ShuangLI59/Person-Search-with-Natural-Language-Description]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)**

### Preprocess
- you can update the parameters for the preprocess in [utils/config.py](./utils/config.py), like `word_count_threshold`
- then run:
```shell
python main.py
```
- the script will produce the  vocabulary map in directory `vocab` and split dataset  in directory `data`

## TODO:
- [x] Preprocess
    - [x] create the vocabulary of the dataset
    - [x] encode the captions
- [ ] DataLoader
    - [ ] CUDK-PEDES dataset
- [ ]  Model: GNA-RNN
    - [ ] Visual units
    - [ ] Attention over visual units
    - [ ] Word-level gates for visual units
    - [ ] train
    - [ ] valid
    - [ ] test
- [ ] Web Visualization

