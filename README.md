[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
# Learning-to-rank
This Repository built a lambdarank algorithm for anime recommendations with gradient boosting algorithm.
The code style has been configured to use Black, and the maximum line length has been set to 120 characters.

## Dataset
I used [Anime Recommendation LTR](https://www.kaggle.com/datasets/ransakaravihara/anime-recommendation-ltr-dataset) in the Kaggle dataset.

## Requirements
By default, hydra-core==1.3.0 was added to the requirements given by the competition. For pytorch, refer to the link at https://pytorch.org/get-started/previous-versions/ and reinstall it with the right version of pytorch for your environment.

You can install a library where you can run the file by typing:
```sh
$ conda env create --file environment.yaml
```

## Run code
Running the learning code shell allows learning
```sh
$ sh scripts/run.sh
```


## Learning to Rank
![image](https://github.com/ds-wook/learning-to-rank/assets/46340424/2ff9bbb4-4b21-4c00-87ad-47d140426dc7)



## Benchmark
TBD


## Results

#### <div align="center"> LightGBM Recommendation </div>

|                       Anime Name                      |        Already Liked         | Predicted Score |
|:------------------------------------------------------|:----------------------------:|:---------------:|
|  Pokemon Movie 02: Maboroshi no Pokemon Lugia Bakutan |      Majo no Takkyuubin      |      3.025      |
|     Toaru Kagaku no Railgun: Motto Marutto Railgun    |    Tenkuu no Shiro Laputa    |      2.967      |
|                    Ten made Tobaso                    |       Pumpkin Scissors       |      2.967      |
|                       Mikan-bune                      |       Omoide Poroporo        |      2.866      |
|             Dorami-chan: A Blue Straw Hat             | Heisei Tanuki Gassen Ponpoko |      2.828      |
| K: Seven Stories Movie 3 - Side:Green - Uwagaki Sekai |       Tonari no Totoro       |      2.771      |
|        Kawasaki Frontale x Tentai Senshi Sunred       |         Zetsuai 1989         |      2.729      |
|                      X²: Double X                     |           Monster            |      2.729      |
|          Xiong Chumo Zhi: Chunri Dui Dui Peng         |         xxxHOLiC Kei         |      2.649      |
|                   Kazoku no Hanashi                   |       Shounen Onmyouji       |      2.641      |

#### <div align="center"> CatBoost Recommendation </div>

|                  Anime Name                  |        Already Liked         | Predicted Score |
|:---------------------------------------------|:----------------------------:|:---------------:|
|   Shaonu Qianxian: Renxing Xiao Juchang 2    |      Majo no Takkyuubin      |      4.152      |
| Digimon Adventure: 20 Shuunen Memorial Story |    Tenkuu no Shiro Laputa    |      4.152      |
|              Koneko no Rakugaki              |       Pumpkin Scissors       |      4.071      |
|                Chiisana Jumbo                |       Omoide Poroporo        |      3.959      |
|    Sam to Chip no wa Hachamecha Dai Race     | Heisei Tanuki Gassen Ponpoko |      3.798      |
|               Ringo to Shoujo                |       Tonari no Totoro       |      3.798      |
|                  Shake-chan                  |         Zetsuai 1989         |      3.755      |
|                 ChäoS;Child                  |           Monster            |      3.601      |
|               Tamagawa Kyoudai               |         xxxHOLiC Kei         |      3.599      |
|                Inazuma Eleven                |       Shounen Onmyouji       |      3.599      |

## Reference
+ [Which Tricks are Important for Learning to Rank?](https://openreview.net/pdf?id=MXfTQp8bZF)
+ [ARE NEURAL RANKERS STILL OUTPERFORMED BY GRADIENT BOOSTED DECISION TREES?](https://openreview.net/pdf?id=Ut1vF_q_vC)