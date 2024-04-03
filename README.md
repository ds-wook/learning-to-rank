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
|                                Anime Name                                |        Already Liked         | Predicted Score |
|:-------------------------------------------------------------------------|:----------------------------:|:---------------:|
|                  Ougon no Hou: El Cantare no Rekishikan                  |      Majo no Takkyuubin      |      3.906      |
|                                Connected                                 |    Tenkuu no Shiro Laputa    |      3.832      |
|                               Yume no Kawa                               |       Pumpkin Scissors       |      3.832      |
|                               Yuusei Kamen                               |       Omoide Poroporo        |      3.832      |
|                  Sekai Meisaku Douwa: Mori wa Ikiteiru                   | Heisei Tanuki Gassen Ponpoko |      3.832      |
|                           Mii-chan no Tenohira                           |       Tonari no Totoro       |      3.832      |
|                                 Tarareba                                 |         Zetsuai 1989         |      3.805      |
|                      Shakugan no Shana II (Second)                       |           Monster            |      3.797      |
| Slam Dunk: Hoero Basketman-damashii! Hanamichi to Rukawa no Atsuki Natsu |         xxxHOLiC Kei         |      3.797      |
|                                  Gift±                                   |       Shounen Onmyouji       |      3.797      |

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