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
|               Anime Name              |        Already Liked         | Predicted Score |
|:--------------------------------------|:----------------------------:|:---------------:|
|           Ningen Doubutsuen           |      Majo no Takkyuubin      |      3.062      |
|        Bakugan Battle Brawlers        |    Tenkuu no Shiro Laputa    |      2.981      |
| Chain Chronicle: Haecceitas no Hikari |       Pumpkin Scissors       |      2.981      |
|                Kure-nai               |       Omoide Poroporo        |      2.821      |
|       Break Blade Picture Drama       | Heisei Tanuki Gassen Ponpoko |      2.819      |
|            Coral no Tanken            |       Tonari no Totoro       |      2.805      |
|                  None                 |         Zetsuai 1989         |      2.795      |
|    Rokujouma no Shinryakusha!? (TV)   |           Monster            |      2.795      |
|          Platonic Chain: Web          |         xxxHOLiC Kei         |      2.793      |
|   Seikimatsu Occult Gakuin Specials   |       Shounen Onmyouji       |      2.793      |


#### <div align="center"> CatBoost Recommendation </div>

|                     Anime Name                    |        Already Liked         | Predicted Score |
|:--------------------------------------------------|:----------------------------:|:---------------:|
|                     Gad Guard                     |      Majo no Takkyuubin      |      4.797      |
|                     Fuyu no Hi                    |    Tenkuu no Shiro Laputa    |      4.566      |
|          Hatsukoi Limited.: Gentei Shoujo         |       Pumpkin Scissors       |      4.566      |
|                  Slam Dunk Movie                  |       Omoide Poroporo        |      4.541      |
|           Koukaku Kidoutai Nyuumon Arise          | Heisei Tanuki Gassen Ponpoko |      4.522      |
| Joshiochi!: 2-kai kara Onnanoko ga... Futtekita!? |       Tonari no Totoro       |      4.266      |
|                   Ai Yori Aoshi                   |         Zetsuai 1989         |      4.266      |
|           Shin Mitsubachi Maya no Bouken          |           Monster            |      4.141      |
|             Denshinbashira no Okaasan             |         xxxHOLiC Kei         |      4.141      |
|       Schoolgirl Strikers: Animation Channel      |       Shounen Onmyouji       |      4.126      |


## Reference
+ [Which Tricks are Important for Learning to Rank?](https://openreview.net/pdf?id=MXfTQp8bZF)
+ [ARE NEURAL RANKERS STILL OUTPERFORMED BY GRADIENT BOOSTED DECISION TREES?](https://openreview.net/pdf?id=Ut1vF_q_vC)