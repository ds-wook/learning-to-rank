[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
# learning-to-rank
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
TBD


## Benchmark
TBD


## Results

#### LightGBM
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

#### Catboost

|                        Anime Name                       |        Already Liked         | Predicted Score |
|:--------------------------------------------------------|:----------------------------:|:---------------:|
|                     Kirari Kagayaku                     |      Majo no Takkyuubin      |      4.234      |
|                       Oh Baby Plus                      |    Tenkuu no Shiro Laputa    |      4.152      |
|               Wanna. SpartanSex Spermax!!!              |       Pumpkin Scissors       |      4.152      |
|          Taiyou to Tsuki no Kodomo-tachi (2018)         |       Omoide Poroporo        |      4.071      |
|       Inazuma Eleven: Saikyou Gundan Ogre Shuurai       | Heisei Tanuki Gassen Ponpoko |      3.798      |
| Sylvanian Families Mini Gekijou: Omoigakenai Okyakusama |       Tonari no Totoro       |      3.798      |
|                    Shouwa Monogatari                    |         Zetsuai 1989         |      3.755      |
|                Kunimatsu-sama no Otoridai               |           Monster            |      3.639      |
|                Girls Bravo: Second Season               |         xxxHOLiC Kei         |      3.628      |
|                          Comics                         |       Shounen Onmyouji       |      3.599      |
