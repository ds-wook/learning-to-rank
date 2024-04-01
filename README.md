[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
# learning-to-rank
This Repository built a lambdarank algorithm for anime recommendations with gradient boosting algorithm.
The code style has been configured to use Black, and the maximum line length has been set to 120 characters.

## Dataset
+ [Anime Recommendation LTR](https://www.kaggle.com/datasets/ransakaravihara/anime-recommendation-ltr-dataset)

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
|Anime Name|Already Liked|Predicted Score|
|:-----------------------------|:----------------------------:|:---------------:|
|         Shukka Fuufu         |      Majo no Takkyuubin      |      3.025      |
|            Dallos            |       Pumpkin Scissors       |      3.025      |
|         Inori no Te          |       Omoide Poroporo        |      2.967      |
|     Hortensia Saga (TV)      | Heisei Tanuki Gassen Ponpoko |      2.967      |
|  Top Secret: The Revelation  |       Tonari no Totoro       |      2.828      |
|       Junlin Chengxia        |         Zetsuai 1989         |      2.828      |
|        Chargeman Ken!        |           Monster            |      2.828      |
| Miss Monochrome: Music Clips |         xxxHOLiC Kei         |      2.828      |
|          Kabi Usagi          |       Shounen Onmyouji       |      2.739      |
