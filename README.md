[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
# Learning-to-rank
This Repository built a lambdarank algorithm for anime recommendations with gradient boosting algorithm.
The code style has been configured to use Black, and the maximum line length has been set to 120 characters.

## Dataset
I used [Anime Recommendation LTR](https://www.kaggle.com/datasets/ransakaravihara/anime-recommendation-ltr-dataset) in the Kaggle dataset.

## Requirements

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

It is recommended that latest version of poetry should be installed in advance.

```sh
$ poetry --version
Poetry (version 1.8.5)
```

Python version should be higher than `3.11`.

```sh
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```sh
$ poetry shell
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```sh
$ poetry install
```

### Note

If you want to add package to `pyproject.toml`, please use following command.

```sh
$ poetry add "package==1.0.0"
```

Then, update `poetry.lock` to ensure that repository members share same environment setting.

```sh
$ poetry lock
```

## Run code
CatBoost parameter Setting
```yaml
params:
  task_type: CPU
  loss_function: StochasticRank:metric=NDCG
  eval_metric: NDCG
  learning_rate: 0.05
  l2_leaf_reg: 0.02
  bagging_temperature: 3
  min_data_in_leaf: 57
  od_type: Iter
  iterations: 20000
  allow_writing_files: False
```

Running the learning code shell allows learning
```sh
$ sh scripts/run.sh
```

## Benchmark

|Algorithm|NDCG@20|NDCG@50|NDCG@100|
|:--------|:-----:|:-----:|:------:|
|LightGBM - LambdaMart|0.00788|0.00932|0.009782|
|LightGBM - XenDCG|0.00987|0.01113|0.01108|
|CatBoost - YetiRank|**0.01281**|0.01159|0.01121|
|CatBoost - LambdaMart|0.01196|**0.012353**|**0.01196**|

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



## What not worked
+ Stochastic Rank method took a lot of training time.


## Reference
+ [Which Tricks are Important for Learning to Rank?](https://openreview.net/pdf?id=MXfTQp8bZF)
+ [ARE NEURAL RANKERS STILL OUTPERFORMED BY GRADIENT BOOSTED DECISION TREES?](https://openreview.net/pdf?id=Ut1vF_q_vC)

## Lint setting
We use `ruff` for linting and code quality checks. Ruff is a fast Python linter, written in Rust, that integrates well with pre-commit hooks. It helps in maintaining code quality by enforcing coding standards and catching potential issues early in the development process.

To add `ruff` to your development dependencies, run the following command:

```sh
poetry add --group dev ruff pre-commit
```

Pre-commit settings can be configured in the `.pre-commit-config.yaml` file. Create the `.pre-commit-config.yaml` file and add the following configuration:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.4.0
  hooks:
  -   id: trailing-whitespace
  -   id: end-of-file-fixer
  -   id: check-yaml
-   repo: https://github.com/charliermarsh/ruff-pre-commit
  rev: v0.0.272
  hooks:
  -   id: ruff
    args: ["--fix"]
-   repo: https://github.com/psf/black
  rev: 23.3.0
  hooks:
  -   id: black
-   repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.3.0
  hooks:
  -   id: mypy
    args: [--ignore-missing-imports]
```

This configuration ensures that `ruff` is used to automatically fix linting issues, along with other useful hooks for maintaining code quality.
