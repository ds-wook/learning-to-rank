# %%
import numpy as np
import pandas as pd

# %%
path = "../input/anime-recommendation/"
anime_info_df = pd.read_csv(path + "anime_info.csv")
relavence_scores = pd.read_csv(path + "relavence_scores.csv")
user_info = pd.read_csv(path + "user_info.csv")

# %%
relavence_scores.head()
# %%
relavence_scores.info()
# %%
relavence_scores[relavence_scores["user_id"] == 11100]
# %%
relavence_scores[~(relavence_scores["user_id"] == 11100)]
# %%
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def create_genre_flags(df, popular_genres):
    df = df.dropna(subset=["Genres"])
    df["Genres"] = df["Genres"].apply(lambda x: ",".join(s.strip() for s in x.split(",")))
    # use MultiLabelBinarizer to create a one-hot encoded dataframe of the genres
    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb.fit_transform(df["Genres"].str.split(",")), columns=mlb.classes_, index=df.index)
    # create a new dataframe with the movie id and genre columns
    new_df = pd.concat([df["anime_id"], genre_df[popular_genres]], axis=1)
    new_df.columns = ["anime_id"] + popular_genres
    return new_df


popular_genres = [
    "Comedy",
    "Action",
    "Fantasy",
    "Adventure",
    "Kids",
    "Drama",
    "Sci-Fi",
    "Music",
    "Shounen",
    "Slice of Life",
]

anime_genre_info_df = create_genre_flags(anime_info_df, popular_genres)

# %%
anime_genre_info_df.head()
# %%

anime_info_df_final = anime_info_df.merge(anime_genre_info_df, on="anime_id")

# %%
anime_info_df_final
# %%
