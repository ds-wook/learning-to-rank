import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def create_genre_flags(df: pd.DataFrame, popular_genres: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=["Genres"])
    df.loc[:, "Genres"] = df["Genres"].apply(lambda x: ",".join(s.strip() for s in x.split(",")))

    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb.fit_transform(df["Genres"].str.split(",")), columns=mlb.classes_, index=df.index)

    new_df = pd.concat([df["anime_id"], genre_df[popular_genres]], axis=1)
    new_df.columns = ["anime_id"] + popular_genres
    return new_df
