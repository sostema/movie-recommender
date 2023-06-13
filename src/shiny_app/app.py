import pandas as pd
import shinyswatch
import torch
from shiny import App, reactive, render, ui

from movielens_ncf.model import NCF

ratings = pd.read_csv("data/ml-20m/ratings.csv")
users = ratings["userId"].unique()

movies = pd.read_csv("data/ml-20m/movies.csv")
movies = movies.set_index("movieId")
movies = movies.rename(columns={"title": "Movie Title", "genres": "Genres"})


def load_model() -> NCF:
    model_checkpoint: str = "trained_models/last.ckpt"
    model = NCF.load_from_checkpoint(model_checkpoint)
    model.eval()
    return model


model = load_model()


app_ui = ui.page_fluid(
    shinyswatch.theme.minty(),
    ui.input_text("userID", label="User ID", value=str(users[0])),
    ui.input_action_button("submit", label="Recommend movies for this user"),
    ui.row(
        ui.column(
            6,
            ui.output_table("histable", label="Historic Movies"),
        ),
        ui.column(
            6,
            ui.output_table("rectable", label="Recommended Movies"),
        ),
    ),
    title="Movie Recommender System",
)


def server(input, output, session):
    @output
    @render.table
    @reactive.event(input.submit)
    def histable():
        if int(input.userID()) not in users:
            return None
        else:
            return (
                movies.loc[
                    ratings[ratings["userId"] == int(input.userID())]
                    .merge(movies, on="movieId", how="inner")["movieId"]
                    .unique()
                ]
                .style.set_table_attributes('class="dataframe shiny-table table w-auto"')
                .hide(axis="index")
            )

    @output
    @render.table
    @reactive.event(input.submit)
    def rectable():
        if int(input.userID()) not in users:
            return None
        else:
            all_movies = set(movies.index.to_numpy().tolist())
            interacted_items = set(ratings[ratings["userId"] == int(input.userID())]["movieId"].to_numpy().tolist())
            not_interacted_items = set(all_movies) - set(interacted_items)
            movie_input = list(not_interacted_items)
            user_input = [int(input.userID())] * len(movie_input)
            with torch.no_grad():
                predictions: torch.Tensor = model(
                    torch.tensor(user_input).to(model.device), torch.tensor(movie_input).to(model.device)
                ).squeeze()
            indices: torch.Tensor
            _, indices = torch.topk(predictions, 10)
            recommended_movie_df: pd.DataFrame = movies.iloc[indices.detach().cpu().numpy()]
            return recommended_movie_df.style.set_table_attributes('class="dataframe shiny-table table w-auto"').hide(
                axis="index"
            )


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
