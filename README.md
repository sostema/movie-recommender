# Movie Recommender

Сервис для подбора фильмов на базе данных MovieLens 20M. Использует NCF модель для рекомендаций.

## Структуруа проекта

- `data/` - папка с данными
- `trained_models/` - папка с моделями
- `src/` - папка с исходным кодом
- `src/shiny_app/` - папка с веб-приложением на базе Shiny
- `src/movielens_ncf/` - папка с кодом модели NCF
- `logs/` - папка с логами
- `pyproject.toml` - файл проекта

## Установка

1. Скачать репозиторий `git clone git@github.com:sostema/movie-recommender.git`
2. Установить проект и зависимости `pip install .`

## Запуск

1. Скачать данные с [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) и распаковать в папку `data/`
2. Запустить `src/movielens_ncf/train.py` для обучения модели
3. Запустить `src/shiny_app/app.py` для запуска веб-приложения

## Результаты

Результаты обучения модели можно посмотреть в логах Tensorboard: `logs/`
