# shallweswim.today

Website to display swimming conditions at Coney Island / Brighton Beach

## Run locally (directly)

1. `poetry install`
1. `PORT=12345 poetry run python shallweswim/main.py`
1. Visit http://localhost:12345

## Run locally (via Docker)

1. `docker build -t shallweswim .`
1. `docker run -e PORT=80 -p 12345:80 shallweswim`
1. Visit http://localhost:12345

## Deploy

Hosted on Google Cloud Run

1. Run `./build_and_deploy.sh`

## Development

Setup with:

```
# Install poetry from its website (`brew install` version seems problematic on mac)
poetry run pre-commit install
```

Tools used:

- Format python code with `black`
- Type check with `mypy`
  `poetry run mypy --config-file=pyproject.toml .`
- HTML/MD/Yaml formatted with `prettier`
