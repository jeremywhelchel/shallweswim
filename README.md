# shallweswim.today
Website to display swimming conditions at Coney Island / Brighton Beach

## Dependencies

These can be installed with `pip3 install ...`

```
Flask
google-cloud-logging
gunicorn
pandas
seaborn
```

## Run locally
1. `PORT=12345 python3 main.py`
1. Visit http://localhost:12345

## Deploy

Hosted on Google Cloud Run

1. Run `./build_and_deploy.sh`

## Development

- Format with `black *.py`
- Type check with `pytype *.py`
