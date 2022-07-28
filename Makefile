install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C,broad-except,bare-except *.py

train-flaml:
	time flaml train --config config.yaml --dataset data/diabetes.csv

test:
	python -m pytest -vv testing_data.py

all: install format lint test
