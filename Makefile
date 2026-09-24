ENV ?= roofseg

.PHONY: env test

env:
	conda env create -f environment.yml -n $(ENV)

test:
	python -m unittest discover -s tests -v
