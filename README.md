# ML

A simple neural network maker developed to train json data

## Requirements

* [openblas](https://www.openblas.net)
* [json-c](https://github.com/json-c/json-c)

## Installation

```
git clone https://git.juanvalencia.xyz/ml
cd ml
sudo make install
```

## Uninstall

```
sudo make uninstall
```

## Usage
```
Usage: ml train [Options] JSON_FILE
   or: ml predict [-o FILE] FILE
Train and predict json data

Options:
  -h, --help               Show this message
  -a, --alpha=ALPHA        Learning rate (only works with train)
  -e, --epochs=EPOCHS      Epochs to train the model (only works with train)
  -o, --output=FILE        Output file (only works with predict)

Examples:
  $ ml train -e 150 -a 1e-4 housing.json
  $ ml predict housing.json -o predictions.json
```
