# ML

A simple neural network maker developed to train json data

## Requirements

* [openblas](https://www.openblas.net)
* [json-c](https://github.com/json-c/json-c)

## Installation

```
git clone https://git.juanvalencia.xyz/ml
cd ml
make
sudo make install
make install_config
```

## Uninstall

```
sudo make uninstall
```

## Usage

```
Usage: ml [re]train [Options] FILE
   or: ml predict [-Ohv] [-f FORMAT] [-o FILE] [-p INT] FILE

Options:
  -h, --help               Show this message
  -f, --format=FORMAT      Define input or output FILE format if needed
  -a, --alpha=ALPHA        Learning rate (only works with train)
  -e, --epochs=EPOCHS      Epochs to train the model (only works with train)
  -o, --output=FILE        Output file (only works with predict)
  -O, --only-out           Don't show input fields (only works with predict)
  -c, --config=FILE        Configuration filepath [default=~/.config/ml/ml.cfg]
  -p, --precision=INT      Decimals output precision (only works with predict)
                           [default=auto]



Examples:
  $ ml train -e 150 -a 1e-4 housing.json
  $ ml predict housing.json -o predictions.json
```
