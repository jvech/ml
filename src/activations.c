/**
 * ml - a neural network processor written with C
 * Copyright (C) 2023  jvech
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <math.h>

#include "nn.h"

double leaky_relu(double x);
double dleaky_relu(double x);
double relu(double x);
double drelu(double x);
double sigmoid(double x);
double dsigmoid(double x);
double softplus(double x);
double dsoftplus(double x);
double linear(double x);
double dlinear(double x);

struct Activation NN_LEAKY_RELU = {
    .func = leaky_relu,
    .dfunc = dleaky_relu
};

struct Activation NN_SOFTPLUS = {
    .func = softplus,
    .dfunc = dsoftplus
};

struct Activation NN_RELU = {
    .func = relu,
    .dfunc = drelu
};

struct Activation NN_SIGMOID = {
    .func = sigmoid,
    .dfunc = dsigmoid
};

struct Activation NN_LINEAR = {
    .func = linear,
    .dfunc = dlinear,
};

double linear(double x) {return x;}
double dlinear(double x) {return 1.0;}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dsigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double relu(double x) { return (x > 0) ? x : 0; }
double drelu(double x) { return (x > 0) ? 1 : 0; }

double leaky_relu(double x) { return (x > 0) ? x : 0.01 * x; }
double dleaky_relu(double x) { return (x > 0) ? 1 : 0.01; }

double softplus(double x) { return log1p(exp(x)); }
double dsoftplus(double x) { return sigmoid(x); }
