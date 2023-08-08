#include "nn.h"

double leaky_relu(double x);
double dleaky_relu(double x);
double relu(double x);
double drelu(double x);
double sigmoid(double x);
double dsigmoid(double x);
double softplus(double x);
double dsoftplus(double x);

struct Activation NN_LEAKY_RELU = {
    .func = leaky_relu,
    .dfunc = dleaky_relu
};

struct Activation NN_RELU = {
    .func = relu,
    .dfunc = drelu
};

struct Activation NN_SIGMOID = {
    .func = sigmoid,
    .dfunc = dsigmoid
};

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dsigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double relu(double x) { return (x > 0) ? x : 0; }
double drelu(double x) { return (x > 0) ? 1 : 0; }

double leaky_relu(double x) { return (x > 0) ? x : 0.01 * x; }
double dleaky_relu(double x) { return (x > 0) ? 1 : 0.01; }

double softplus(double x) { return log1p(exp(x)); }
double dsoftplus(double x) { return sigmoid(x); }
