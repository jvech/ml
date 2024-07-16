#!/usr/bin/bash
echo 'Network Accuraccy'
echo '-----------------'
gnuplot tests/plots.gpi
echo ''

echo 'Benchmark test'
echo '-------------------'
time ./ml train -c tests/architectures/big_nn.cfg data/gauss2d.json > /dev/null
echo ''
