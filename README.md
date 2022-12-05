# Equivariant_Go

This repository extends the convolutional neural networks built in the textbook [Deep Learning and the Game of Go](https://github.com/maxpumperla/deep_learning_and_the_game_of_go) to play Go.

We construct custom layers in Keras which have the 4-fold symmetry of the Go Board explicitly built into the network, using the work of [Cohen and Welling](https://arxiv.org/abs/1602.07576), hopefully providing more efficient training and better generalization.
