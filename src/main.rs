// what we need to implement a simple MLP inference:
// - matrix multiplication
// - vector addition
// - linear layer with forward function
// - sequential network with forward function
// - activation function (relu, softmax)

#![allow(dead_code)]

mod matrix;
mod nn;

use matrix::Matrix;
use nn::{Layer, Linear, Sequential};

fn main() {
    let input = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
    let linear1 = Linear::new(3, 2);
    let linear2 = Linear::new(2, 1);
    let sequential = Sequential::from_layers(vec![Box::new(linear1), Box::new(linear2)]);
    let output = sequential.forward(input);
    println!("Result of sequential forward:");
    output.print();
}
