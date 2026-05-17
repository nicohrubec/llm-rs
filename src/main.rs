// what we need to implement a simple MLP inference:
// - matrix multiplication
// - vector addition
// - linear layer with forward function
// - sequential network with forward function
// - activation function (relu, softmax)

mod matrix;
mod nn;

use matrix::Matrix;
use nn::Linear;

fn main() {
    let a = Matrix::random(3, 4);
    let b = Matrix::random(4, 3);
    let c = a * b;
    println!("Result of matrix multiplication:");
    c.print();

    let linear = Linear::new(3, 2);
    let input = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
    let output = linear.forward(input);
    println!("Result of linear forward:");
    output.print();
}
