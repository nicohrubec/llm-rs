// what we need to implement a simple MLP inference:
// - matrix multiplication
// - vector addition
// - linear layer with forward function
// - sequential network with forward function
// - activation function (relu, softmax)

mod matrix;

use matrix::Matrix;

fn main() {
    let a = Matrix::random(3, 4);
    let b = Matrix::random(4, 3);
    let c = a * b;
    println!("Result of matrix multiplication:");
    c.print();
}
