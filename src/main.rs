// plan:
// 1. get single token generation working
//   - be able to generate a set of tokens given input
//   - gpt2 model
//     - token embedding
//     - positional encoding
//     - transformer block
//       - layer norm
//       - scaled dot product attention
//       - multi-head attention
//       - feed forward network
//   - then these need to be stacked to form the transformer model
//   - head
//     - linear layer with weights loaded from file
//     - softmax
// 2. correctness/performance checks
//   - load pretrained weights into architecture
//   - load a pytorch model with the same weights + architecture
//   - compare outputs to ensure they are the same
//   - time inference performance and compare (also cpu vs gpu)
// 3. get cpu performance close to pytorch model
//   - get profiles
//   - optimizations (kv cache,memory, simd, multithreading, quantization, etc.)
// 4. get gpu performance close to pytorch model with custom cuda kernels
// 5. batch inference

#![allow(dead_code)]

mod matrix;
mod nn;

use matrix::Matrix;
use nn::{Layer, Linear, ReLU, Sigmoid, Sequential};
use std::time::Instant;

fn main() {
    let input = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
    let linear1 = Linear::new(3, 2);
    let linear2 = Linear::new(2, 1);
    let sequential = Sequential::from_layers(vec![
        Box::new(linear1),
        Box::new(ReLU),
        Box::new(linear2),
        Box::new(Sigmoid),
    ]);
    println!("Sequential network created");

    let start = Instant::now();
    let output = sequential.forward(input);
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);
    println!("Result of sequential forward:");
    output.print();
}
