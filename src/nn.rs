use crate::matrix::Matrix;

pub struct Linear {
    weights: Matrix,
    bias: Matrix,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Matrix::random(input_size, output_size);
        let bias = Matrix::zeros(1, output_size);
        Self { weights, bias }
    }

    pub fn forward(self, input: Matrix) -> Matrix {
        input * self.weights + self.bias
    }
}