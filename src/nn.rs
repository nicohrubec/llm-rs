use crate::matrix::Matrix;

pub trait Layer {
    fn forward(&self, input: Matrix) -> Matrix;
}

pub struct ReLU;
impl Layer for ReLU {
    fn forward(&self, input: Matrix) -> Matrix {
        Matrix::from_vec(input.data.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect(), input.rows, input.cols)
    }
}

pub struct Sigmoid;
impl Layer for Sigmoid {
    fn forward(&self, input: Matrix) -> Matrix {
        Matrix::from_vec(input.data.iter().map(|x| 1.0 / (1.0 + f64::exp(-*x))).collect(), input.rows, input.cols)
    }
}

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
}

impl Layer for Linear {
    fn forward(&self, input: Matrix) -> Matrix {
        input * &self.weights + &self.bias
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn from_layers(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }
}

impl Layer for Sequential {
    fn forward(&self, input: Matrix) -> Matrix {
        let mut output = input;
        for layer in self.layers.iter() {
            output = layer.forward(output);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(3, 2);
        let input = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let output = linear.forward(input);
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 2);
        assert_eq!(output.data.len(), 1 * 2);
    }

    #[test]
    fn test_sequential_forward() {
        let linear1 = Linear::new(3, 2);
        let linear2 = Linear::new(2, 1);
        let sequential = Sequential::from_layers(vec![Box::new(linear1), Box::new(linear2)]);

        let input = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let output = sequential.forward(input);
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 1);
        assert_eq!(output.data.len(), 1);
    }
    
    #[test]
    fn test_relu_forward_positive() {
        let relu = ReLU;
        let input = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let output = relu.forward(input);
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 1 * 3);
        assert_eq!(output.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_relu_forward_negative() {
        let relu = ReLU;
        let input = Matrix::from_vec(vec![-1.0, -2.0, -3.0], 1, 3);
        let output = relu.forward(input);
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 3);
        assert_eq!(output.data.len(), 1 * 3);
        assert_eq!(output.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid;
        let input = Matrix::from_vec(vec![0.0, 1.0, 2.0, 3.0], 1, 4);
        let output = sigmoid.forward(input);
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 4);
        assert_eq!(output.data.len(), 1 * 4);
        assert_eq!(output.data, vec![0.5, 0.7310585786300049, 0.8807970779778823, 0.9525741268224334]);
    }
}