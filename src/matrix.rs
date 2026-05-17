#![allow(dead_code)]

use std::ops::Mul;
use std::ops::Add;
use rand::RngExt;

pub struct Matrix {
    pub data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
        Self { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }

    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    pub fn print(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{} ", self.at(row, col));
            }
            println!();
        }
    }
}

// matrix multiplication
impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "dimension mismatch");
        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.at(i, k) * other.at(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }
}

// matrix addition
impl Add for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "dimension mismatch");
        assert_eq!(self.cols, other.cols, "dimension mismatch");
        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.at(i, j) + other.at(i, j));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_random() {
        let matrix = Matrix::random(3, 4);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data.len(), 3 * 4);

        for value in matrix.data.iter() {
            assert!(*value >= -1.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_matrix_zeros() {
        let matrix = Matrix::zeros(3, 4);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data.len(), 3 * 4);

        for value in matrix.data.iter() {
            assert_eq!(*value, 0.0);
        }
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let matrix = Matrix::from_vec(data, 3, 4);
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data.len(), 3 * 4);
    }

    #[test]
    fn test_matrix_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let matrix = Matrix::from_vec(data, 3, 4);
        assert_eq!(matrix.at(0, 0), 1.0);
        assert_eq!(matrix.at(0, 1), 2.0);
    }

    #[test]
    fn test_matrix_set() {
        let mut matrix = Matrix::random(3, 4);
        matrix.set(0, 0, 1.0);
        assert_eq!(matrix.at(0, 0), 1.0);
    }

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Matrix::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = a * b;
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.data.len(), 2 * 2);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matrix_add() {
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = Matrix::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
        let c = a + b;
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 3);
        assert_eq!(c.data.len(), 2 * 3);
        assert_eq!(c.data, vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }
}
