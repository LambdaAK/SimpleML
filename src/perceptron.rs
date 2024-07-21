use crate::matrix::Matrix;


pub struct Perceptron {
  pub weights: Matrix
}

impl Perceptron {
  pub fn new(x: Matrix, y: Matrix) -> Perceptron {
    // each row of x is a training point
    // each row of y is the corresponding label
    let num_rows: usize = x.rows();
    let num_cols: usize = x.cols();

    // make a weight vector (column vector)
    let mut weights: Vec<Vec<f64>> = Vec::new();
    for _ in 0..num_cols {
      let mut row: Vec<f64> = Vec::new();
      row.push(1.0);
      weights.push(row);
    }
    let weights_without_bias = Matrix::new(weights);

    // stack a bias (0) to the bottom of the weight vector
    
    let bias_matrix: Matrix = Matrix::from_2d(&[[0.0]]);

    let weights = weights_without_bias.stack(&bias_matrix);

    Perceptron { weights }

  }

  fn train(&self) {
    // train the model


  }

  pub fn predict(&self, x: Matrix) -> Matrix {
    // predict the labels of the input data
    // x is a matrix where each row is a training point
    // the output is a matrix where each row is the predicted label of the corresponding training point

    /*
      | x1 |
      | x2 |
      | .. |
      | xn |

      | w1 |
      | w2 |
      | .. |
      | wn |
      | b  |
     */

    // we need to add a column of 1s to the input data
    
    let x_t = x.t();
    
    // make a matrix of 1s that has the same number of rows as x_t

    let ones: Vec<Vec<f64>> = vec![vec![1.0; x_t.cols()]; 1];

    let ones_matrix = Matrix::new(ones);

    // stack the ones matrix on the bottom of x_t

    let x_t_with_bias = x_t.stack(&ones_matrix);

    // tranpose it again

    let new_x = x_t_with_bias.t();


    &new_x * &self.weights

  }
}