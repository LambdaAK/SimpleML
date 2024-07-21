use crate::matrix::Matrix;


struct Perceptron {
  weights: Matrix,
  bias: f64
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
      row.push(0.0);
      weights.push(row);
    }
    let weights = Matrix::new(weights);

    // make a bias
    let bias = 0.0;

    Perceptron {
      weights,
      bias
    }


  }

  
}