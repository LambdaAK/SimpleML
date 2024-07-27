use crate::matrix::Matrix;

pub struct LinearRegression {
  x: Matrix,
  y: Matrix,
  w: Matrix
}

impl LinearRegression {
  pub fn new(x: Matrix, y: Matrix) -> LinearRegression {
    // x is a matrix where each row is a point
    // y is a matrix where each row is a label

    // compute the weight vector

    let lambda: f64 = 0.0001;

    let w = x.clone() * x.clone().t();

    println!("other: \n{}", lambda * Matrix::eye(x.rows()));
    
    let w = w + lambda * Matrix::eye(x.rows());

    println!("A");

    let w = w.inverse();

    println!("B");

    println!("w:\n{}", w);

    println!("w rows: {}", w.rows());
    println!("w cols: {}", w.cols());
    println!("y rows: {}", y.rows());
    println!("y cols: {}", y.cols());

    let w = y.clone().t() * w * x.clone();

    println!("C");

    println!("w: {}", w);

    LinearRegression {
      x : x.clone(),
      y : y.clone(),
      w
    }

  }
}