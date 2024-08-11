use crate::{math::{max, var, Expr}, matrix::Matrix, optim::Optim};



pub struct SVM {

  pub w: Matrix,
  pub b: f64

}

impl SVM {
  pub fn compute_loss_function(x: Matrix, y: Matrix, C: f64) -> Expr {

    // get the dimensionality of the points

    let num_points = x.rows();
    let num_features = x.cols();

    let points = x.rows_as_matrices();

    let mut squared_norm = Expr::Num(0.0);

    for i in 0 .. num_features {
      squared_norm = squared_norm + var(&format!("w{}", i + 1)).pow(Expr::Num(2.0));
    }
    
    squared_norm = 0.5 * squared_norm;

    let mut sum = Expr::Num(0.0);

    for i in 0 .. num_points {
      let point = &points[i];
      let label = y.get(i, 0);

      let mut term = Expr::Num(0.0);

      for j in 0 .. num_features {
        term = term + var(&format!("w{}", j + 1)) * point.get(0, j);
      }

      term = term + var("b");

      term = label * term;

      term = 1.0 - term;

      term = max(Expr::Num(0.0), term);

      sum = sum + term;

    }

    sum = C * sum;

    let loss_function = squared_norm + sum;
    
    loss_function

  }
}

impl SVM {

  pub fn new (x: Matrix, y: Matrix, C: f64) -> SVM {

    // each row of x is a training point
    // each row of y is the corresponding label

    let num_rows: usize = x.rows();
    let num_cols: usize = x.cols();

    let loss_function = Self::compute_loss_function(x, y, C);

    println!("loss function: {}", loss_function);


    // find the w and b that minimize the loss function

    let optimizer = Optim::new(0.00001, 10000);

    let minimizer = optimizer.optimize(&loss_function);

    let minimizing_bias = minimizer.get(0);

    let mut minimizing_weights = vec![];

    for i in 1 .. minimizer.rows() {
      minimizing_weights.push(minimizer.get(i));
    }

    println!("b: {}", minimizing_bias);

    let minimizing_weights = Matrix::new(vec![minimizing_weights]).t();

    println!("w: \n{}", minimizing_weights);

    SVM {
      w: minimizing_weights,
      b: minimizing_bias
    }

  }

  pub fn predict(&self, x: Matrix) -> f64 {
    (&self.w.t() * &x).get(0, 0) + self.b
  }

}