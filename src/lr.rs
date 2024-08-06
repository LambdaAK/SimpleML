use crate::{math::Expr, matrix::Matrix, optim::Optim};

///
/// A linear regression model that can be trained and used to predict labels in a regression problem
/// 
pub struct LinearRegression {
  x: Matrix,
  y: Matrix,
  pub w: Matrix,
  pub b: f64
}

impl LinearRegression {
  ///
  /// Create a new linear regression model from training data and labels
  /// Computes the weight vector using the normal equation (closed form)
  /// This is only efficient for relatively small datasets
  /// 
  pub fn new_closed(x: Matrix, y: Matrix) -> LinearRegression {
    // x is a matrix where each row is a point
    // y is a matrix where each row is a label

    // compute the weight vector

    let lambda: f64 = 0.0001;

    let mut x = x.clone();

    // add a column of ones to x on the right side

    for i in 0 .. x.rows() {
      x.data[i].push(1.0);
    }

    x.cols += 1;

    let w = x.clone() * x.clone().t();
    
    let w = w + lambda * Matrix::eye(x.rows());

    let w = w.inverse().unwrap(); // TODO: handle the None case

    let mut w = (y.clone().t() * w * x.clone()).t();

    // extract the bias from w
    
    // w is a vector with 1 column

    let b = w.get(w.rows() - 1, 0);

    // remove the bias from w

    w.data.pop();

    w.rows -= 1;

    LinearRegression {
      x : x.clone(),
      y : y.clone(),
      w,
      b
    }

  }
  
  ///
  /// Create a new linear regression model from training data and labels
  /// Computes the weight vector using gradient descent
  /// This is efficient for large datasets
  /// 
  pub fn new_optim(x: Matrix, y: Matrix) -> LinearRegression {
    // x is a matrix where each row is a point
    // y is a matrix where each row is a label

    
    let weight_length = x.cols();
    // the variables in the loss function will be b and w, ww, www, wwww, ....
    // b is the bias and w, ww, www, wwww, are the weights in the weight vector
    // initialize them to 0

    let points = x.rows_as_matrices();
    let labels = y.rows_as_matrices();

    let mut loss_function: Expr = Expr::Num(0.0);

    // for each point and label, add (w*x1 + w2*x2 + ... + b - y)^2 to the loss function

    for i in 0 .. points.len() {
      let point = &points[i];
      let label = labels[i].get(0, 0);

      // compute the prediction

      let mut term = Expr::Var("b".to_string());

      for j in 0 .. weight_length {
        term = term + Expr::Var(format!("w{}", j + 1)) * point.get(0, j);
      }

      term = term - Expr::Num(label);

      term = term.pow(Expr::Num(2.0));

      loss_function = loss_function + term;

    }
    
    println!("loss function: {}", loss_function);

    // find the minimizer of the loss function

    let optimizer = Optim::new(0.000001, 20000);

    let minimizer = optimizer.optimize(&loss_function);

    // extract the bias and weights from the minimizer vector

    let minimizing_bias = minimizer.get(0);

    let mut minimizing_weights = vec![];

    for i in 1 .. minimizer.rows() {
      minimizing_weights.push(minimizer.get(i));
    }

    LinearRegression {
      x,
      y,
      w: Matrix::new(vec![minimizing_weights]),
      b: minimizing_bias
    }

  }

  ///
  /// Predict the label of a point
  /// 
  pub fn predict(&self, x: Matrix) -> f64 {

    (&self.w.t() * &x).get(0, 0) + self.b
  }

}