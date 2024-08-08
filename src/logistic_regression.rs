use crate::{math::{e, ln, var, Expr}, matrix::Matrix, optim::Optim};



pub struct LogisticRegression {
  x: Matrix,
  y: Matrix,
  w: Matrix,
  b: f64
}

impl LogisticRegression {
  fn compute_loss_function(x: Matrix, y: Matrix, lambda: f64) -> Expr {
    let mut loss_function = Expr::Num(0.0);

    let points = x.rows_as_matrices();

    let labels = y.rows_as_matrices().iter()
      .map(|m| m.get(0, 0)).collect::<Vec<f64>>();

    for i in 0 .. points.len() {
      let point = &points[i];
      let label = labels[i];

      let mut term = Expr::Num(0.);

      for j in 0 .. point.cols() {
        term = term + var(&format!("w{}", j + 1)) * point.get(0, j);
      }

      term = term + var("b"); // w^T * x + b

      term = term * Expr::Num(-1.0); // - (w^T * x + b)

      term = e().pow(term); // exp(-(w^T * x + b))

      term = Expr::Num(1.0) + term; // 1 + exp(-(w^T * x + b))

      term = 1.0 / term; // 1 / (1 + exp(-(w^T * x + b)))

      if label == 1.0 {
        println!("a");
        term = ln(term);
      }
      else {
        println!("b");
        term = ln(1.0 - term);
      }

      loss_function = loss_function + term;

    }

    loss_function = loss_function / Expr::Num(points.len() as f64) * Expr::Num(-1.0);

    // add the regularization term, if lambda is not 0

    if lambda <= 0.0 {
      return loss_function
    }

    let mut reg_term = Expr::Num(0.0);

    for i in 0 .. x.cols() {
      reg_term = reg_term + var(&format!("w{}", i + 1)).pow(Expr::Num(2.0));
    }

    reg_term = lambda * reg_term;

    loss_function = loss_function + reg_term;

    loss_function
  } 

  /**
   * 
   * L(w, b) = (-1)/n * sum(i = 1, n, y_i * ln(1 / (1 + exp(-(w^T * x_i + b))) + (1 - y_i) * ln(1 - 1 / (1 + exp(-(w^T * x_i + b)))) + lambda/2 * ||w||^2
   */


  pub fn new(x: Matrix, y: Matrix, lambda: f64) -> LogisticRegression {
    
    let loss_function = Self::compute_loss_function(x.clone(), y.clone(), lambda);

    println!("loss function: {}", loss_function);

    //panic!();


    let optimizer = Optim::new(0.2, 1500);

    let minimizer = optimizer.optimize(&loss_function);

    let minimizing_bias = minimizer.get(0);

    let mut minimizing_weights = vec![];

    for i in 1 .. minimizer.rows() {
      minimizing_weights.push(minimizer.get(i));
    }

    LogisticRegression {
      x,
      y,
      w: Matrix::new(vec![minimizing_weights]),
      b: minimizing_bias
    }

  }

  fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
  }

  pub fn predict(&self, x: Matrix) -> f64 {
    let inner_prod = (&self.w.t() * &x).get(0, 0);
    let term = inner_prod + self.b;

    let pred = Self::sigmoid(term);

    println!("prediction: {}", pred);

    if pred >= 0.5 {
      1.0
    }
    else {
      0.0
    }
    
  }

}