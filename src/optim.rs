use crate::math::Expr;
use crate::matrix::{self as mat, ColVec};

pub struct Optim {
  pub learning_rate: f64,
  pub max_iter: i32,

}

impl Optim {
  pub fn new(learning_rate: f64, max_iter: i32) -> Optim {
    Optim {
      learning_rate,
      max_iter
    }
  }

  /**
   * Uses gradient descent to find the minimizer of f.
   */
  pub fn optimize(&self, f: &Expr) -> ColVec {
    // get the set of variables in f
    let vars = f.vars();
    // make a vector for the minimizer
    // the length of the vector is the number of variables in f
    let mut minimizer = mat::ColVec::new(vec![0.0000001; vars.len()]);

    // iterate over the number of iterations

    let gradient_function = f.grad(vars.clone());

    println!("gradient function: {}", gradient_function);

    for _ in 0 .. self.max_iter {
      // compute the gradient of f at the minimizer
      // substitute the variables in the gradient with the values in the minimizer
      let mut grad = gradient_function.clone();

      for i in 0 .. vars.len() {
        grad = grad.subs(&vars[i].clone(), Expr::Num(minimizer[i]));
      }

      grad = grad.eval();

      println!("evaluated: {}", grad);

      // convert grad to a math matrix

      let grad = match grad.to_float_matrix() {
        Some(m) => m,
        None => panic!("unable to get evaluated gradient")
      };

      println!("evaluated gradient: \n{}", grad);

      // update the minimizer

      for i in 0 .. vars.len() {

        minimizer[i] = minimizer[i] - self.learning_rate * grad.data[i][0];
      }

      
      
    };

    minimizer


  }
}