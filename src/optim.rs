use crate::math::Expr;
use crate::matrix::{self as mat, ColVec};

///
/// A multivariate function optimizer that uses gradient descent to find the minimizer of a function
pub struct Optim {
  pub learning_rate: f64,
  pub max_iter: i32,

}

impl Optim {
  ///
  /// Create a new optimizer with a learning rate and maximum number of iterations
  /// 
  /// # Arguments
  /// 
  /// `learning_rate` - The learning rate for the optimizer
  /// `max_iter` - The maximum number of iterations to run the optimizer for
  pub fn new(learning_rate: f64, max_iter: i32) -> Optim {
    Optim {
      learning_rate,
      max_iter
    }
  }

  ///
  /// Optimize a function using gradient descent
  /// 
  /// # Arguments
  /// 
  /// `f` - The function to optimize
  /// 
  /// # Returns
  /// 
  /// The computed minimizer of the function
  pub fn optimize(&self, f: &Expr) -> ColVec {
    println!("optimizing function: {}", f);
    // get the set of variables in f
    let vars = f.vars();
    // make a vector for the minimizer
    // the length of the vector is the number of variables in f
    let mut minimizer = mat::ColVec::new(vec![0.0000001; vars.len()]);

    // iterate over the number of iterations

    let gradient_function = f.grad(vars.clone());

    for i in 0 .. self.max_iter {
      // compute the gradient of f at the minimizer
      // substitute the variables in the gradient with the values in the minimizer
      let mut grad = gradient_function.clone();

      for i in 0 .. vars.len() {
        grad = grad.subs(&vars[i].clone(), Expr::Num(minimizer[i]));
      }

      grad = grad.eval();

      // convert grad to a math matrix

      let grad = match grad.to_float_matrix() {
        Some(m) => m,
        None => panic!("unable to get evaluated gradient")
      };

      println!("gradient: \n{}", grad);

      // update the minimizer

      for i in 0 .. vars.len() {

        minimizer[i] = minimizer[i] - self.learning_rate * grad.data[i][0];
      }

      println!("iteration: {}", i);
      println!("minimizer: \n{}", minimizer);

      // substitute the variables in f with the values in the minimizer

      let mut f_eval = f.clone();

      for i in 0 .. vars.len() {
        f_eval = f_eval.subs(&vars[i].clone(), &Expr::Num(minimizer[i]));
      }

      // evaluate f

      let f_eval = f_eval.eval();

      println!("f: {}", f_eval)
      
      
    };

    minimizer

  }
}