mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;



fn main() {
    
  let e = Expr::Var("x".to_string());
  let e2 = Expr::Num(2.0);
  let e3 = Expr::Add(Box::new(e), Box::new(e2));

  let e4 = e3.subs("x", &Expr::Num(3.0));

  let res = e4.eval();

  println!("{}", res);
}