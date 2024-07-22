mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;


fn main() {
  let input = "x ^ (2 * x)";
  let tokens = token::lex_tokens(input);
  let expr = parser::parse_l4(&tokens).0.eval();

  let derivative = expr.diff(&"x".to_string());

  println!("Expression: {}", expr);
  println!("Derivative: {}", derivative.subs(&"x".to_string(), &Expr::Num(2.0)).eval());
}