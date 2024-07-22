mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;


fn main() {
  let input = "ln(x * x) * ln(2 * x)";
  let tokens = token::lex_tokens(input);
  let (expr, _) = parser::parse_l5(&tokens);

  println!("Expr: {}", expr);

  let derivative = expr.diff(&"x".to_string());

  println!("Derivative: {}", derivative);

}