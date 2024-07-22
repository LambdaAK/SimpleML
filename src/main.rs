mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;


fn main() {
  let input = "2 ^ ln(ln(x + y + 2 * z))";
  let tokens = token::lex_tokens(input);
  let (expr, _) = parser::parse_l5(&tokens);

  println!("Expr: {}", expr);
  println!("Gradient\n{}", expr.grad().eval());

}