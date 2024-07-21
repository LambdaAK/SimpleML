mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;



fn main() {
  let input = "x ^ 3 + x ^ 2 + 2 * x + 1";
  let tokens = token::lex_tokens(input);
  let expr = parser::parse_l4(&tokens).0;

  println!("tokens: {:?}", tokens);
  println!("expr: {}", expr);

}