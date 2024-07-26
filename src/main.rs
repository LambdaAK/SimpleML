mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;


fn main() {
  let input = "e ^ x";
  let tokens = token::lex_tokens(input);
  let (expr, _) = parser::parse_l5(&tokens);

  // [1, 2, 3, 4, 5]
  // make that vector

  
}