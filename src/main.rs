mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;



fn main() {
  let input = "2 + 2 * 2 ^ ^ ^ ^  * * * * 1";
  let tokens = token::lex_tokens(input);

  println!("tokens: {:?}", tokens)

}