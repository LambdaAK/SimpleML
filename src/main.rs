mod matrix;
use math::Expr;

use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;


fn main() {
  let input = "a ^ 2 + ln(b ^ 1000)";
  let tokens = token::lex_tokens(input);
  let (expr, _) = parser::parse_l5(&tokens);

  println!("Expr: {}", expr);
  println!("Gradient\n{}", expr.grad().subs(&"a".to_string(), Expr::Num(1.)).subs(&"b".to_string(), Expr::Num(2.)).eval());

}