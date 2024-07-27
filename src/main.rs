mod matrix;
use math::Expr;
use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;
mod KNN;
mod lr;


// Macro to create a matrix
macro_rules! matrix {
  ( $( [ $( $x:expr ),* ] ),* ) => {
      {
          let data = vec![
              $(
                  vec![$($x),*],
              )*
          ];
          let rows = data.len();
          let cols = if rows > 0 { data[0].len() } else { 0 };
          Matrix {
              data,
              rows,
              cols,
          }
      }
  };
}

fn main() {

  let x = matrix![
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0]
  ];

  let y = matrix![
    [1.1],
    [2.05],
    [3.21],
    [4.1],
    [4.9999],
    [6.0],
    [7.0],
    [8.0]
  ];

  let lr = lr::LinearRegression::new(x, y);
  
}