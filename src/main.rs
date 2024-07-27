mod matrix;
use math::Expr;
use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;
mod KNN;

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


  // make a 2 by 2 matrix

  let xTr = matrix![
    [0.0, 0.0, 1.1, 2.2],
    [1.0, 1.0, 1.1, 2.2],
    [2.0, 2.0, 1.1, 2.2],
    [3.0, 3.0, 1.1, 2.2],
    [4.0, 4.0, 1.1, 2.2],
    [5.0, 5.0, 1.1, 2.2],
    [6.0, 6.0, 1.1, 2.2],
    [7.0, 7.0, 1.1, 2.2],
    [8.0, 8.0, 1.1, 2.2],
    [9.0, 9.0, 1.1, 2.2]
  ];

  let yTr = matrix![
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0]
  ];

  println!("xTr: \n{}", xTr);
  println!("yTr: \n{}", yTr);

  let m : KNN::KNN = KNN::KNN::new(1, xTr, yTr);

  let test_point = matrix![
    [1000000.0, 0.0, 1.1, 2.2]
  ];

  let prediction: f64 = m.predict(test_point);

  println!("prediction: {}", prediction);


  
}