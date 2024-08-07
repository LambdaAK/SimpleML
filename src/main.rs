mod matrix;
use logistic_regression::LogisticRegression;
use lr::LinearRegression;
use math::Expr;
use matrix::ColVec;
use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;
mod math;
mod token;
mod parser;
mod KNN;
mod lr;
mod optim;
use std::time::{SystemTime, UNIX_EPOCH};
mod logistic_regression;

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

macro_rules! row_vec {
  ( $( $x:expr ),* ) => {
      {
          let data = vec![
              vec![$($x),*]
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

macro_rules! col_vec {
  ( $( $x:expr ),* ) => {
      {
          let data = vec![
              $(
                  vec![$x],
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
        [1., 1.],
        [1., 2.],
        [2., 2.],
        [2., 3.],
        [3., 3.],
        [3., 4.],
        [50., 51.],
        [75., 101.],
        [51., 61.],
        [100., 101.]
    ];

    let y = col_vec![
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        1.,
        1.,
        1.,
        1.
    ];

    let lr = LogisticRegression::new(x.clone(), y);

    // classify each point in x

    for point in x.rows_as_matrices() {
        let prediction = lr.predict(point);
        println!("prediction: {}", prediction);
    }



    

}