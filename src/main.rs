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
    
    let v1 = ColVec::new(vec![1.0, 2.0, 1., 100.1]);
    let v2 = ColVec::new(vec![4.0, 1000.1, 1., 100.2]);
    let v3 = ColVec::new(vec![4.0, 10000.12, 1., -100.1123]);
    let v4 = ColVec::new(vec![4.0, 10000.12, 1., -1347134.]);

    let vectors = vec![v1, v2, v3, v4];

    let ortho = ColVec::orthonormalize(vectors);

    for v in &ortho {
        println!("{}", v);
    }

    // for each pair of vectors, compute their dot product and verify that it's 0

    for i in 0 .. ortho.len() {
        for j in i + 1 .. ortho.len() {
            let dot_product = ortho[i].dot(&ortho[j]);
            println!("dot product: {}", dot_product);
        }
    }

    // for each vector, verify that its norm is 1

    for v in &ortho {
        println!("norm: {}", v.norm());
    }


}