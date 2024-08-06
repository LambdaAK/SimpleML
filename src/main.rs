mod matrix;
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
    
    let mut x_points = Vec::new();

    for i in 0 .. 20 {
        x_points.push(vec![i as f64]);
    }

    let x = Matrix::new(x_points);

    let mut y_points = Vec::new();

    for i in 0 .. 20 {
        y_points.push(vec![(2 * i + 5) as f64]);
    }

    let y = Matrix::new(y_points);

    println!("x: \n{}", x);
    println!("y: \n{}", y);

    let lr = LinearRegression::new_optim(x, y, 0.0002, 1000000000);

    println!("w: {}", lr.w);
    println!("b: {}", lr.b);
}