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
    
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for i in 0 .. 20 {
        inputs.push(i as f64);
        outputs.push((2 * i + 3) as f64);
    }

    let mut inputs2 = Vec::new();
    inputs2.push(inputs);
    let mut outputs2 = Vec::new();
    outputs2.push(outputs);

    let x = Matrix::new(inputs2).t();
    let y = Matrix::new(outputs2).t();


    let lr = LinearRegression::new_optim(x, y);
   
    println!("w: \n{}", lr.w);
    println!("b: {}", lr.b);

    




}