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

/*
QRResult(Q=array([
    [-0.12309149,  0.90453403,  0.40824829],
    [-0.49236596,  0.30151134, -0.81649658],
    [-0.86164044, -0.30151134,  0.40824829]]), 
    
    R=array(
    [[-8.12403840e+00, -9.60113630e+00, -1.10782342e+01],
    [ 0.00000000e+00,  9.04534034e-01,  1.80906807e+00],
    [ 0.00000000e+00,  0.00000000e+00, -7.58790979e-16]]))

*/


fn main() {
    
    let x = matrix![
        [1., 1., 1., 1.],
        [2., 2., 2.01, 2.025],
        [3.1, 3.0001, 3.001, 3.0000021],
        [4.1, 4.0001, 4.001, 4.0000021],
        [5.1, 5.0001, 5.001, 5.0000021],
        [6.1, 6.0001, 6.001, 6.0000021],
        [7.1, 7.0001, 7.001, 7.0000021],
        [8.1, 8.0001, 8.001, 8.0000021],
        [9.1, 9.0001, 9.001, 9.0000021],
        [10.1, 10.0001, 10.001, 10.0000021]
    ];

    let c = x.center_data();

    let c = c.cov_matrix();

    println!("{}", c);

    println!("{}", c.rref());

    let eigenspaces = c.eig();

    for eigenspace in eigenspaces {
        let lambda = eigenspace.eigenvalue;
        
        println!("Eigenvalue: {}", lambda);

        println!("\nEigenbasis:");

        for v in eigenspace.basis {
            println!("{}", v);
        }
    }


}