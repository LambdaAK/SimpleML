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
mod SVM;

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

    // seperable by x + 1

    let x = matrix![
        [1., 4.],
        [2., 5.5],
        [3., 6.7],
        [4., 8.2],
        [5., 9.1],

        [1., -0.5],
        [2., 0.2],
        [3., 1.1],
        [4., 2.3],
        [5., 3.4]
        
    ];


    let y = col_vec![
        1.,
        1.,
        1.,
        1.,
        1.,
        -1.,
        -1.,
        -1.,
        -1.,
        -1.
    ];

    let svm = SVM::SVM::new(x.clone(), y.clone(), 10.0);
    let p = Perceptron::new(x.clone(), y);

    
    // classify all of those points

    let points = x.rows_as_matrices();


    println!("Perceptron predictions");

    for i in 0..points.len() {
        let point = &points[i];
        let label = p.predict(&point);
        println!("{}: {}", i, label);
    }


    println!("SVM predictions");
    for i in 0..points.len() {
        let point = &points[i];
        let label = svm.predict(point.t());
        println!("{}: {}", i, label);
    }


}