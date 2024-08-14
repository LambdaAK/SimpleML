mod matrix;
use logistic_regression::LogisticRegression;
use lr::LinearRegression;
use math::Expr;
use matrix::{ColVec, Kernel};
use DecisionTree::Node;

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
mod NN;
mod KernelPerceptron;
mod DecisionTree;

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

    // Define the features matrix x using the matrix! macro
    let x = vec![
        ColVec::new(vec![1., 2.]),
        ColVec::new(vec![2., 3.]),
        ColVec::new(vec![3., 4.]),
        ColVec::new(vec![4., 5.]),
        ColVec::new(vec![5., 6.]),
        ColVec::new(vec![6., 7.]),
        ColVec::new(vec![7., 8.]),
        ColVec::new(vec![8., 9.]),

    ];

    let y = ColVec::new(vec![1., -1., 1., -1., 1., -1., 1., -1.]);

    let mut tree = Node::new(x, y);

    println!("{}", tree);

    


}