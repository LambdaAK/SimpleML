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
        ColVec::new(vec![1., 2., 3.]),
        ColVec::new(vec![4., 5., 6.]),
        ColVec::new(vec![7., 8., 9.]),
        ColVec::new(vec![10., 11., 12.]),
        ColVec::new(vec![13., 14., 15.]),
        ColVec::new(vec![16., 17., 18.]),
        ColVec::new(vec![19., 20., 21.]),
        ColVec::new(vec![22., 23., 24.]),
        ColVec::new(vec![25., 26., 27.]),
        ColVec::new(vec![28., 29., 30.]),
        ColVec::new(vec![31., 32., 33.]),
        // many more points

        ColVec::new(vec![34., 35., 36.]),
        ColVec::new(vec![37., 38., 39.]),
        ColVec::new(vec![40., 41., 42.]),
        ColVec::new(vec![43., 44., 45.]),

    ];

    let y = ColVec::new(vec![1., 1., 1., 1., 1., 1., -1., -1., -1., -1., -1., -1., 1., 1., -1.]);

    let tree = Node::new(x.clone(), y);

    // predict the class of each point in the dataset

    let test = vec![
        ColVec::new(vec![15.1, 16.02]),
    ];

    for i in 0..test.len() {
        println!("Prediction for \n{} is {}", test[i], tree.predict(test[i].clone()));
    }

}