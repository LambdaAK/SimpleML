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

    let x = matrix![
        [1., 2.],
        [-1., -2.],
        [0.5, 100.],
        [-1.5, 15.]
        
    ];

    let y = col_vec![1., -1., 1., -1.];

    let svm = SVM::SVM::new(x, y, 10.0);

    
    let input1 = col_vec![1., 2.];
    let input2 = col_vec![-1., -2.];
    let input3 = col_vec![0.5, 100.];
    let input4 = col_vec![-1.5, 15.];

    let prediction1 = svm.predict(input1);
    let prediction2 = svm.predict(input2);
    let prediction3 = svm.predict(input3);
    let prediction4 = svm.predict(input4);

    println!("prediction 1: {}", prediction1);
    println!("prediction 2: {}", prediction2);
    println!("prediction 3: {}", prediction3);
    println!("prediction 4: {}", prediction4);






}

/*
>>> np.linalg.eig(a)
EigResult(eigenvalues=array([3.06035199e-02, 4.27016678e+01]), eigenvectors=array([[-0.8927853 , -0.45048242],
       [ 0.45048242, -0.8927853 ]]))
*/