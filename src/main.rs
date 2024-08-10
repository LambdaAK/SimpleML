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

    // y = 2x + 1
    
    let x = matrix![
        [1.646, 3.3246, 1.11341414],
        [2.0877175153, 5., 1.13413414],
        [3.242424525, 7.1353, 1.36336],
        [4.245245, 9.46464644786, 1.313414],
        [5., 11.245245245, 1.0008765567865],
        [6.46848486, 13.1351351351555, 1.689699699],
        [7., 15.06957, 1.355],
        [8.46747, 17.4674747, 1.689969],
        [9.4684848, 19.4848, 1.00689689],
        [10.68, 21.46846846846848468, 1.13589035]
    ];

    let c = x.center_data();

    let c = c.cov_matrix();

    println!("{}", c);

    println!("{}", c.rref());

    let (eigenvalues, eigenvectors) = c.eig();

    println!("eigenvalues");
    
    for i in 0 .. eigenvalues.len() {
        let l = eigenvalues[i];
        let v = eigenvectors[i].clone();

        println!("l: {}", l);
        println!("v: {}", v);

        let c_times_v = c.clone() * v.clone().as_matrix();

        println!("c_times_v: {}", c_times_v);

        let l_times_v = l * v.clone();

        println!("l_times_v: {}", l_times_v);




    }




}

/*
>>> np.linalg.eig(a)
EigResult(eigenvalues=array([3.06035199e-02, 4.27016678e+01]), eigenvectors=array([[-0.8927853 , -0.45048242],
       [ 0.45048242, -0.8927853 ]]))
*/