mod matrix;
use crate::matrix::Matrix;
mod perceptron;


fn arr_to_vec(arr: &[f64]) -> Vec<f64> {
  let mut vec = Vec::new();
  for i in arr.iter() {
    vec.push(*i)
  }
  return vec
}


fn arr_2d_to_matrix<const ROWS: usize, const COLS: usize>(arr: &[[f64; COLS]; ROWS]) -> Matrix {
  let mut vec = Vec::new();
  for row in arr.iter() {
      vec.push(arr_to_vec(row));
  }
  Matrix::new(vec)
}





fn main() {
    let training_data = [[1.0, 2.0], [5.0, 6.0]];
    let labels = [[0.0, 1.0]];

    let x = arr_2d_to_matrix(&training_data);
    let y = arr_2d_to_matrix(&labels).trans();


    let x = Matrix::new(vec![vec![1.0, 2.0], vec![5.0, 6.0]]);
    let y = Matrix::new(vec![vec![ 0.0, 1.0]]);

    let z = x.stack(&y);

    println!("x: \n{}", x);
    println!("y: \n{}", y);
    println!("z: \n{}", z);
    
}