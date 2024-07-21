mod matrix;
use crate::matrix::Matrix;
mod perceptron;



fn main() {
    let training_data = [[1.0, 2.0], [5.0, 6.0]];
    let labels = [[0.0, 1.0]];

    let x = Matrix::from_2d(&training_data);
    let y = Matrix::from_2d(&labels);

    let z = x.stack(&y);

    println!("x: \n{}", x);
    println!("y: \n{}", y);
    println!("z: \n{}", z);
    
}