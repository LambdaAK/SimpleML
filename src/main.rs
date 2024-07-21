mod matrix;
use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;



fn main() {
    let training_data = [[1.0, 2.0], [5.0, 6.0]];
    let labels = [[0.0, 1.0]];

    let x = Matrix::from_2d(&training_data);
    let y = Matrix::from_2d(&labels);

    let model: Perceptron = Perceptron::new(x, y);

    let test_points = Matrix::from_2d(&[[1.0, 2.0], [5.0, 6.0], [1.0, 2.0], [2.0, -5.0]]);
    
    let predictions = model.predict(test_points);

    println!("predictions\n{}", predictions)


}