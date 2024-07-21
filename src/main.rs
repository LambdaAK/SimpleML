mod matrix;
use crate::matrix::Matrix;
mod perceptron;
use crate::perceptron::Perceptron;



fn main() {
    let training_data = [
      [1., 1.],
      [1., 2.],
      [5., 5.],
      [10., 10.]
    ];

    let labels = [
      [1.],
      [1.],
      [-1.],
      [-1.]
    ];

    let x = Matrix::from_2d(&training_data);
    let y = Matrix::from_2d(&labels);

    let model: Perceptron = Perceptron::new(x, y);



    // test the model

    let test_data = [[1.0, 2.0], [7.0, 9.0], [-1., -1.], [100., 200.]];
    let test_data_vec = Matrix::from_2d(&test_data);

    let predictions = model.predict(&test_data_vec);

    println!("Predictions\n{}", predictions);
    


}