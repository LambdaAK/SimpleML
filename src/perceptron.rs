use crate::matrix::Matrix;


pub struct Perceptron {
  pub weights: Matrix
}

impl Perceptron {
  pub fn new(x: Matrix, y: Matrix) -> Perceptron {
    // each row of x is a training point
    // each row of y is the corresponding label
    let num_rows: usize = x.rows();
    let num_cols: usize = x.cols();

    // make a weight vector (column vector)
    let mut weights: Vec<Vec<f64>> = Vec::new();
    for _ in 0..num_cols {
      let mut row: Vec<f64> = Vec::new();
      row.push(1.0);
      weights.push(row);
    }
    let weights_without_bias = Matrix::new(weights);

    // stack a bias (0) to the bottom of the weight vector
    
    let bias_matrix: Matrix = Matrix::from_2d(&[[0.0]]);

    let weights = weights_without_bias.stack(&bias_matrix);

    let mut model = Perceptron { weights };

    model.train(x, y, 10);

    model

  }

  fn train(&mut self, x: Matrix, y: Matrix, max_epochs: usize) {
    // train the model

    let points = x.rows_as_matrices();
    let labels = y.rows_as_matrices();

    println!("labels: {}", labels.len());
      
    for _ in 0 .. max_epochs {
      // for each training point, classify the point and update if needed
      
      let predictions = self.predict(&x);
      let pred_rows = predictions.rows_as_matrices();

      let errors = predictions.mul_pointwise(&y);

      println!("Errors\n{}", errors);
      
      // errors should be a matrix where each row is 1 element long
      // if the element is negative or 0, the prediction was wrong, so we need to update the weights

      for i in 0..pred_rows.len() {
        let error = errors.data[i][0];
        println!("error: {}", error);
        if error <= 0.0 {
          println!("updating weights");
          // update the weights
          let point = &points[i];

          // add a 1 to the end of the point

          let ones: Vec<Vec<f64>> = vec![vec![1.0; 1]; 1];

          let ones_matrix = Matrix::new(ones);

          let point = (point.t().stack(&ones_matrix)).t();

          println!("point: {}", point);

          let label_of_point  = &labels[i].data[0][0];

          println!("label: {}", label_of_point);

          let update = (point * *label_of_point).t();

          println!("update: {}", update);
          println!("weightss: {}", self.weights);

          self.weights = self.weights.clone() + update;
        }
      }

    }



  }

  pub fn predict(&self, x: &Matrix) -> Matrix {
    // predict the labels of the input data
    // x is a matrix where each row is a training point
    // the output is a matrix where each row is the predicted label of the corresponding training point

    /*
      | x1 |
      | x2 |
      | .. |
      | xn |

      | w1 |
      | w2 |
      | .. |
      | wn |
      | b  |
     */

    // we need to add a column of 1s to the input data
    
    let x_t = x.t();
    
    // make a matrix of 1s that has the same number of rows as x_t

    let ones: Vec<Vec<f64>> = vec![vec![1.0; x_t.cols()]; 1];

    let ones_matrix = Matrix::new(ones);

    // stack the ones matrix on the bottom of x_t

    let x_t_with_bias = x_t.stack(&ones_matrix);

    // tranpose it again

    let new_x = x_t_with_bias.t();


    let preds = &new_x * &self.weights;

    // make each element the sign of the element

    let mut vec = Vec::new();
    for i in 0..preds.rows() {
      let mut row = Vec::new();
      for j in 0..preds.cols() {
        row.push(preds.data[i][j].signum());
      }
      vec.push(row);
    }

    Matrix::new(vec)

  }
}