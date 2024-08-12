use crate::matrix::{ColVec, Kernel, Matrix};


pub struct KernelPerceptron {
  alpha: ColVec,
  x: Vec<ColVec>,
  y: ColVec,
  kernel: Kernel
}

impl KernelPerceptron {
  pub fn new(x: Matrix, y: ColVec, kernel: Kernel) -> KernelPerceptron {
    let x: Vec<ColVec> = x.rows_as_col_vecs();


    // alpha is a vector of length equal to the number of training points

    let mut alpha = ColVec::new(vec![0.0; x.len()]);
    


    // represent the weight vector as y * alpha * x

    // train the alpha vector

    // h(x) = sign(sum(alpha_i * y_i * x_i^T * x))

    loop {

      // for each training point, if the prediction is wrong, update alpha

      let mut errors = 0.;

      // classify each point

      for i in 0 .. x.len() {

        // classify x_i

        let mut sum = 0.;

        for j in 0 .. x.len() {
          // compute the term in the sum

          let inner_prod = kernel.eval(&x[i], &x[j]);

          sum += alpha.get(j) * y.get(j) * inner_prod;

        }

        let prediction = if sum >= 0.0 {
          1.0
        } else {
          -1.0
        };

        if prediction != y.get(i) {
          // update alpha

          alpha.set(i, y.get(i) * alpha.get(i) + 0.0001);
          errors += 1.0;
        }

      }
      
      if errors == 0.0 {
        break;
      }
      else {
        println!("errors: {}", errors);
      }

      println!("alpha:\n{}", alpha);


    }
  
    KernelPerceptron {
      alpha,
      x,
      y,
      kernel
    }

  }

  pub fn predict(&self, x: ColVec) -> f64 {
    let mut sum = 0.0;

    for i in 0 .. self.x.len() {

      let inner_prod = self.kernel.eval(&x, &self.x[i]);

      sum += self.alpha.get(i) * self.y.get(i) * inner_prod;
      
    }
    
    if sum >= 0.0 {
      1.0
    } else {
      -1.0
    }

  }
}

