use std::cmp::Ordering;

use crate::matrix::Matrix;

pub struct KNN {
  k: i32,
  points: Vec<Matrix>,
  y: Vec<f64>
}

impl KNN {
  pub fn new(k: i32, x: Matrix, y: Matrix) -> KNN {

    // each row of x is a point

    let rows = x.rows_as_matrices();

    // convert y into a Vec<f64>

    let y = y.into_vec();

    KNN {
      k,
      points: rows,
      y
    }
   
  }
  fn sort_labels_by_floats(labels: Vec<f64>, distances: Vec<f64>) -> Vec<f64> {
    let mut zipped: Vec<(f64, f64)> = labels.iter().zip(distances.iter()).map(|(a, b)| (*a, *b)).collect();
    zipped.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    zipped.iter().map(|(a, _)| *a).collect()
  }

  pub fn predict(&self, x: Matrix) -> f64 {
    let mut distances:Vec<f64> = vec![];

    // compute the distance between x and each point

    for point in &self.points {
      let distance = point.euclidean_distance(&x);
      distances.push(distance.unwrap()); // TODO: get rid of the unwrap here
    }

    // sort the points by distance from x

    let sorted_labels = KNN::sort_labels_by_floats(self.y.clone(), distances);

    // take the mode of the first k labels

    let mut counts: Vec<(f64, i32)> = vec![]; // (label, count)

    let first_k = sorted_labels.iter().take(self.k as usize);

    for label in first_k {
      let mut found = false;
      for count in &mut counts {
        if count.0 == *label {
          count.1 += 1;
          found = true;
          break;
        }
      }
      if !found {
        counts.push((*label, 1));
      }
    }

    counts.sort_by(|a, b| a.1.cmp(&b.1));

    counts.last().unwrap().0
    
  }

}

