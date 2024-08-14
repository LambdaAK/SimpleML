use core::fmt;
use std::fmt::{Display, Formatter};

use crate::matrix::ColVec;

pub enum Node {
  Internal {
    feature: usize,
    value: f64,
    left: Box<Node>,
    right: Box<Node>,
  },
  Leaf {
    class: f64 // 1.0 or -1.0
  }
}

/// Calculate the Gini impurity of a node
fn tree_impurity(y: ColVec) -> f64 {
  /*
    sum of p_k(1 - p_k)

    p_k is the proportion of class k in the node

  */

  // compute the proportion of each class in the node

  // the two classes should be 1.0 and -1.0

  let n = y.rows();

  let mut pos_count = 0.;
  let mut neg_count = 0.;

  for i in 0 .. n {
    if y[i] == 1.0 {
      pos_count += 1.;
    }
    else if y[i] == -1.0 {
      neg_count += 1.;
    }
    else {
      panic!("Invalid class label {} in DecisionTree tree_impurity", y[i]);
    }
  }

  let pos_prop = pos_count / (n as f64);

  let neg_prop = neg_count / (n as f64);

  let impurity = pos_prop * (1. - pos_prop) + neg_prop * (1. - neg_prop);
  
  return impurity;

}


impl Node {
  pub fn new(x: Vec<ColVec>, y: ColVec) -> Node {


    // check the base cases


    // base case 1: all labels are the same

    let mut all_same = true;

    for i in 1 .. y.rows() {
      if y[i] != y[0] {
        all_same = false;
        break;
      }
    }

    if all_same {
      return Node::Leaf{
        class: y[0]
      };
    }

    // base case 2: all input vectors are the same

    let mut all_same = true;

    for i in 1 .. x.len() {
      if x[i] != x[0] {
        all_same = false;
        break;
      }
    }



    let mut best_feature: usize = 0;
    let mut best_threshold: f64 = 0.0;
    let mut best_impurity: f64 = 1.0;

      // Iterate over all possible features
      for f in 0..x[0].rows() {

          // Create a vector of tuples: (feature_value, input_vector, label)
          let mut feature_pairs: Vec<(f64, ColVec, f64)> = x.iter()
              .zip(y.clone().into_iter())
              .map(|(xi, yi)| (xi.get(f), xi.clone(), yi))
              .collect();

          // Sort the vector of tuples based on the feature value
          feature_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

          // Manually unzip the sorted feature_pairs into three separate vectors
          let mut feature_values = Vec::new();
          let mut sorted_x = Vec::new();
          let mut sorted_y = Vec::new();

          for (feature_value, x_vec, y_value) in feature_pairs {
              feature_values.push(feature_value);
              sorted_x.push(x_vec);
              sorted_y.push(y_value);
          }

          
          // Iterate over all possible splits of the feature

          for i in 0 .. sorted_x.len() {
            let threshold = feature_values[i];

            let left_y = sorted_y.iter().take(i).map(|&y| y).collect();
            let right_y = sorted_y.iter().skip(i).map(|&y| y).collect();

            let left_impurity = tree_impurity(left_y);
            let right_impurity = tree_impurity(right_y);

            let impurity = left_impurity + right_impurity;

            if impurity < best_impurity {
              best_impurity = impurity;
              best_feature = f;
              best_threshold = threshold;
            }
          }
          
      }

      // we now have the best feature and threshold to split on

      // split the data into left and right subsets based on this split

      let mut left_x = Vec::new();
      let mut left_y = Vec::new();
      let mut right_x = Vec::new();
      let mut right_y = Vec::new();

      for i in 0 .. x.len() {
        if x[i][best_feature] < best_threshold {
          left_x.push(x[i].clone());
          left_y.push(y[i]);
        }
        else {
          right_x.push(x[i].clone());
          right_y.push(y[i]);
        }
      }

      // recursively build the left and right subtrees

      let left = Self::new(left_x, ColVec::new(left_y));
      let right = Self::new(right_x, ColVec::new(right_y));

      return Node::Internal {
        feature: best_feature,
        value: best_threshold,
        left: Box::new(left),
        right: Box::new(right)
      };

  }
}


impl Display for Node {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    match self {
      Node::Leaf { class } => write!(f, "Leaf({})", class),
      Node::Internal { feature, value, left, right } => {
        write!(f, "Internal({}, {}, {}, {})", feature, value, left, right)
      }
    }
  }
}