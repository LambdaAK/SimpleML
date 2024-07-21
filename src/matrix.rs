
use std::ops;


pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                s.push_str(&self.data[i][j].to_string());
                s.push_str(" ");
            }
            s.push_str("\n");
        }
        write!(f, "{}", s)
    }
}

/*
Matrix operations

Matrix operations take references to matrices as arguments and return a new matrix
*/

impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Matrix {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrix dimensions do not match");
        }
        let mut vec = Vec::new();
        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..self.cols {
                row.push(self.data[i][j] + rhs.data[i][j]);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: self.rows,
            cols: self.cols
        }
    }
}


impl ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Matrix {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrix dimensions do not match");
        }
        let mut vec = Vec::new();
        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..self.cols {
                row.push(self.data[i][j] - rhs.data[i][j]);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: self.rows,
            cols: self.cols
        }
    }
}

impl ops::Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {

        if self.cols != rhs.rows {
            panic!("Matrix dimensions do not match");
        }
        let mut vec = Vec::new();
        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * rhs.data[k][j];
                }
                row.push(sum);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: self.rows,
            cols: rhs.cols
        }
    }
}

impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        &self * &rhs
    }
}

impl ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        let mut vec = Vec::new();
        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..self.cols {
                row.push(self.data[i][j] * rhs);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: self.rows,
            cols: self.cols
        }
    }
}

impl ops::Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        let mut vec = Vec::new();
        for i in 0..rhs.rows {
            let mut row = Vec::new();
            for j in 0..rhs.cols {
                row.push(rhs.data[i][j] * self);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: rhs.rows,
            cols: rhs.cols
        }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), rows: self.rows.clone(), cols: self.cols.clone() }
    }
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        // make sure that all rows are the same length
        for i in 1..rows {
            if data[i].len() != cols {
                panic!("All rows must have the same length");
            }
        }
        Matrix {
            data,
            rows,
            cols
        }   
    }
    
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn t(&self) -> Matrix {
        let mut vec = Vec::new();
        for i in 0..self.cols {
            let mut row = Vec::new();
            for j in 0..self.rows {
                row.push(self.data[j][i]);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: self.cols,
            cols: self.rows
        }
    }

    pub fn det(&self) -> f64 {
        if (self.rows != self.cols) {
            panic!("Matrix is not square");
        }
        if self.rows == 1 {
            return self.data[0][0]
        }

        let mut det = 0.0;
        for i in 0..self.cols {
            let cofactor = self.data[0][i] * self.cofactor(0, i);
            det += cofactor;
        }
        return det;
        
    }

    pub fn minor(&self, row: usize, col: usize) -> Matrix {
        let mut sub_matrix: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.rows {
            if i == row {
                continue;
            }
            let mut sub_row = Vec::new();
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                sub_row.push(self.data[i][j]);
            }
            sub_matrix.push(sub_row);
        }
        return Matrix {
            data: sub_matrix,
            rows: self.rows - 1,
            cols: self.cols - 1
        }
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f64 {
        // compute the minor matrix
        let minor_matrix = self.minor(row, col);

        // compute the determinant of the minor matrix
        let minor_det = minor_matrix.det();

        // compute the cofactor
        let cofactor = if (row + col) % 2 == 0 {minor_det} else {- minor_det};

        return cofactor;
    }

    pub fn adjugate(&self) -> Matrix {

        let mut adjugate_matrix = Vec::new();
        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..self.cols {
                let cofactor = self.cofactor(i, j);
                row.push(cofactor)
            }
            adjugate_matrix.push(row);
        }

        let adj = Matrix {
            data: adjugate_matrix,
            rows: self.rows,
            cols: self.cols
        };

        return adj.t();
    }

    pub fn inverse(&self) -> Matrix {
        let det = self.det();
        if det == 0.0 {
            panic!("Matrix is singular and cannot be inverted");
        }
        let adjugate = self.adjugate();
        let inverse = adjugate * (1.0 / det);
        return inverse;
    }


    /**
     * Vertically stack two matrices
     * They must have the same number of columns
     * 
     */
    pub fn stack(&self, other: &Matrix) -> Matrix {

        if self.cols() != other.cols() {
            panic!("Matrices must have the same number of columns to be stacked");
        }

        let mut vec = Vec::new();

        for i in 0..self.rows() {
            let mut row = Vec::new();
            for j in 0..self.cols() {
                row.push(self.data[i][j]);
            }
            vec.push(row);
        }

        for i in 0..other.rows() {
            let mut row = Vec::new();
            for j in 0..other.cols() {
                row.push(other.data[i][j]);
            }
            vec.push(row);
        }

        return Matrix {
            data: vec,
            rows: self.rows() + other.rows(),
            cols: self.cols()
        }
    }

    pub fn rows_as_matrices(&self) -> Vec<Matrix> {
        let mut matrices : Vec<Matrix> = Vec::new();

        for i in 0..self.rows() {
            let row = self.data[i].clone();
            let m = Matrix::row_vec(row.as_slice());
            matrices.push(m);
        }
        
        return matrices;

    }


}

impl Matrix {
    fn arr_to_vec(arr: &[f64]) -> Vec<f64> {
        let mut vec = Vec::new();
        for i in arr.iter() {
          vec.push(*i)
        }
        return vec
      }
      
      pub fn from_2d<const ROWS: usize, const COLS: usize>(arr: &[[f64; COLS]; ROWS]) -> Matrix {
        let mut vec = Vec::new();
        for row in arr.iter() {
            vec.push(Self::arr_to_vec(row));
        }
        Matrix::new(vec)
      }

    pub fn col_vec(arr: &[f64]) -> Matrix {
        let mut vec = Vec::new();
        for i in arr.iter() {
            vec.push(vec![*i])
        }
        Matrix::new(vec)
    }

    pub fn row_vec(arr: &[f64]) -> Matrix {
        Self::col_vec(arr).t()
    }
}



#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_matrix_addition() {
    assert_eq!(1, 1)
  }

}