use std::{fmt::{Display, Formatter}, ops::{self, IndexMut}};

pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize
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

impl ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
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

impl ops::Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Matrix {
        &self + &rhs
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

impl ops::Sub<Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Matrix {
        self.clone() - rhs
    }
}

impl ops::Sub<&Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        &self - rhs
    }
}

impl ops::Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        self.clone() - rhs.clone()
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

impl ops::Mul<&f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &f64) -> Matrix {
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

impl ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        &self * &rhs
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

        println!("computing determinant of matrix: \n{}", self);

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

        println!("computing adjugate of matrix: \n{}", self);

        let mut adjugate_matrix = Vec::new();
        for i in 0..self.rows {
            println!("i: {}", i);
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

    pub fn into_vec(&self) -> Vec<f64> {
        // must have either 1 row or 1 column (or both)
        if self.rows != 1 && self.cols != 1 {
            panic!("Matrix must be a vector in Matrix::into_vec()")
        };

        let mut vec = Vec::new();

        if self.rows == 1 {
            for i in 0 .. self.cols {
                vec.push(self.data[0][i])
            }
        }

        else {
            for i in 0 .. self.rows {
                vec.push(self.data[i][0])
            }
        }
        
        return vec;
    }

    pub fn mul_pointwise(&self, x: &Matrix) -> Matrix {

        // dimensions must be exactly the same
        if self.rows != x.rows || self.cols != x.cols {
            panic!("Matrix dimensions do not match in mul_pointwise");
        }

        let mut vec = Vec::new();

        for i in 0..self.rows {
            let mut row = Vec::new();
            for j in 0..self.cols {
                row.push(self.data[i][j] * x.data[i][j]);
            }
            vec.push(row);
        }

        return Matrix {
            data: vec,
            rows: self.rows,
            cols: self.cols
        }
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

impl Matrix {
    pub fn euclidean_norm(&self) -> Result<f64, String> {
        // must be a vector
        if self.cols != 1 && self.rows != 1 {
            return Err("Matrix must be a vector".to_string());
        }

        // if it's a row vector, transpose it
        let v = if self.rows == 1 {self.t()} else {self.clone()};

        // take the inner product with itself
        let inner_product = v.t() * v;

        // take the square root of the inner product

        return Ok(inner_product.data[0][0].sqrt());
    }

    pub fn euclidean_distance(&self, x: &Matrix) -> Result<f64, String> {
        let diff = self - x;
        diff.euclidean_norm()
    }
}

impl Matrix {
    pub fn eye(n: usize) -> Matrix {
        let mut vec = Vec::new();
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..n {
                if i == j {
                    row.push(1.0);
                }
                else {
                    row.push(0.0);
                }
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows: n,
            cols: n
        }
    }
}

pub struct RowVec {
    data: Vec<f64>,
    cols: usize
}

pub struct ColVec {
    data: Vec<f64>,
    rows: usize
}

impl RowVec {
    pub fn new (data: &Vec<f64>) -> RowVec {
        RowVec {
            data: data.clone(),
            cols: data.len()
        }
    }

    pub fn t(&self) -> ColVec {
        ColVec {
            data: self.data.clone(),
            rows: self.cols
        }
    }

    pub fn as_matrix(&self) -> Matrix {
        Matrix::row_vec(self.data.as_slice())
    }

}

impl ColVec {
    pub fn new(data: Vec<f64>) -> ColVec {
        ColVec {
            data: data.clone(),
            rows: data.len()
        }
    }

    pub fn t(&self) -> RowVec {
        RowVec {
            data: self.data.clone(),
            cols: self.rows
        }
    }

    pub fn as_matrix(&self) -> Matrix {
        Matrix::col_vec(self.data.as_slice())
    }
}

impl Display for RowVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.cols {
            s.push_str(&self.data[i].to_string());
            s.push_str(" ");
        }
        write!(f, "{}", s)
    }
}

impl Display for ColVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.rows {
            s.push_str(&self.data[i].to_string());
            s.push_str("\n");
        }
        write!(f, "{}", s)
    }
}

// indexing for vectors

impl ops::Index<usize> for RowVec {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl ops::Index<usize> for ColVec {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for RowVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl IndexMut<usize> for ColVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
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