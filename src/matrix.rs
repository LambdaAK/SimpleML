use std::{fmt::{Display, Formatter}, ops::{self, Add, IndexMut, Mul, Sub}};

use crate::math::Expr;

// Macro to create a matrix
macro_rules! matrix {
    ( $( [ $( $x:expr ),* ] ),* ) => {
        {
            let data: Vec<Vec<f64>> = vec![
                $(
                    vec![$($x),*],
                )*
            ];
            let rows = data.len();
            let cols = if rows > 0 { data[0].len() } else { 0 };
            Matrix {
                data,
                rows,
                cols,
            }
        }
    };
  }

#[derive(Debug)]
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

impl Matrix {

    pub fn orthonormalize(&self) -> Matrix {
        // Orthonormalize the columns of the matrix

        let transposed = self.t();

        let rows_as_col_vecs = transposed.rows_as_col_vecs();

        let orthonormalized = ColVec::orthonormalize(rows_as_col_vecs);

        let mut vec = Vec::new();

        for i in 0 .. orthonormalized.len() {
            vec.push(orthonormalized[i].data.clone());
        }

        return Matrix::new(vec).t();

    }
}

impl Matrix {
    pub fn qr(&self) -> (Matrix, Matrix) {
        let q = self.orthonormalize();

        let r = q.t() * self;
        
        (q, r)
    }
}

impl Matrix {

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut vec = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::new();
            for _ in 0 .. cols {
                let val: f64 = rand::random::<f32>() as f64;
                // negate it with 50% probability
                let val = if rand::random::<bool>() {val} else {-val};
                row.push(val);
            }
            vec.push(row);
        }

        Matrix {
            data: vec,
            rows,
            cols
        }
    }
}

impl Matrix {

    pub fn nullspace(&self) -> Vec<ColVec> {
            // Solve the equation Ax = 0
    // A is a matrix, and x is a vector. We want to find all vectors x such that Ax = 0

    let r = self.rref();

    // Go through the rows of r, and find the pivot columns
    let mut pivot_columns = Vec::new(); // List to store pivot columns

    for i in 0..r.rows() {
        for j in 0..r.cols() {
            if r.get(i, j) != 0.0 {
                pivot_columns.push(j); // Found the pivot, record the column index
                break; // Move to the next row after finding the pivot
            }
        }
    }

    // Initialize vectors that store the basis for the nullspace
    let mut nullspace_basis = Vec::new();

    for i in 0..r.cols() {
        if !pivot_columns.contains(&i) {
            // This column is not a pivot column, so it is a free variable
            // We set it to 1 and all other free variables to 0
            // Then solve for the pivot variables

            let mut free_vars = vec![0.0; r.cols()];
            free_vars[i] = 1.0;

            // Solve for the pivot variables
            for j in 0..r.rows() {
                if j < pivot_columns.len() {
                    let pivot_col = pivot_columns[j];
                    let mut sum = 0.0;
                    for k in 0..r.cols() {
                        sum += r.get(j, k) * free_vars[k];
                    }
                    if r.get(j, pivot_col) != 0.0 {
                        free_vars[pivot_col] = -sum / r.get(j, pivot_col);
                    }
                }
            }

            nullspace_basis.push(ColVec::new(free_vars));
        }
    }

    // Handle the case where the matrix is all zeros
    if nullspace_basis.is_empty() && r.rows() == r.cols() {
        for i in 0..r.cols() {
            let mut basis_vector = vec![0.0; r.cols()];
            basis_vector[i] = 1.0;
            nullspace_basis.push(ColVec::new(basis_vector));
        }
    }

    nullspace_basis


    }

    pub fn rref(&self) -> Matrix {
        let mut m = self.clone(); // Clone the matrix to avoid modifying the original
        let mut current_row = 0;  // Keep track of the current row to process
    
        // Step 1: Convert to Row Echelon Form (REF)
        for col in 0..m.cols() {
            // Find the first row that has a non-zero element in the current column
            let mut row_non_zero = current_row;
            while row_non_zero < m.rows() && m.get(row_non_zero, col) == 0.0 {
                row_non_zero += 1;
            }
    
            if row_non_zero == m.rows() {
                // If no non-zero element found in this column, continue to the next column
                continue;
            }
    
            // Swap the rows if needed to bring the pivot row to the current row
            if row_non_zero != current_row {
                m.swap_rows_mut(current_row, row_non_zero);
            }
    
            // Scale the row so that the pivot element is 1
            let scale = 1.0 / m.get(current_row, col);
            for j in 0..m.cols() {
                m.data[current_row][j] *= scale;
            }
    
            // Eliminate the current element from all rows below
            for row_below in current_row + 1..m.rows() {
                let factor = m.get(row_below, col);
                for j in 0..m.cols() {
                    m.data[row_below][j] -= factor * m.get(current_row, j);
                }
            }
    
            // Move to the next row
            current_row += 1;
        }
    
        // After row reduction, move all zero rows to the bottom
        let mut last_non_zero_row = 0;
        for i in 0..m.rows() {
            let is_zero_row = (0..m.cols()).all(|col| m.get(i, col) == 0.0);
            if !is_zero_row {
                m.swap_rows_mut(last_non_zero_row, i);
                last_non_zero_row += 1;
            }
        }
    
        // Step 2: Convert to Reduced Row Echelon Form (RREF)
        for i in (0..current_row).rev() {  // Iterate over the rows in reverse order
            // Find the pivot column in the current row
            let pivot_col = (0..m.cols()).find(|&col| m.get(i, col) != 0.0);
    
            if let Some(pivot_col) = pivot_col {
                // For each row above the current row, eliminate the pivot element
                for j in 0..i {
                    let factor = m.get(j, pivot_col);
                    for k in 0..m.cols() {
                        m.data[j][k] -= factor * m.get(i, k);
                    }
                }
            }
        }
    
        m // Return the matrix in Reduced Row Echelon Form (RREF)
    }
    

    
    
}

pub struct EigenSpace {
    pub eigenvalue: f64,
    pub basis: Vec<ColVec>
}

impl Matrix {
    pub fn eig(&self) -> Vec<EigenSpace> {

        // must be square

        if (self.rows() != self.cols()) {
            panic!("Matrix must be square to compute eigenvalues");
        }

        // use the QR algorithm to compute the eigenvalues

        let mut a = self.clone();

        for i in 0 .. 100 {

            // get the bottom right element of a

            let bottom_right = a.get(a.rows() - 1, a.cols() - 1);
            println!("{}", i);
            println!("{}", a);
            let (q, r) = (&a - bottom_right * Matrix::eye(a.rows())).qr();
            a = r * q;
        }

        // the eigenvalues are the diagonal elements of a

        let mut eigenvalues = Vec::new();

        // there could be duplicate eigenvalues. Get rid of duplicates

        for i in 0 .. a.rows() {
            let lambda = a.get(i, i);
            
            // add lambda to eigenvalues only if it is not already in the list
            if !eigenvalues.contains(&lambda) {
                eigenvalues.push(lambda)
            }
        }

        // compute the eigenvectors

        let mut eigenspaces: Vec<EigenSpace> = Vec::new();

        for i in 0 .. eigenvalues.len() {
            let lambda = eigenvalues[i];
            let a_minus_lamnda_i = &a - &Matrix::eye(a.rows()) * lambda;
            let nullspace = a_minus_lamnda_i.nullspace();

            let mut eigenbasis = Vec::new();

            nullspace.iter().for_each(|v| {
                eigenbasis.push(v.clone())
            });

            eigenspaces.push(EigenSpace{
                eigenvalue: lambda,
                basis: eigenbasis
            });

        }
        
        return eigenspaces;

    }


    
}


/*
Matrix operations

Matrix operations take references to matrices as arguments and return a new matrix
*/

// Binary operations for matrices

// addition
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

impl ops::Add<&Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        &self + rhs
    }
}

impl ops::Add<Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Matrix) -> Matrix {
        self + &rhs
    }
}

// subtraction
impl ops::Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
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
        self - &rhs
    }
}

impl ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Matrix {
        &self - &rhs
    }
}

impl ops::Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        &self * rhs
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

impl ops::Mul<Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        self * &rhs
    }
}

impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        &self * &rhs
    }
}



// scalar multiplication
impl ops::Mul<f64> for &Matrix {
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

impl ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix {
        &self * rhs
    }
}

impl ops::Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        rhs * self
    }
}

impl ops::Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        &rhs * self
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

    pub fn lu_decomposition(&self) -> (Matrix, Matrix) {
        // LU decomposition
        // A = LU
        // L is a lower triangular matrix
        // U is an upper triangular matrix

        // make sure the matrix is square
        if self.rows != self.cols {
            panic!("Matrix is not square");
        }

        let n = self.rows;

        let mut l = Matrix::eye(n);
        let mut u = self.clone();

        for i in 0..n {
            for j in i+1..n {

                let factor = u.data[j][i] / u.data[i][i];

                
                l.data[j][i] = factor;
                for k in i..n {
                    u.data[j][k] -= factor * u.data[i][k];
                }
            }
        }

        return (l, u);
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

    pub fn inverse_old(&self) -> Matrix {
        let det = self.det();
        if det == 0.0 {
            panic!("Matrix is singular and cannot be inverted");
        }
        let adjugate = self.adjugate();

        let inverse = adjugate * (1.0 / det);
        return inverse;
    }

    pub fn inverse(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            panic!("Matrix is not square, failure to compute determinant");
        }
    
        let mut acc = Self::eye(self.rows());
        let mut m = self.clone();
        
        for row_num in 0..m.rows() {
            // Find the first row that has a non-zero element in the row_num-th position
            let mut row_num_non_zero = row_num;
            let mut found = false;
    
            while row_num_non_zero < m.rows() {
                if m.get(row_num_non_zero, row_num) != 0.0 {
                    found = true;
                    break;
                }
                row_num_non_zero += 1;
            }
    
            if !found {
                // This matrix is not invertible, so the determinant is 0
                return Option::None;
            }
    
            // Swap the rows and update acc, if needed
            if row_num_non_zero != row_num {
                m.swap_rows_mut(row_num, row_num_non_zero);
                acc.swap_rows_mut(row_num, row_num_non_zero);
            }
    
            // Scale the row so that the row_num-th element is 1
            let scale = 1.0 / m.get(row_num, row_num);
    
            for j in 0..m.cols() {
                m.data[row_num][j] *= scale;
                acc.data[row_num][j] *= scale;
            }
    
            // Eliminate the row_num-th element from all rows below
            for row_below in row_num + 1..m.rows() {
                let factor = m.get(row_below, row_num);
                for j in 0..m.cols() {
                    m.data[row_below][j] -= factor * m.get(row_num, j);
                    acc.data[row_below][j] -= factor * acc.get(row_num, j);
                }
            }
        }

        // now, m is in row-echelon form.
        // We need to put it into reduced row-echelon form

        for row_num in (0 .. m.rows()).rev() {
            // we know that the row_num-th element of the row_num-th row is 1
            // we want to subtract some multiple of the row_num-th row from all rows to make the row_num-th element of all rows above the row_num-th row 0
            for row_above in (0 .. row_num).rev() {
                let factor = m.get(row_above, row_num);
                for j in 0 .. m.cols() {
                    m.data[row_above][j] -= factor * m.get(row_num, j);
                    acc.data[row_above][j] -= factor * acc.get(row_num, j);
                }
            }
        }

        // m is now the identity matrix, so acc is the inverse of the original matrix

        return Option::Some(acc);
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

    pub fn rows_as_col_vecs(&self) -> Vec<ColVec> {
        let mut col_vecs : Vec<ColVec> = Vec::new();

        for i in 0 .. self.rows() {
            let mut data = Vec::new();
            for j in 0 .. self.cols() {
                data.push(self.data[i][j]);
            }
            col_vecs.push(ColVec::new(data));
        }

        return col_vecs;
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

impl Matrix {
    pub fn swap_rows(&self, i: usize, j: usize) -> Matrix {
        let mut m = self.clone();
        m.swap_rows_mut(i, j);
        return m;
    }

    pub fn swap_rows_mut(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }

}

impl Matrix {
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }
}

impl Matrix {
    pub fn det(&self) -> f64 {
        if self.rows != self.cols {
            panic!("Matrix is not square, failure to compute determinant");
        }
    
        let mut acc: f64 = 1.0;
        let mut m = self.clone();
        
        for row_num in 0..m.rows() {
            // Find the first row that has a non-zero element in the row_num-th position
            let mut row_num_non_zero = row_num;
            let mut found = false;
    
            while row_num_non_zero < m.rows() {
                if m.get(row_num_non_zero, row_num) != 0.0 {
                    found = true;
                    break;
                }
                row_num_non_zero += 1;
            }
    
            if !found {
                // This matrix is not invertible, so the determinant is 0
                return 0.0;
            }
    
            // Swap the rows and update acc, if needed
            if row_num_non_zero != row_num {
                m = m.swap_rows(row_num, row_num_non_zero);
                acc *= -1.0;
            }
    
            // Scale the row so that the row_num-th element is 1
            let scale = 1.0 / m.get(row_num, row_num);
    
            for j in 0..m.cols() {
                m.data[row_num][j] *= scale;
            }
    
            // Update acc
            acc /= scale;
    
            // Eliminate the row_num-th element from all rows below
            for row_below in row_num + 1..m.rows() {
                let factor = m.get(row_below, row_num);
                for j in 0..m.cols() {
                    m.data[row_below][j] -= factor * m.get(row_num, j);
                }
            }
        }
    
        // Multiply the diagonal elements into acc
        for i in 0..m.rows() {
            acc *= m.get(i, i);
        }
    
        // acc is now the determinant of the matrix
        return acc;
    }
    
}

impl Matrix {
    pub fn zero_matrix(rows: usize, cols: usize) -> Matrix {
        let mut vec = Vec::new();
        for _ in 0 .. rows {
            let mut row = Vec::new();
            for _ in 0 .. cols {
                row.push(0.0);
            }
            vec.push(row);
        }
        Matrix {
            data: vec,
            rows,
            cols
        }
    }
}

impl Matrix {

    pub fn center_data(&self) -> Self {
        /*
            Each row is a data point
            find the mean point and subtract it from each data point 
        */

        let rows = self.rows_as_col_vecs();

        let n = rows.len();

        let mut sum = ColVec::new(vec![0.0; rows[0].rows()]);

        for i in 0 .. n {
            sum = sum + &rows[i];
        }

        let mean = sum * (1.0 / n as f64);

        let mut centered_rows = Vec::new();

        for i in 0 .. n {
            centered_rows.push(&rows[i] - &mean);
        }

        let mut centered_data = Matrix::zero_matrix(n, rows[0].rows());

        for i in 0 .. n {
            for j in 0 .. rows[0].rows() {
                centered_data.data[i][j] = centered_rows[i].get(j);
            }
        }

        return centered_data;


    }

    pub fn cov_matrix(&self) -> Self {
        /*
            The rows of this matrix are vectors
            Each row is a point in an m dimensional space
            There are n points
        */
        
        let vectors: Vec<Matrix> = self.rows_as_matrices();

        let n = vectors.len(); // the number of points
        let m = vectors[0].cols(); // the number of dimensions per vector

        let mut cov_matrix = Matrix::zero_matrix(m, m);

        for i in 0 .. n {
            let v = &vectors[i];
            let v_t = v.t();

            // v is a column vector and v_t is a row vector
            // v_t * v is an m x m matrix

            cov_matrix = cov_matrix + v_t * v;
        }

        cov_matrix = cov_matrix * (1.0 / n as f64);

        cov_matrix

    }
}

impl Matrix {
    pub fn char_poly(&self) -> Expr {
        todo!("unimplemented char_poly");
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        // check if the dimensions are the same
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        let epsilon = 1e-6;

        // each pair of entries must be within epsilon of each other

        for i in 0..self.rows {
            for j in 0..self.cols {
                if (self.data[i][j] - other.data[i][j]).abs() > epsilon {
                    return false;
                }
            }
        }

        return true;

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

impl Clone for ColVec {
    fn clone(&self) -> Self {
        ColVec {
            data: self.data.clone(),
            rows: self.rows
        }
    }
}

impl ColVec {
    pub fn new(data: Vec<f64>) -> ColVec {
        ColVec {
            data: data.clone(),
            rows: data.len()
        }
    }

    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    pub fn rows(&self) -> usize {
        self.rows
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

    pub fn dot(&self, other: &ColVec) -> f64 {
        if self.rows != other.rows {
            panic!("Vector dimensions must match in ColVec::dot");
        }

        let mut sum = 0.0;

        for i in 0 .. self.rows {
            sum += self.data[i] * other.data[i];
        }

        sum
    }

    pub fn proj_onto(&self, other: &ColVec) -> ColVec {
        let numerator = self.dot(other);
        let denominator = other.dot(other);

        let factor = numerator / denominator;

        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(factor * other.data[i]);
        }

        ColVec::new(data)
    }

    pub fn orthogonalize(vectors: Vec<ColVec>) -> Vec<ColVec> {
        // use Gram-Schmidt orthogonalization algorithm

        // all vectors must have the same length

        if vectors.len() == 0 {
            return Vec::new();
        }

        let n = vectors[0].rows();

        for i in 1 .. vectors.len() {
            let vector = &vectors[i];
            if vector.rows() != n {
                panic!("All vectors must have the same dimensionality in ColVec::orthogonalize");
            }
        }

        // the number of vectors must be at most n

        if vectors.len() > n {
            panic!("The number of vectors must be at most n in ColVec::orthogonalize");
        }

        let mut orthogonalized: Vec<ColVec> = Vec::new();

        for i in 0 .. vectors.len() {
            let mut vector = vectors[i].clone();

            for j in 0 .. i {
                let proj = vector.proj_onto(&orthogonalized[j]);
                vector = vector - proj;
            }

            orthogonalized.push(vector);
        }

        return orthogonalized;

    }

    pub fn norm(&self) -> f64 {
        let mut sum = 0.0;

        for i in 0 .. self.rows {
            sum += self.data[i] * self.data[i];
        }

        sum.sqrt()
    }

    pub fn normalize(&self) -> ColVec {
        let norm = self.norm();

        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(self.data[i] / norm);
        }

        ColVec::new(data)
    }

    pub fn orthonormalize(vectors: Vec<ColVec>) -> Vec<ColVec> {
        let orthogonalized = ColVec::orthogonalize(vectors);

        let mut orthonormalized: Vec<ColVec> = Vec::new();

        for i in 0 .. orthogonalized.len() {
            let vector = orthogonalized[i].normalize();
            orthonormalized.push(vector);
        }

        return orthonormalized;
    }


}

impl Sub for ColVec {
    type Output = ColVec;

    fn sub(self, other: ColVec) -> ColVec {
        if self.rows != other.rows {
            panic!("Vector dimensions must match in ColVec::sub");
        }

        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(self.data[i] - other.data[i]);
        }

        ColVec::new(data)
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

impl Add for ColVec {
    type Output = ColVec;

    fn add(self, other: ColVec) -> ColVec {
        if self.rows != other.rows {
            panic!("Vector dimensions must match in ColVec::add");
        }

        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(self.data[i] + other.data[i]);
        }

        ColVec::new(data)
    }
}

impl Add<&ColVec> for ColVec {
    type Output = ColVec;

    fn add(self, other: &ColVec) -> ColVec {
        if self.rows != other.rows {
            panic!("Vector dimensions must match in ColVec::add");
        }

        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(self.data[i] + other.data[i]);
        }

        ColVec::new(data)
    }
}

impl Add<ColVec> for &ColVec {
    type Output = ColVec;

    fn add(self, other: ColVec) -> ColVec {
        self.clone() + other
    }
}

impl Mul<f64> for ColVec {
    type Output = ColVec;

    fn mul(self, scalar: f64) -> ColVec {
        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(self.data[i] * scalar);
        }

        ColVec::new(data)
    }
}

impl Mul<f64> for &ColVec {
    type Output = ColVec;

    fn mul(self, scalar: f64) -> ColVec {
        self.clone() * scalar
    }
}

impl Mul<ColVec> for f64 {
    type Output = ColVec;

    fn mul(self, other: ColVec) -> ColVec {
        other * self
    }
}

impl Mul<&ColVec> for f64 {
    type Output = ColVec;

    fn mul(self, other: &ColVec) -> ColVec {
        other.clone() * self
    }
}



impl Sub<&ColVec> for ColVec {
    type Output = ColVec;

    fn sub(self, other: &ColVec) -> ColVec {
        if self.rows != other.rows {
            panic!("Vector dimensions must match in ColVec::sub");
        }

        let mut data = Vec::new();

        for i in 0 .. self.rows {
            data.push(self.data[i] - other.data[i]);
        }

        ColVec::new(data)
    }
}

impl Sub<ColVec> for &ColVec {
    type Output = ColVec;

    fn sub(self, other: ColVec) -> ColVec {
        self.clone() - other
    }
}

impl Sub<&ColVec> for &ColVec {
    type Output = ColVec;

    fn sub(self, other: &ColVec) -> ColVec {
        self.clone() - other.clone()
    }
}



#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_matrix_addition() {
  
    
    let m1 = matrix![[1.0, 2.0], [3.0, 4.0]];
    let m2 = matrix![[5.0, 6.0], [7.0, 8.0]];
    let m3 = matrix![[6.0, 8.0], [10.0, 12.0]];

    assert_eq!(m1 + m2, m3);

    let m1 = matrix![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let m2 = matrix![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    let m3 = matrix![[6.0, 8.0], [10.0, 12.0], [14.0, 16.0]];

    assert_eq!(m1 + m2, m3);

    let m1 = matrix![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let m2 = matrix![[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
    let m3 = matrix![[6.0, 8.0], [10.0, 12.0], [14.0, 16.0]];

    assert_eq!(&m1 + &m2, m3);

    assert_eq!(&m1 + &m1, 2.0 * &m1);

    let m1 = matrix![[1.0, 2.0], [3.0, 4.0]];
    let m2 = matrix![[5.0, 6.0], [7.0, 8.0]];
    let m3 = /* diff */ matrix![[-4.0, -4.0], [-4.0, -4.0]];

    assert_eq!(m1 - m2, m3);

    
  }

    #[test]
  fn id() {
    for _ in 0 .. 10000 {
        let dim = rand::random::<usize>() % 10 + 1;
        let m = Matrix::random(dim, dim);
        let id = Matrix::eye(dim);
        assert_eq!(&m * &id, m);
        assert_eq!(id * &m, m);
    }

    for _ in 0 .. 100 {
        let dim = rand::random::<usize>() % 100 + 1;
        let m = Matrix::random(dim, dim);
        assert_eq!(&m * 1.0, m);
        assert_eq!(1.0 * &m, m);
    }

    for _ in 0 .. 100 {
        let dim = rand::random::<usize>() % 100 + 1;
        let m = Matrix::random(dim, dim);
        assert_eq!(&m * 0.0, Matrix::new(vec![vec![0.0; dim]; dim]));
    }

    for _ in 0 .. 100 {
        let dim = rand::random::<usize>() % 10 + 1;

        let a = Matrix::random(dim, dim);
        let b = Matrix::random(dim, dim);
        let c = Matrix::random(dim, dim);

        assert_eq!(&a * (&b * &c), (&a * &b) * &c);
        assert_eq!(&a * (&b + &c), &a * &b + &a * &c);
        assert_eq!((&a + &b) * &c, &a * &c + &b * c);

    }

  }

}

