use std::{collections::HashSet, fmt::{Display, Formatter}, ops::{self, Add, Sub}};

use crate::{matrix, token::ConstantToken};

/*
  Fun enum with implementations
*/

pub enum UnaryFun {
  Ln,
  ReLU,
  ReLUPrime
}

pub enum BinaryFun {
  Max,
  MaxPrime
}


impl UnaryFun {
  pub fn from_str(s: &str) -> Option<UnaryFun> {
    match s {
      "ln" => Some(UnaryFun::Ln),
      "relu" => Some(UnaryFun::ReLU),
      _ => None
    }
  }
}

impl Clone for UnaryFun {
  fn clone(&self) -> Self {
    match self {
      UnaryFun::Ln => UnaryFun::Ln,
      UnaryFun::ReLU => UnaryFun::ReLU,
      UnaryFun::ReLUPrime => UnaryFun::ReLUPrime,
    }
  }
}

impl Clone for BinaryFun {
  fn clone(&self) -> Self {
    match self {
      BinaryFun::Max => BinaryFun::Max,
      BinaryFun::MaxPrime => BinaryFun::MaxPrime
    }
  }
}

impl Display for UnaryFun {
  fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
    match self {
      UnaryFun::Ln => write!(f, "ln"),
      UnaryFun::ReLU => write!(f, "ReLU"),
      UnaryFun::ReLUPrime => write!(f, "ReLU'"),
    }
  }
}

impl Display for BinaryFun {
  fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
    match self {
      BinaryFun::Max => write!(f, "max"),
      BinaryFun::MaxPrime => write!(f, "max'")
    }
  }
}

/*
  Expr enum with implementations
*/

pub enum Expr {
  Num(f64),
  Const(ConstantToken),
  Add(Box<Expr>, Box<Expr>),
  Sub(Box<Expr>, Box<Expr>),
  Mul(Box<Expr>, Box<Expr>),
  Div(Box<Expr>, Box<Expr>),
  Pow(Box<Expr>, Box<Expr>),
  UnaryApp(UnaryFun, Box<Expr>),
  BinaryApp(BinaryFun, Box<Expr>, Box<Expr>),
  Var(String)
}

impl Clone for Expr {
    fn clone(&self) -> Self {
        match self {
            Self::Num(arg0) => Self::Num(arg0.clone()),
            Self::Const(arg0) => Self::Const(arg0.clone()),
            Self::Add(arg0, arg1) => Self::Add(arg0.clone(), arg1.clone()),
            Self::Sub(arg0, arg1) => Self::Sub(arg0.clone(), arg1.clone()),
            Self::Mul(arg0, arg1) => Self::Mul(arg0.clone(), arg1.clone()),
            Self::Div(arg0, arg1) => Self::Div(arg0.clone(), arg1.clone()),
            Self::Pow(arg0, arg1) => Self::Pow(arg0.clone(), arg1.clone()),
            Self::UnaryApp(arg0, arg1) => Self::UnaryApp(arg0.clone(), arg1.clone()),
            Self::BinaryApp(arg0, arg1, arg2) => Self::BinaryApp(arg0.clone(), arg1.clone(), arg2.clone()),
            Self::Var(arg0) => Self::Var(arg0.clone())
        }
    }
}
/*
  Display implementation for Expr
*/

impl std::fmt::Display for Expr {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      Expr::Num(n) => write!(f, "{}", n),
      Expr::Const(c) => {
        match c {
          ConstantToken::Pi => write!(f, "π"),
          ConstantToken::E => write!(f, "e")
        }
      },
      Expr::Add(a, b) => write!(f, "({} + {})", a, b),
      Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
      Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
      Expr::Div(a, b) => write!(f, "({} / {})", a, b),
      Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
      Expr::UnaryApp(fun, arg) => write!(f, "{}({})", fun, arg),
      Expr::BinaryApp(fun, a, b) => write!(f, "{}({}, {})", fun, a, b),
      Expr::Var(v) => write!(f, "{}", v)
    }
  }
}

/*
  Getting the vars in an expression
*/

impl Expr {

  pub fn vars(&self) -> Vec<String> {
    let vars_set = self.vars_aux();
    let mut vars_vec = Vec::new();
    for var in vars_set {
      vars_vec.push(var);
    };
    vars_vec.sort();
    vars_vec
  }

  fn vars_aux(&self) -> HashSet<String> {
    match self {
      Expr::Num(_) => HashSet::new(),
      Expr::Const(_) => HashSet::new(),
      Expr::Add(a, b) => {
        let mut vars = a.vars_aux();
        vars.extend(b.vars_aux());
        vars
      },
      Expr::Sub(a, b) => {
        let mut vars = a.vars_aux();
        vars.extend(b.vars_aux());
        vars
      },
      Expr::Mul(a, b) => {
        let mut vars = a.vars_aux();
        vars.extend(b.vars_aux());
        vars
      },
      Expr::Div(a, b) => {
        let mut vars = a.vars_aux();
        vars.extend(b.vars_aux());
        vars
      },
      Expr::Pow(a, b) => {
        let mut vars = a.vars_aux();
        vars.extend(b.vars_aux());
        vars
      },
      Expr::UnaryApp(_, a) => a.vars_aux(),
      Expr::BinaryApp(_, a, b) => {
        let mut vars = a.vars_aux();
        vars.extend(b.vars_aux());
        vars
      },
      Expr::Var(v) => {
        let mut vars = HashSet::new();
        vars.insert(v.clone());
        vars
      }
    }
  }
}

/*
  Substitution in Expr
*/

impl Expr {
  pub fn subs(&self, var: &str, val: &Expr) -> Expr {
    match self {
        Expr::Num(_) => self.clone(),
        Expr::Const(_) => self.clone(),
        Expr::Add(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          (new_a + new_b).eval()
        },
        Expr::Sub(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          (new_a - new_b).eval()
        },
        Expr::Mul(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          Expr::Mul(Box::new(new_a), Box::new(new_b))
        },
        Expr::Div(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          Expr::Div(Box::new(new_a), Box::new(new_b))
        },
        Expr::Pow(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          Expr::Pow(Box::new(new_a), Box::new(new_b))
        },
        Expr::UnaryApp(f, a) => {
          let new_a = a.subs(var, val);
          Expr::UnaryApp(f.clone(), Box::new(new_a))
        },
        Expr::BinaryApp(f, a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          Expr::BinaryApp(f.clone(), Box::new(new_a), Box::new(new_b))
        },
        Expr::Var(v) => {
          if v == var {
            val.clone()
          }
          else {
            self.clone()
          }
        }
    }
  }
}

/*
  Evaluation of expressions
*/

impl Expr {
  pub fn eval(&self) -> Expr {

    fn eval_add(a: &Expr, b: &Expr) -> Expr {
      let a = a.eval();
      let b = b.eval();

      // check if one of them is 0

      match (&a, &b) {
        (Expr::Num(x), Expr::Num(y)) => Expr::Num(x + y),
        (Expr::Num(0.), _) => b,
        (_, Expr::Num(0.)) => a,
        _ => a + b
      }
    }

    fn eval_sub(a: &Expr, b: &Expr) -> Expr {
      let a = a.eval();
      let b = b.eval();

      // check if one of them is 0

      match (&a, &b) {
        (Expr::Num(0.), Expr::Num(n)) => Expr::Num(-n),
        (Expr::Num(x), Expr::Num(y)) => Expr::Num(x - y),
        (_, Expr::Num(0.)) => a,
        _ => a - b
      }

    }

    fn eval_mul(a: &Expr, b: &Expr) -> Expr {
      let a = a.eval();
      let b = b.eval();

      match (&a, &b) {
        (Expr::Num(x), Expr::Num(y)) => Expr::Num(x * y),
        (Expr::Num(0.), _) => Expr::Num(0.),
        (_, Expr::Num(0.)) => Expr::Num(0.),
        (Expr::Num(1.), _) => b,
        (_, Expr::Num(1.)) => a,
        _ => a * b
      }
    }

    fn eval_div(a: &Expr, b: &Expr) -> Expr {
      let a = a.eval();
      let b = b.eval();

      match (&a, &b) {
        (Expr::Num(x), Expr::Num(y)) => Expr::Num(x / y),
        (Expr::Num(0.), _) => Expr::Num(0.),
        (_, Expr::Num(1.)) => a,
        _ => a / b
      }
    }

    match self {
      Expr::Num(_) => self.clone(),
      Expr::Const(c) => Self::eval_const(c),
      Expr::Add(a, b) => eval_add(a, b),
      Expr::Sub(a, b) => eval_sub(a, b),
      Expr::Mul(a, b) => eval_mul(a, b),
      Expr::Div(a, b) => eval_div(a, b),
      Expr::Pow(a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        // if new_a and new_b are both Num, then we can add them and output a new Num
        match (&new_a, &new_b) {
          (Expr::Num(x), Expr::Num(y)) => Expr::Num(x.powf(*y)),
          _ => Expr::Pow(Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::UnaryApp(f, a) => {
        let new_a = a.eval();
        match (f, &new_a) {
          (UnaryFun::Ln, Expr::Num(x)) => Expr::Num(x.ln()),
          (UnaryFun::ReLU, Expr::Num(x)) => Expr::Num(x.max(0.0)),
          (UnaryFun::ReLUPrime, Expr::Num(x)) => Expr::Num(if x > &0.0 { 1.0 } else { 0.0 }),
          _ => Expr::UnaryApp(f.clone(), Box::new(new_a))
        }
      },

      Expr::BinaryApp(f, a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        match (f, &new_a, &new_b) {
          (BinaryFun::Max, Expr::Num(x), Expr::Num(y)) => Expr::Num(x.max(*y)),
          (BinaryFun::MaxPrime, Expr::Num(x), Expr::Num(y)) => Expr::Num(if x > y { 1.0 } else { 0.0 }),
          _ => Expr::BinaryApp(f.clone(), Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::Var(_) => self.clone()

    }
  }

  fn eval_const(c: &ConstantToken) -> Expr {
    match c {
      ConstantToken::Pi => Expr::Num(std::f64::consts::PI),
      ConstantToken::E => Expr::Num(std::f64::consts::E)
    }
  }

}

/*
  Calculus operations on Expr
*/
impl Expr {

  // Numerical differentiation
  pub fn diff_num(&self, wrt: &String) -> Expr {
    // The derivative is taken with respect to wrt
    /*
      The derivative is approximately (f(x + h) - f(x)) / h
      for small h > 0
    */

    let h = 0.000001;

    let x_plus_h = Expr::Var(wrt.clone()) + h;

    let f_x_plus_h = self.subs(wrt, &x_plus_h).eval();

    let f_x = self.eval();

    (f_x_plus_h - f_x) / h
  }

  
  pub fn diff(&self, wrt: &String) -> Expr {

    // The derivative is taken with respect to wrt
    match self {
      Expr::Num(_) => Expr::Num(0.0),
      Expr::Const(_) => Expr::Num(0.0),
      Expr::Var(v) => {
        /*
         * If this is wrt, then the derivative is 1
         */
        if v == wrt {
          Expr::Num(1.0)
        }
        else {
          Expr::Num(0.0)
        }
      },

      Expr::Add(a, b) => {
        let new_a = a.diff(wrt);
        let new_b = b.diff(wrt);
        new_a + new_b
      },

      Expr::Sub(a, b) => {
        let new_a = a.diff(wrt);
        let new_b = b.diff(wrt);
        new_a - new_b
      },

      Expr::Mul(a, b) => {
        let new_a = a.diff(wrt);
        let new_b = b.diff(wrt);
        let a = a.eval();
        let b = b.eval();
        new_a * b.clone() + a.clone() * new_b
      },

      Expr::Div(a, b) => {
        let new_a = a.diff(wrt);
        let new_b = b.diff(wrt);
        let a = a.eval();
        let b = b.eval();
        (new_a * b.clone() - a * new_b) / (b.clone() * b.clone())
      },

      Expr::Pow(a, b) => {

        let a_prime = a.diff(wrt);
        let b_prime = b.diff(wrt);

        let a = a.eval();
        let b = b.eval();
        
        let coeff = Expr::Pow(Box::new(a.clone()), Box::new(b.clone()));

        let rest = ((b * a_prime) / a.clone()) + Expr::UnaryApp(UnaryFun::Ln, Box::new(a)) * b_prime.clone();

        (coeff * rest).eval()
      },

      Expr::UnaryApp(f, a) => Self::diff_app(f, a, wrt).eval(),

      Expr::BinaryApp(f, a, b) => {
        panic!("Diff BinaryApp not implemented")
      }
            
    }
  }

  fn diff_app(f: &UnaryFun, a: &Expr, wrt: &String) -> Expr {
    match f {
      UnaryFun::Ln => {
        let new_a = a.diff(wrt);
        new_a / a.clone()
      },
      UnaryFun::ReLU => {
        Expr::UnaryApp(UnaryFun::ReLUPrime, Box::new(a.clone())) * a.diff(wrt)
      },
      UnaryFun::ReLUPrime => {
        Expr::Num(0.0)
      }
    }
  }

  pub fn grad(&self, vars: Vec<String>) -> Matrix {
    let mut data = Vec::new();
    
    // for each variable, compute the derivative of the expression with respect to that variable
    
    for var in &vars {
      let deriv = self.diff_num(&var);
      let mut v = Vec::new();
      v.push(deriv);
      data.push(v);
    };

    Matrix {
      data,
      rows: (&vars).len(),
      cols: 1
    }

  }

  pub fn hessian(&self) -> Matrix {
    let vars = self.vars();
    let mut data = Vec::new();
    for var in &vars {
      let mut row = Vec::new();
      for var2 in &vars {
        let deriv = self.diff(&var);
        let deriv2 = deriv.diff(&var2);
        row.push(deriv2);
      }
      data.push(row);
    }
    Matrix {
      data,
      rows: (&vars).len(),
      cols: (&vars).len()
    }
  
  }

}

/*
  Matrix struct with implementations
*/

pub struct Matrix {
  pub data: Vec<Vec<Expr>>,
  rows: usize,
  cols: usize,
}

impl Matrix {
  /**
   * Turns a matrix of expressions into a matrix of floats. If this is possible,
   * returns Some(matrix::Matrix), otherwise, returns None.
   */
  pub fn to_float_matrix(&self) -> Option<matrix::Matrix> {
    // try to evaluate each element of the matrix to a number
    // if this is not possible, return None

    let mut data = Vec::new();

    for i in 0..self.rows {
      let mut row = Vec::new();
      for j in 0..self.cols {
        let elem = self.data[i][j].eval();
        match elem {
          Expr::Num(n) => row.push(n),
          _ => return None
        }
      };
      data.push(row);
    }

    return Some(matrix::Matrix::new(data));

  }
}

/*
  Substitution in Matrix
*/
impl Matrix {
  pub fn subs(&self, var: &String, val: Expr) -> Matrix {
    let mut new_data = Vec::new();
    for i in 0..self.rows {
      let mut row = Vec::new();
      for j in 0..self.cols {
        row.push(self.data[i][j].subs(var, &val));
      }
      new_data.push(row);
    }
    Matrix {
      data: new_data,
      rows: self.rows,
      cols: self.cols
    }
  }
}

/*
  Evaluation of Matrix
*/
impl Matrix {
  pub fn eval(&self) -> Matrix {
    let mut new_data = Vec::new();
    for i in 0..self.rows {
      let mut row = Vec::new();
      for j in 0..self.cols {
        row.push(self.data[i][j].eval());
      }
      new_data.push(row);
    }
    Matrix {
      data: new_data,
      rows: self.rows,
      cols: self.cols
    }
  }
}

impl Clone for Matrix {
  fn clone(&self) -> Self {
    let mut data = Vec::new();
    for i in 0..self.rows {
      let mut row = Vec::new();
      for j in 0..self.cols {
        row.push(self.data[i][j].clone());
      }
      data.push(row);
    }
    Matrix {
      data,
      rows: self.rows,
      cols: self.cols
    }
  }
}

/*
  Display implementation for Matrix
*/
impl std::fmt::Display for Matrix {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    let mut s = String::new();
    for i in 0..self.rows {
      s.push_str("[");
      for j in 0..self.cols {
        s.push_str(&format!("{}", self.data[i][j]));
        if j < self.cols - 1 {
          s.push_str(", ");
        }
      }
      s.push_str("]");
      if i < self.rows - 1 {
        s.push_str("\n");
      }
    }
    write!(f, "{}", s)
  }
}

/*
  Constructors for expressions
*/

pub fn n(n: f64) -> Expr {
  Expr::Num(n)
}

pub fn e() -> Expr {
  Expr::Const(ConstantToken::E)
}

pub fn pi() -> Expr {
  Expr::Const(ConstantToken::Pi)
}

pub fn ln(x: Expr) -> Expr {
  Expr::UnaryApp(UnaryFun::Ln, Box::new(x))
}

pub fn relu(x: Expr) -> Expr {
  Expr::UnaryApp(UnaryFun::ReLU, Box::new(x))
}

pub fn relu_prime(x: Expr) -> Expr {
  Expr::UnaryApp(UnaryFun::ReLUPrime, Box::new(x))
}

pub fn max(x: Expr, y: Expr) -> Expr {
  Expr::BinaryApp(BinaryFun::Max, Box::new(x), Box::new(y))
}

pub fn var(v: &str) -> Expr {
  Expr::Var(v.to_string())
}

/*
  Arithmetic operations for expressions
*/

/*
  Add
*/
impl Add for Expr {
  type Output = Expr;

  fn add(self, other: Expr) -> Expr {
    Expr::Add(Box::new(self), Box::new(other))
  }
}

impl Add<&Expr> for Expr {
  type Output = Expr;

  fn add(self, other: &Expr) -> Expr {
    Expr::Add(Box::new(self), Box::new(other.clone()))
  }
}

impl Add<Expr> for &Expr {
  type Output = Expr;

  fn add(self, other: Expr) -> Expr {
    Expr::Add(Box::new(self.clone()), Box::new(other))
  }
}

impl Add<Expr> for f64 {
  type Output = Expr;

  fn add(self, other: Expr) -> Expr {
    Expr::Add(Box::new(Expr::Num(self)), Box::new(other))
  }
}

impl Add<f64> for Expr {
  type Output = Expr;

  fn add(self, other: f64) -> Expr {
    Expr::Add(Box::new(self), Box::new(Expr::Num(other)))
  }
}

/*
  Sub
*/

impl Sub for Expr {
  type Output = Expr;

  fn sub(self, other: Expr) -> Expr {
    Expr::Sub(Box::new(self), Box::new(other))
  }
}

impl Sub<Expr> for f64 {
  type Output = Expr;

  fn sub(self, other: Expr) -> Expr {
    Expr::Sub(Box::new(Expr::Num(self)), Box::new(other))
  }
}

impl Sub<f64> for Expr {
  type Output = Expr;

  fn sub(self, other: f64) -> Expr {
    Expr::Sub(Box::new(self), Box::new(Expr::Num(other)))
  }
}


/*
  Mul
*/
impl ops::Mul for Expr {
  type Output = Expr;

  fn mul(self, other: Expr) -> Expr {
    Expr::Mul(Box::new(self), Box::new(other))
  }
}

impl ops::Mul<Expr> for f64 {
  type Output = Expr;

  fn mul(self, other: Expr) -> Expr {
    Expr::Mul(Box::new(Expr::Num(self)), Box::new(other))
  }
}

impl ops::Mul<f64> for Expr {
  type Output = Expr;

  fn mul(self, other: f64) -> Expr {
    Expr::Mul(Box::new(self), Box::new(Expr::Num(other)))
  }
}

/*
  Div
*/

impl ops::Div for Expr {
  type Output = Expr;

  fn div(self, other: Expr) -> Expr {
    Expr::Div(Box::new(self), Box::new(other))
  }
}

impl ops::Div<Expr> for f64 {
  type Output = Expr;

  fn div(self, other: Expr) -> Expr {
    Expr::Div(Box::new(Expr::Num(self)), Box::new(other))
  }
}

impl ops::Div<f64> for Expr {
  type Output = Expr;

  fn div(self, other: f64) -> Expr {
    Expr::Div(Box::new(self), Box::new(Expr::Num(other)))
  }
}

impl Expr {
  pub fn pow(&self, other: Expr) -> Expr {
    Expr::Pow(Box::new(self.clone()), Box::new(other))
  }
}