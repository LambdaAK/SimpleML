use std::{collections::HashSet, fmt::{Display, Formatter}, ops::{self, Add, Sub}};

use crate::matrix;

pub enum Const {
  Pi,
  E
}

impl Clone for Const {
  fn clone(&self) -> Self {
    match self {
      Const::Pi => Const::Pi,
      Const::E => Const::E
    }
  }
}

pub enum Fun {
  Ln,
  ReLU,
  ReLUPrime
}

impl Fun {
  pub fn from_str(s: &str) -> Option<Fun> {
    match s {
      "ln" => Some(Fun::Ln),
      "relu" => Some(Fun::ReLU),
      _ => None
    }
  }
}

impl Clone for Fun {
  fn clone(&self) -> Self {
    match self {
      Fun::Ln => Fun::Ln,
      Fun::ReLU => Fun::ReLU,
      Fun::ReLUPrime => Fun::ReLUPrime
    }
  }
}

impl Display for Fun {
  fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
    match self {
      Fun::Ln => write!(f, "ln"),
      Fun::ReLU => write!(f, "ReLU"),
      Fun::ReLUPrime => write!(f, "ReLU'")
    }
  }
}

pub enum Expr {
  Num(f64),
  Const(Const),
  Add(Box<Expr>, Box<Expr>),
  Sub(Box<Expr>, Box<Expr>),
  Mul(Box<Expr>, Box<Expr>),
  Div(Box<Expr>, Box<Expr>),
  Pow(Box<Expr>, Box<Expr>),
  App(Fun, Box<Expr>),
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
            Self::App(arg0, arg1) => Self::App(arg0.clone(), arg1.clone()),
            Self::Var(arg0) => Self::Var(arg0.clone())
        }
    }
}

impl std::fmt::Display for Expr {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      Expr::Num(n) => write!(f, "{}", n),
      Expr::Const(c) => {
        match c {
          Const::Pi => write!(f, "π"),
          Const::E => write!(f, "e")
        }
      },
      Expr::Add(a, b) => write!(f, "({} + {})", a, b),
      Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
      Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
      Expr::Div(a, b) => write!(f, "({} / {})", a, b),
      Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
      Expr::App(fun, arg) => write!(f, "{}({})", fun, arg),
      Expr::Var(v) => write!(f, "{}", v)
    }
  }
}

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
      Expr::App(_, a) => a.vars_aux(),
      Expr::Var(v) => {
        let mut vars = HashSet::new();
        vars.insert(v.clone());
        vars
      }
    }
  }
}

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
        Expr::App(f, a) => {
          let new_a = a.subs(var, val);
          Expr::App(f.clone(), Box::new(new_a))
        }
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

      Expr::App(f, a) => {
        let new_a = a.eval();
        match (f, &new_a) {
          (Fun::Ln, Expr::Num(x)) => Expr::Num(x.ln()),
          _ => Expr::App(f.clone(), Box::new(new_a.clone()))
        }
      },

      Expr::Var(_) => self.clone()

    }
  }

  fn eval_const(c: &Const) -> Expr {
    match c {
      Const::Pi => Expr::Num(std::f64::consts::PI),
      Const::E => Expr::Num(std::f64::consts::E)
    }
  }

}

impl Expr {
  
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

        let rest = ((b * a_prime) / a.clone()) + Expr::App(Fun::Ln, Box::new(a)) * b_prime.clone();

        (coeff * rest).eval()
      },

      Expr::App(f, a) => Self::diff_app(f, a, wrt).eval(),

    }
  }

  fn diff_app(f: &Fun, a: &Expr, wrt: &String) -> Expr {
    match f {
      Fun::Ln => {
        let new_a = a.diff(wrt);
        new_a / a.clone()
      },
      Fun::ReLU => {
        Expr::App(Fun::ReLUPrime, Box::new(a.clone())) * a.diff(wrt)
      },
      Fun::ReLUPrime => {
        Expr::Num(0.0)
      }
    }
  }

  pub fn grad(&self) -> Matrix {
    let vars = self.vars();
    let mut data = Vec::new();
    
    // for each variable, compute the derivative of the expression with respect to that variable
    
    for var in &vars {
      let deriv = self.diff(&var);
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

}

impl Add for Expr {
  type Output = Expr;

  fn add(self, other: Expr) -> Expr {
    Expr::Add(Box::new(self), Box::new(other))
  }
}

impl Sub for Expr {
  type Output = Expr;

  fn sub(self, other: Expr) -> Expr {
    Expr::Sub(Box::new(self), Box::new(other))
  }
}

impl ops::Mul for Expr {
  type Output = Expr;

  fn mul(self, other: Expr) -> Expr {
    Expr::Mul(Box::new(self), Box::new(other))
  }
}

impl ops::Div for Expr {
  type Output = Expr;

  fn div(self, other: Expr) -> Expr {
    Expr::Div(Box::new(self), Box::new(other))
  }
}

pub struct Matrix {
  data: Vec<Vec<Expr>>,
  rows: usize,
  cols: usize,
}

impl Matrix {
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

