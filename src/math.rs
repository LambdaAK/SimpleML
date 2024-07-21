
pub enum Expr {
  Num(f64),
  Add(Box<Expr>, Box<Expr>),
  Sub(Box<Expr>, Box<Expr>),
  Mul(Box<Expr>, Box<Expr>),
  Div(Box<Expr>, Box<Expr>),
  Pow(Box<Expr>, Box<Expr>),
  Var(String)
}







impl Clone for Expr {
    fn clone(&self) -> Self {
        match self {
            Self::Num(arg0) => Self::Num(arg0.clone()),
            Self::Add(arg0, arg1) => Self::Add(arg0.clone(), arg1.clone()),
            Self::Sub(arg0, arg1) => Self::Sub(arg0.clone(), arg1.clone()),
            Self::Mul(arg0, arg1) => Self::Mul(arg0.clone(), arg1.clone()),
            Self::Div(arg0, arg1) => Self::Div(arg0.clone(), arg1.clone()),
            Self::Pow(arg0, arg1) => Self::Pow(arg0.clone(), arg1.clone()),
            Self::Var(arg0) => Self::Var(arg0.clone())
        }
    }
}

impl std::fmt::Display for Expr {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      Expr::Num(n) => write!(f, "{}", n),
      Expr::Add(a, b) => write!(f, "({} + {})", a, b),
      Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
      Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
      Expr::Div(a, b) => write!(f, "({} / {})", a, b),
      Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),
      Expr::Var(v) => write!(f, "{}", v)
    }
  }
}

impl Expr {
  pub fn subs(&self, var: &str, val: &Expr) -> Expr {
    match self {
        Expr::Num(_) => self.clone(),
        Expr::Add(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          Expr::Add(Box::new(new_a), Box::new(new_b))
        },
        Expr::Sub(a, b) => {
          let new_a = a.subs(var, val);
          let new_b = b.subs(var, val);
          Expr::Sub(Box::new(new_a), Box::new(new_b))
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
    match self {
      Expr::Num(_) => self.clone(),
      Expr::Add(a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        // if new_a and new_b are both Num, then we can add them and output a new Num
        match (&new_a, &new_b) {
          (Expr::Num(x), Expr::Num(y)) => Expr::Num(x + y),
          _ => Expr::Add(Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::Sub(a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        // if new_a and new_b are both Num, then we can add them and output a new Num
        match (&new_a, &new_b) {
          (Expr::Num(x), Expr::Num(y)) => Expr::Num(x - y),
          _ => Expr::Sub(Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::Mul(a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        // if new_a and new_b are both Num, then we can add them and output a new Num
        match (&new_a, &new_b) {
          (Expr::Num(x), Expr::Num(y)) => Expr::Num(x * y),
          _ => Expr::Mul(Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::Div(a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        // if new_a and new_b are both Num, then we can add them and output a new Num
        match (&new_a, &new_b) {
          (Expr::Num(x), Expr::Num(y)) => Expr::Num(x / y),
          _ => Expr::Div(Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::Pow(a, b) => {
        let new_a = a.eval();
        let new_b = b.eval();
        // if new_a and new_b are both Num, then we can add them and output a new Num
        match (&new_a, &new_b) {
          (Expr::Num(x), Expr::Num(y)) => Expr::Num(x.powf(*y)),
          _ => Expr::Pow(Box::new(new_a), Box::new(new_b))
        }
      },

      Expr::Var(_) => self.clone()

    }
  }
}

impl Expr {
  pub fn diff(&self, wrt: &String) -> Expr {
    // The derivative is taken with respect to wrt
    match self {
      Expr::Num(_) => Expr::Num(0.0),
      Expr::Var(v) => {
        /**
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
        Expr::Add(Box::new(new_a), Box::new(new_b)).eval()
      },

      Expr::Sub(a, b) => {
        let new_a = a.diff(wrt);
        let new_b = b.diff(wrt);
        Expr::Sub(Box::new(new_a), Box::new(new_b)).eval()
      },
      
      _ => {
        todo!("Implement the rest of the differentiation rules")
      }

    }
  }
}