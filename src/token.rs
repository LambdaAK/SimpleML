

pub enum Token {
  LeftParen,
  RightParen,
  Num(f64),
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Var(String),

}
// debug
// derive
impl std::fmt::Debug for Token {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      Token::LeftParen => write!(f, "LeftParen"),
      Token::RightParen => write!(f, "RightParen"),
      Token::Num(n) => write!(f, "Num({})", n),
      Token::Add => write!(f, "Add"),
      Token::Sub => write!(f, "Sub"),
      Token::Mul => write!(f, "Mul"),
      Token::Div => write!(f, "Div"),
      Token::Pow => write!(f, "Pow"),
      Token::Var(v) => write!(f, "Var({})", v)
    }
  }
}
impl std::fmt::Display for Token {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      Token::LeftParen => write!(f, "("),
      Token::RightParen => write!(f, ")"),
      Token::Num(n) => write!(f, "{}", n),
      Token::Add => write!(f, "+"),
      Token::Sub => write!(f, "-"),
      Token::Mul => write!(f, "*"),
      Token::Div => write!(f, "/"),
      Token::Pow => write!(f, "^"),
      Token::Var(v) => write!(f, "{}", v)
    }
  }
}



/**
 * Lexes a single token from the input string, 
 */
pub fn lex_single_token(input: &[char]) -> (Token, &[char]) {

  match input {
    ['(', rest @ ..] => (Token::LeftParen, rest),
    [')', rest @ ..] => (Token::RightParen, rest),
    ['+', rest @ ..] => (Token::Add, rest),
    ['-', rest @ ..] => (Token::Sub, rest),
    ['*', rest @ ..] => (Token::Mul, rest),
    ['/', rest @ ..] => (Token::Div, rest),
    ['^', rest @ ..] => (Token::Pow, rest),
    [c, rest @ ..] if c.is_digit(10) => {
      let mut num = String::new();
      num.push(*c);
      let mut rest = rest;
      while let Some((next, new_rest)) = rest.split_first() {
        if next.is_digit(10) || next == &'.' {
          num.push(*next);
          rest = new_rest;
        } else {
          break;
        }
      }
      (Token::Num(num.parse().unwrap()), rest)
    },
    _ => panic!("Lexing failed, tokens: {}", input.iter().collect::<String>())
  }
}

pub fn lex_tokens(input: &str) -> Vec<Token> {
  let mut tokens = Vec::new();

  let mut input_chars: &[char] = &input.chars().collect::<Vec<char>>();

  // get rid of leading whitespace

  while let Some((next, rest)) = input_chars.split_first() {
    if next.is_whitespace() {
      input_chars = rest;
    } else {
      break;
    }
  }
  
  while !input_chars.is_empty() {
    let (token, rest) = lex_single_token(input_chars);
    tokens.push(token);
    input_chars = rest;
    while let Some((next, rest)) = input_chars.split_first() {
      if next.is_whitespace() {
        input_chars = rest;
      } else {
        break;
      }
    }
  };
  
  tokens
}