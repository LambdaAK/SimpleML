use crate::token::Token;
use crate::math::Expr;



pub fn parse_l4 (tokens: &[Token]) -> (Expr, &[Token]) {

  let (mut expr, mut rest) = parse_l3(tokens);

  loop {
    match rest {
      [Token::Add, tail @ ..] => {
        let (next_expr, next_rest) = parse_l3(tail);
        expr = Expr::Add(Box::new(expr), Box::new(next_expr));
        rest = next_rest;
      },
      [Token::Sub, tail @ ..] => {
        let (next_expr, next_rest) = parse_l3(tail);
        expr = Expr::Sub(Box::new(expr), Box::new(next_expr));
        rest = next_rest;
      },
      _ => return (expr, rest)
    }
  }

}

fn parse_l3(tokens: &[Token]) -> (Expr, &[Token]) {
  // parse a list of l2s

  let (mut expr, mut rest) = parse_l2(tokens);

  loop {
    match rest {
      [Token::Mul, tail @ ..] => {
        let (next_expr, next_rest) = parse_l2(tail);
        expr = Expr::Mul(Box::new(expr), Box::new(next_expr));
        rest = next_rest;
      },
      [Token::Div, tail @ ..] => {
        let (next_expr, next_rest) = parse_l2(tail);
        expr = Expr::Div(Box::new(expr), Box::new(next_expr));
        rest = next_rest;
      },
      _ => return (expr, rest)
    }
  }

}

fn parse_l2(tokens: &[Token]) -> (Expr, &[Token]) {
  // parse a list of l1s

  let (mut expr, mut rest) = parse_l1(tokens);

  loop {
    match rest {
      [Token::Pow, tail @ ..] => {
        let (next_expr, next_rest) = parse_l1(tail);
        expr = Expr::Pow(Box::new(expr), Box::new(next_expr));
        rest = next_rest;
      },
      _ => return (expr, rest)
    }
  }
}

fn parse_l1 (tokens: &[Token]) -> (Expr, &[Token]) {
  match tokens {
    [Token::Num(x), rest @ ..] => (Expr::Num(*x), rest),
    [Token::Var(s), rest @ ..] => (Expr::Var(s.to_string()), rest),
    [Token::LeftParen, rest @ ..] => {
      let (expr, rest) = parse_l4(rest);
      match rest {
        [Token::RightParen, rest @ ..] => (expr, rest),
        _ => panic!("Expected a right paren")
      }
    },
    _ => panic!("Expected a number or a left paren in parse_l1")
  }
}

