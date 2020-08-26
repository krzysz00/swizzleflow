// Copyright (C) 2019 Krzysztof Drewniak et al.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use crate::errors::*;

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenType {
    Ident(String),
    Str(String),
    Number(i64),
    LParen,
    RParen,
    LSquare,
    RSquare,
    LCurly,
    RCurly,
    Equal,
    Colon,
    Question,
    Comma,
    Annot,
    Fold,
    Define,
    Range,
    Target,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TokenType::*;
        match self {
            Ident(s) => write!(f, "{}", s),
            Str(s) => write!(f, "\"{}\"", s.replace("\"", "\\\"")),
            Number(n) => write!(f, "{}", n),
            LParen => write!(f, "("),
            RParen => write!(f, ")"),
            LSquare => write!(f, "["),
            RSquare => write!(f, "]"),
            LCurly => write!(f, "{{"),
            RCurly => write!(f, "}}"),
            Equal => write!(f, "="),
            Colon => write!(f, ":"),
            Question => write!(f, "?"),
            Comma => write!(f, ","),
            Annot => write!(f, "@"),
            Fold => write!(f, "fold"),
            Define => write!(f, "define"),
            Range => write!(f, "range"),
            Target => write!(f, "target"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub t: TokenType,
    pub line: usize,
    pub col: usize,
}


fn finish_ident(s: &str) -> TokenType {
    if let Ok(n) = s.parse::<i64>() {
        return TokenType::Number(n);
    }
    match s {
        "fold" => TokenType::Fold,
        "define" => TokenType::Define,
        "range" => TokenType::Range,
        "target" => TokenType::Target,
        _ => TokenType::Ident(s.to_owned()),
    }
}

fn char_to_taken(c: char) -> Option<TokenType> {
    use TokenType::*;
    match c {
        '(' => Some(LParen),
        ')' => Some(RParen),
        '[' => Some(LSquare),
        ']' => Some(RSquare),
        '{' => Some(LCurly),
        '}' => Some(RCurly),
        ':' => Some(Colon),
        '?' => Some(Question),
        ',' => Some(Comma),
        '@' => Some(Annot),
        '+' => Some(Fold),
        '*' => Some(Fold),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LexMode {
    Normal, Ident, String, Comment,
}

pub fn lex(input: &str) -> Result<Vec<Token>> {
    use TokenType::*;

    let mut col = 0;
    let mut line = 1;

    let mut start_line = 0;
    let mut start_col = 1;
    let mut start_pos = 0;

    let mut ret = Vec::<Token>::new();

    let mut mode = LexMode::Normal;

    let maybe_deident = |start_pos: usize, pos: usize, mode: LexMode| {
        if mode == LexMode::Ident {
            Some(finish_ident(&input[start_pos.. pos]))
        }
        else {
            None
        }
    };

    // Add trailing \n for uniformity at EOF
    for (pos, c) in input.char_indices().chain(std::iter::once((input.len(), '\n'))) {
        if mode == LexMode::String {
            if c == '"' && pos > 0 && AsRef::<[u8]>::as_ref(input)[pos - 1] != b'\\' {
                let s = input[start_pos+1..pos].replace("\\\"", "\"");
                ret.push(Token {t: Str(s), col: start_col, line: start_line});
                mode = LexMode::Normal;
            }
        }
        else if mode == LexMode::Comment { }
        else if let Some(t) = char_to_taken(c) {
            if let Some(t2) = maybe_deident(start_pos, pos, mode) {
                ret.push(Token {t: t2, line: start_line, col: start_col});
                mode = LexMode::Normal;
            }
            ret.push(Token {t, line, col});
        }
        else if c.is_whitespace() {
            if let Some(t) = maybe_deident(start_pos, pos, mode) {
                ret.push(Token {t, line: start_line, col: start_col});
                mode = LexMode::Normal;
            }
        }
        else if c == '#' {
            if let Some(t) = maybe_deident(start_pos, pos, mode) {
                ret.push(Token {t, line: start_line, col: start_col});
            }
            mode = LexMode::Comment;
        }
        else if c == '"' {
            if let Some(t) = maybe_deident(start_pos, pos, mode) {
                ret.push(Token {t, line: start_line, col: start_col});
            }

            start_pos = pos;
            start_line = line;
            start_col = col;
            mode = LexMode::String;
        }
        else if mode == LexMode::Normal {
            start_pos = pos;
            start_line = line;
            start_col = col;
            mode = LexMode::Ident;
        }

        if c == '\n' {
            line += 1;
            col = 0;
            if mode == LexMode::Comment {
                mode = LexMode::Normal;
            }
        }
        else {
            col += 1;
        }
    }
    if mode == LexMode::String {
        Err(ErrorKind::UnclonedString(start_line, start_col).into())
    }
    else {
        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn unclosed_string() {
        lex("\"foo").unwrap();
    }

    #[test]
    fn token_split() {
        let input = "a b?c:d,e[f]g(h)i{j}+k*l@m";
        let tokens = lex(input).unwrap();
        // a to l is 12 letters with 10 separators (space doesn't count)
        println!("{:?}", tokens);
        assert_eq!(tokens.len(), 25);
    }

    #[test]
    fn word_test() {
        use TokenType::*;
        let input = "fold define\nfoo\ttarget \"foo\\\"\" range";
        let tokens = lex(input).unwrap().iter().map(|tok| tok.t.clone()).collect::<Vec<_>>();
        assert_eq!(&tokens, &[Fold, Define, Ident("foo".to_owned()),
                              Target, Str("foo\"".to_owned()), Range]);
    }

    #[test]
    fn test_comments() {
        let input = "foo#bar\nbaz";
        let tokens = lex(input).unwrap();
        assert_eq!(tokens.len(), 2);
    }
}
