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
use crate::misc::{ShapeVec};
use crate::lexer::{Token,TokenType};
use crate::builtins::{Opt,OptMap};
use crate::state::{Value, Symbolic,Gather};

use std::collections::{HashSet, HashMap, BTreeMap, BTreeSet};
use std::convert::TryInto;
use std::fmt::Write;
use std::num::NonZeroUsize;

use ndarray::{Ix, Ixs, ArrayD};
use itertools::Itertools;

use TokenType::*;

// The grammar:
// Program ::= Statement+ EOF
// Statement ::=
//    Annot* 'define' Definition
//  | Annot* Ident [':' Type] = Literal | Call
//  | 'goal' Goal
// Annot ::= '@' Ident
// Type ::= '[' (epsilon | (Number ',')* Number [',']) ']'

// ArrFinal(E) ::= E [','] ']' | Array(E) [','] ']'
// ArrCont(E) ::= E ',' | Array(E) ','
// Array(E) ::= '[' ArrCont(E)* ArrFinal(E)

// Tuple(E) ::= '(' (E ',')* E [','] ')'

// Options ::= epsilon | '{' (Option ',')* [Option]'}'
// Option ::= Ident | Ident '=' Number | Ident '=' Type

// Range ::= 'range(' Number ','  Number ')
// Literal ::= 'fold'* (Array(Number) | Range)
// CallArgs ::= epsilon | (Ident ',')* Ident
// Call ::= [Fold ['(']] Ident Options '('Args ')' ['->' Type] [')'] # Parent

// Gather ::= Array(Number | '(' [Number ','] (Number | Tuple(Number)) ')')
// DefineOne ::= Ident Tuple(Type) '->' Type '=' Gather
// DefineManyPart ::= (Ident | String) '=' Gather
// ManyGathers ::= epsilon | (DefineManyPart ',')* DefineManyPart [',']
// DefineMany ::= Ident Tuple(Type) '->' Type '=' '{' ManyGathers '}'
// Definiton ::= DefineOne | DefineMany

// Goal ::= Ident Options | Literal
// type ParseFunc<T> = Fn(&[Token], usize) -> Result<(T, usize)>;

#[inline(always)]
fn error_enum(tok: &Token, err: &'static str) -> Error {
    ErrorKind::ParseError(tok.clone(), err).into()
}

#[inline(always)]
fn error<T>(tok: &Token, err: &'static str) -> Result<T> {
    Err(ErrorKind::ParseError(tok.clone(), err).into())
}

fn recognize(typ: TokenType, err: &'static str)
             -> impl Fn(&[Token], usize) -> Result<((), usize)> {
    move |toks, pos| {
        if toks[pos].t.matches(&typ) {
            Ok(((), pos + 1))
        }
        else {
            error(&toks[pos], err)
        }
    }
}

fn parse_ident(toks: &[Token], pos: usize) -> Result<(String, usize)> {
    let tok = &toks[pos];
    if let Ident(ref s) = tok.t {
        Ok((s.clone(), pos + 1))
    }
    else {
        error(tok, "identifier")
    }
}

fn parse_num(toks: &[Token], pos: usize) -> Result<(i64, usize)> {
    let tok = &toks[pos];
    if let Number(n) = tok.t {
        Ok((n, pos + 1))
    }
    else {
        error(tok, "number")
    }
}

fn parse_seq<T, Fstart, Fitem, Fend>(start: Fstart, mut item: Fitem, end: Fend,
                                     toks: &[Token], pos: usize) -> Result<(Vec<T>, usize)>
where Fstart: Fn(&[Token], usize) -> Result<((), usize)>,
      Fitem: FnMut(&[Token], usize) -> Result<(T, usize)>,
      Fend: Fn(&[Token], usize) -> Result<((), usize)> {
    let (_, mut pos) = start(toks, pos)?;
    let mut ret = Vec::<T>::new();
    loop {
        if let Ok((_, pos2)) = end(toks, pos) {
            return Ok((ret, pos2));
        }
        let (val, new_pos) = item(toks, pos)?;
        ret.push(val);
        if let Ok((_, pos2)) = end(toks, new_pos) {
            return Ok((ret, pos2));
        }
        else if toks[new_pos].t == Comma {
            pos = new_pos + 1;
        }
        else {
            return error(&toks[new_pos], "comma or closing delimiter");
        }
    }
}

fn parse_type(toks: &[Token], pos: usize) -> Result<(ShapeVec, usize)> {
    let (nums, p2) = parse_seq(recognize(LSquare, "["),
                               |toks, pos| parse_num(toks, pos)
                               .and_then(|(n, p)|
                                         if n > 0 { Ok((n as usize, p)) }
                                         else { error(&toks[pos], "positive number") } ),
                               recognize(RSquare, ", or ]"), toks, pos)?;
    if nums.len() > 0 {
        Ok((nums.into(), p2))
    }
    else {
        error(&toks[pos], "non-empty list")
    }
}

fn parse_array_internal<T, F>(data: &mut Vec<T>, shapes: &mut Vec<Option<usize>>, n: usize,
                              item: &F, toks: &[Token], mut pos: usize) -> Result<(usize, usize)>
where F: Fn(&[Token], usize) -> Result<(T, usize)>, T: std::fmt::Debug {
    // First time visiting array level
    if n == shapes.len() {
        shapes.push(None);
    }
    let mut count = 0;
    let subarray = toks[pos].t == LSquare;
    loop {
        // Empty arrays aren't allowed here
        if subarray {
            if toks[pos].t != LSquare {
                return error(&toks[pos], "[");
            }
            let (count, new_pos) = parse_array_internal(data, shapes, n + 1,
                                                        item, toks, pos + 1)?;
            match shapes[n + 1] {
                Some(k) if k != count => {
                    return Err(ErrorKind::BadArrayLen(count, k, toks[pos].clone()).into());
                }
                None => {
                    shapes[n + 1] = Some(count);
                }
                _ => ()
            };
            pos = new_pos;
        }
        else {
            let (x, new_pos) = item(toks, pos)?;
            data.push(x);
            pos = new_pos;
        }
        count += 1;

        if toks[pos].t == Comma {
            pos = pos + 1;
        }
        else {
            if toks[pos].t == RSquare {
                return Ok((count, pos + 1));
            }
            else {
                return error(&toks[pos], ", or ]")
            }
        }

        if toks[pos].t == RSquare {
            return Ok((count, pos + 1));
        }
    }
}

fn parse_array<T, F>(item: F, toks: &[Token], pos: usize) -> Result<(ArrayD<T>, usize)>
where F: Fn(&[Token], usize) -> Result<(T, usize)>, T: std::fmt::Debug {
    if toks[pos].t != LSquare {
        return error(&toks[pos], "[");
    }
    let mut data: Vec<T> = vec![];
    let mut shapes: Vec<Option<usize>> = vec![];
    let (outer_count, pos) =
        parse_array_internal(&mut data, &mut shapes, 0, &item, toks, pos + 1)?;
    shapes[0] = Some(outer_count);
    let shapes = shapes.into_iter().collect::<Option<Vec<usize>>>().unwrap();
    // If there's not the right amount of elements in there, something's up
    let arr = ArrayD::from_shape_vec(shapes, data).unwrap();
    return Ok((arr, pos));
}


fn parse_option(toks: &[Token], pos: usize) -> Result<((String, Opt), usize)> {
    if let Ident(ref s) = toks[pos].t {
        match &toks[pos + 1].t {
            Comma | RCurly => {
                Ok(((s.clone(), Opt::Flag), pos + 1))
            },
            Equal => {
                match &toks[pos + 2].t {
                    Number(n) => {
                        Ok(((s.clone(), Opt::Int(*n as isize)), pos + 3))
                    }
                    LSquare => {
                        let (ns, new_pos) =
                            parse_seq(recognize(LSquare, "["),
                                      |t, p| parse_num(t, p).map(|(n, p2)| (n as isize, p2)),
                                      recognize(RSquare, ", or ]"), toks, pos + 2)?;
                        Ok(((s.clone(), Opt::Arr(ns)), new_pos))
                    }
                    _ => {
                        error(&toks[pos + 2], "number or array")
                    }
                }
            },
            _ => error(&toks[pos + 1], ", or =")
        }
    }
    else {
        error(&toks[pos], "option name")
    }
}

fn parse_options(toks: &[Token], pos: usize) -> Result<(Option<OptMap>, usize)> {
    if toks[pos].t != LCurly {
        Ok((None, pos))
    }
    else {
        parse_seq(recognize(LCurly, "{ or ("), parse_option, recognize(RCurly, ", or }")
                  ,toks, pos)
            .map(|(v, p)| (Some(v.into_iter().collect::<OptMap>()), p))
    }
}

fn parse_two_args<A, B, Fa, Fb>(f_a: Fa, f_b: Fb,
                                toks: &[Token], pos: usize) -> Result<((A, B), usize)>
where Fa: Fn(&[Token], usize) -> Result<(A, usize)>,
      Fb: Fn(&[Token], usize) -> Result<(B, usize)> {
    if toks[pos].t != LParen {
        return error(&toks[pos + 1], "(");
    }
    let (a, p2) = f_a(toks, pos+1)?;
    if toks[p2].t != Comma {
        return error(&toks[p2], ",")
    }
    let (b, p3) = f_b(toks, p2 +1)?;
    if toks[p3].t != RParen {
        return error(&toks[p3], ")");
    }
    Ok(((a, b), p3 + 1))
}

fn parse_literal(shape: Option<&ShapeVec>,
                 toks: &[Token], pos: usize) -> Result<(ArrayD<Value>, usize)> {
    let n_folds = toks[pos..].iter().take_while(|tok| tok.t == Fold).count();
    let pos = pos + n_folds;
    let parsed = match &toks[pos].t {
        Range => {
            let ((n1, n2), new_pos) = parse_two_args(parse_num, parse_num, toks, pos + 1)?;
            if n2 <= n1 || n1 < 0 || n2 < 0 {
                return Err(ErrorKind::InvalidRange(n1, n2,
                                                   toks[pos].clone()).into());
            }
            let len = (n2 - n1) as usize;
            let shape = shape.map(|v| v.to_vec())
                .unwrap_or_else(|| vec![len]);
            let data = (n1..n2).map(|x| Value::Symbol(x as Symbolic))
                .collect();
            let arr = ArrayD::from_shape_vec(shape, data)
                .map_err(|_| Error::from(
                    ErrorKind::InvalidRange(n1, n2,
                                            toks[pos].clone())))?;
            Ok((arr, new_pos))
        },
        LSquare => {
            let (a, np) =
                parse_array(|t, p| parse_num(t, p)
                            .map(|(n, p2)|(Value::Symbol(n as Symbolic), p2)),
                            toks, pos)?;
            if let Some(s) = shape {
                if s.as_slice() != a.shape() &&
                    s.as_slice() != &a.shape()[0..a.ndim()-n_folds] {
                    return Err(ErrorKind::ShapeMismatch(s.to_vec(), a.shape().to_vec()).into());
                }
            }
            Ok((a, np))
        }
        _ => error(&toks[pos], "range or ["),
    };
    let (mut arr, pos) = parsed?;
    for _ in 0..n_folds {
        arr = arr.map_axis(ndarray::Axis(arr.ndim() - 1),
                           |data| Value::fold(data.to_vec()));
    }
    Ok((arr, pos))
}

fn signed_to_opt_ix(index: &[isize], shape: &[usize]) -> isize {
    index.iter().copied().zip(shape.iter().copied())
        .try_fold(0isize, |acc, (idx, s)| {
            // Bail on overflow
            let ss: isize = s.try_into().ok()?;
            if idx < 0 || idx > ss {
                None
            }
            else {
                Some(acc * ss + idx)
            }
        }).unwrap_or(-1)
}

fn parse_gather_element(shapes: &[ShapeVec],
                        toks: &[Token], pos: usize) -> Result<((Ix, Ixs), usize)> {
    if let Ok((n, p2)) = parse_num(toks, pos) {
        return Ok(((0, n as Ixs), p2));
    }
    else if toks[pos].t == LParen {
        let (arg, p2) =
            if toks[pos+1].t == LParen { (0, pos + 1) }
            else {
                let (n, p2) = parse_num(toks, pos + 1)
                    .map(|(n, p)| (if n < 0 { shapes.len() } else { n as usize }, p))
                    .map_err(|_| error_enum(&toks[pos + 1], "argument index or ("))?;
                if toks[p2].t == Comma {
                    (n, p2 + 1)
                }
                else {
                    return error(&toks[p2], ",")
                }
            };
        if let Ok((idx, pos)) = parse_num(toks, p2) {
            let (_, pos) = recognize(RParen, ")")(toks, pos)?;
            return Ok(((arg, idx as isize), pos));
        }
        let (items, pos) = parse_seq(recognize(LParen, "index or ("),
                              |t, p| parse_num(t, p).map(|(n, p2)| (n as Ixs, p2)),
                                   recognize(RParen, ", or )"), toks, p2)?;
        let idx = signed_to_opt_ix(&items, &shapes[arg]);
        let (_, pos) = recognize(RParen, ")")(toks, pos)?;
        return Ok(((arg, idx), pos));
    }
    else {
        error(&toks[pos], "index or tuple")
    }
}

fn parse_gather_array(shapes: &[ShapeVec], out_shape: &ShapeVec,
                      toks: &[Token], pos: usize) -> Result<(ArrayD<(Ix, Ixs)>, usize)> {
    let (arr, pos) = parse_array(|t, p| parse_gather_element(shapes, t, p), toks, pos)?;
    let arr = arr.into_shape(out_shape.as_slice())
        .map_err(|_| error_enum(&toks[pos], "array that fits specified output shape"))?;
    Ok((arr, pos))
}

fn parse_one_gather(shapes: &[ShapeVec], out_shape: &ShapeVec,
                    toks: &[Token], pos: usize) -> Result<(Gather, usize)> {
    let name = match &toks[pos].t {
        Ident(s) => Ok(s.clone()),
        Str(s) => Ok(s.clone()),
        _ => error(&toks[pos], "identifier or quoted string")
    };
    let name = name?;
    let (_, pos) = (recognize(Equal, "="))(toks, pos + 1)?;
    let (arr, pos) = parse_gather_array(shapes, out_shape, toks, pos)?;
    let gather = Gather::new_raw(arr, name);
    Ok((gather, pos))
}


fn parse_many_gathers(shapes: &[ShapeVec], out_shape: &ShapeVec,
                      toks: &[Token], pos: usize) -> Result<(Vec<Gather>, usize)> {
    parse_seq(recognize(LCurly, "{"),
              |t, p| parse_one_gather(shapes, out_shape, t, p),
              recognize(RCurly, ", or }"),
              toks, pos).map(|(arr, p)|
                             (arr.into_iter()
                              .collect::<HashSet<_>>().into_iter()
                              .collect::<Vec<_>>(), p))
}

const STATEMENT_START_ERR: &'static str = "'goal', 'define', or identifier";
fn parse_define(toks: &[Token], pos: usize) ->
    Result<((String, Vec<ShapeVec>, ShapeVec, Vec<Gather>), usize)>
{
    let (_, pos) = recognize(Define, STATEMENT_START_ERR)(toks, pos)?;
    let (name, pos) = parse_ident(toks, pos)?;
    let (shapes, pos) = parse_seq(recognize(LParen, "("), parse_type,
                                  recognize(RParen, ")"), toks, pos)?;
    let (_, pos) = recognize(Arrow, "->")(toks, pos)?;
    let (out_shape, pos) = parse_type(toks, pos)?;
    let (_, pos) = recognize(Equal, "=")(toks, pos)?;
    let main_result = match toks[pos].t {
        LSquare => parse_gather_array(&shapes, &out_shape, toks, pos)
            .map(|(g, p)| (vec![Gather::new_raw(g, name.clone())], p)),
        LCurly => parse_many_gathers(&shapes, &out_shape, toks, pos),
        _ => error(&toks[pos], "gather or set of gathers ([ or {])"),
    };
    let (arr, pos) = main_result?;
    Ok(((name, shapes, out_shape, arr), pos))
}

fn parse_goal(toks: &[Token], pos: usize) -> Result<(ArrayD<Value>, usize)> {
    let (_, pos) = recognize(Goal, STATEMENT_START_ERR)(toks, pos)?;
    let (_, pos) = recognize(Colon, ":")(toks, pos)?;
    let (shape, pos) = parse_type(toks, pos)?;
    match toks[pos].t {
        Equal => parse_literal(Some(&shape), toks, pos + 1),
        Ident(ref s) => {
            let (options, pos) = parse_options(toks, pos + 1)?;
            let goal = crate::builtins::goal(s, shape.as_slice(), options.as_ref())?;
            Ok((goal, pos))
        }
        _ => error(&toks[pos], "'=' [literal] or builtin gather"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VarIdx {
    Dep(usize),
    Here(usize),
}

#[derive(Clone, Debug)]
pub struct Dependency {
    pub parent_idx: VarIdx,
    pub used_at: BTreeSet<usize>,
    pub var: String,
    pub shape: ShapeVec,
}

#[derive(Clone, Debug)]
pub enum StmtType {
    Initial(ArrayD<Value>),
    Gathers(Vec<Gather>, Option<NonZeroUsize>),
    Block { body: Vec<Statement>, deps: Vec<Dependency> }
}

impl StmtType {
    pub fn is_initial(&self) -> bool {
        match self {
            StmtType::Initial(_) => true,
            _ => false,
        }
    }

    pub fn is_block(&self) -> bool {
        match self {
            StmtType::Block {body: _body, deps: _deps} => true,
            _ => false,
        }
    }
}


#[derive(Clone, Debug)]
pub struct Statement {
    pub op: StmtType,
    pub var: String,
    pub args: Vec<VarIdx>,
    pub in_shapes: Vec<ShapeVec>,
    pub out_shape: ShapeVec,
    pub name: String,
    pub used_at: BTreeSet<usize>,
    pub prune: bool,
}

type DefsMap = HashMap<(String, Vec<ShapeVec>, ShapeVec), Vec<Gather>>;

#[derive(Clone, Debug, Default)]
struct Scopes {
    maps: Vec<HashMap<String, VarIdx>>,
    deps: Vec<Vec<Dependency>>,
    stmts: Vec<Vec<Statement>>,
}

impl Scopes {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_rec(&mut self, key: &str, idx: usize) -> Option<VarIdx> {
        self.maps[idx].get(key).copied()
            .or_else(|| {
                if idx == 0 {
                    None
                }
                else { self.get_rec(key, idx - 1).map(|res| {
                    let deps_idx = self.deps[idx].len();
                    let shape = self._get_shape(idx - 1, res);
                    let var = self._get_var(idx - 1, res);
                    self.deps[idx].push(Dependency { parent_idx: res,
                                                     used_at: BTreeSet::new(),
                                                     shape, var });
                    VarIdx::Dep(deps_idx)
                })
                }
            })
    }

    pub fn get(&mut self, key: &str) -> Option<VarIdx> {
        self.get_rec(key, self.maps.len() - 1)
    }

    fn _get_shape(&self, scope_idx: usize, var: VarIdx) -> ShapeVec {
        match var {
            VarIdx::Dep(i) => self.deps[scope_idx][i].shape.clone(),
            VarIdx::Here(i) => self.stmts[scope_idx][i].out_shape.clone(),
        }
    }

    fn _get_var(&self, scope_idx: usize, var: VarIdx) -> String {
        match var {
            VarIdx::Dep(i) => self.deps[scope_idx][i].var.clone(),
            VarIdx::Here(i) => self.stmts[scope_idx][i].var.clone(),
        }
    }

    pub fn get_shape(&self, var: VarIdx) -> ShapeVec {
        self._get_shape(self.stmts.len() - 1, var)
    }

    pub fn contains_key(&mut self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn push_scope(&mut self) {
        self.maps.push(HashMap::new());
        self.deps.push(Vec::new());
        self.stmts.push(Vec::new());
    }

    pub fn pop_scope(&mut self) -> (Vec<Statement>, Vec<Dependency>) {
        let _ = self.maps.pop();
        let stmts = self.stmts.pop().expect("At least one scope");
        let deps = self.deps.pop().expect("At least one scope");
        (stmts, deps)
    }

    pub fn push_statement(&mut self, stmt: Statement) -> usize {
        let scope_idx = self.stmts.len() - 1;
        let stmt_pos = self.stmts[scope_idx].len();
        self.maps[scope_idx].insert(stmt.var.clone(), VarIdx::Here(stmt_pos));
        self.stmts[scope_idx].push(stmt);
        stmt_pos
    }

    pub fn update_used_at(&mut self, idx: usize, args: &[VarIdx]) {
        let scope_idx = self.stmts.len() - 1;
        for arg in args.iter().copied() {
            match arg {
                VarIdx::Here(i) => self.stmts[scope_idx][i].used_at.insert(idx),
                VarIdx::Dep(i) => self.deps[scope_idx][i].used_at.insert(idx),
            };
        }
    }
}

fn parse_call<'t>(custom_fns: &DefsMap, scopes: &mut Scopes,
                  var: &str, out_shape: &mut ShapeVec, prune: Option<bool>,
                  toks: &'t [Token], pos: usize) -> Result<((), usize)> {
    let (is_fold, fold_paren, pos) =
        if toks[pos].t == Fold {
            if toks[pos+1].t == LParen { (true, true, pos+2) }
            else { (true, false, pos + 1) }
        } else { (false, false, pos) };

    let pos = if toks[pos].t == Question { pos + 1 } else { pos };

    let ident_pos = pos;

    let (name, pos) = parse_ident(toks, pos)
        .map_err(|_| error_enum(&toks[pos], "identifier or ["))?;
    let (mut options, pos) = parse_options(toks, pos)?;
    let fold_len = if is_fold {
        crate::builtins::int_option(options.as_ref(), "fold_len")
            .and_then(|x| NonZeroUsize::new(x as usize))
            .or_else(|| {
                let len = out_shape.pop().unwrap();
                options.get_or_insert_with(|| BTreeMap::new())
                    .insert("fold_len".to_owned(), Opt::Int(len as isize));
                // Already returns an option
                NonZeroUsize::new(len)
            })
    } else { None };

    let (args, pos) = if let Some(i) = scopes.get(&name) {
        (vec![i], pos)
    } else {
        parse_seq(recognize(LParen, "( or {"),
                                |t, p| parse_ident(t, p)
                                .and_then(|(s, p2)| scopes.get(&s)
                                          .ok_or_else(|| Error::from(
                                              ErrorKind::ParseError(t[p].clone(),
                                              "previously-defined variable")))
                                          .map(|n| (n, p2))),
                  recognize(RParen, ", or )"), toks, pos)?
    };

    let pos = if fold_paren {
        recognize(RParen, ") to close fold")(toks, pos)?.1
    } else { pos };

    let in_shapes = args.iter().copied().map(|i| scopes.get_shape(i)).collect::<Vec<_>>();
    let mut gather_out_shape = out_shape.clone();
    if let Some(l) = fold_len { gather_out_shape.push(l.get()) }

    let mut lookup = (name, in_shapes, gather_out_shape);
    let gathers =
        if scopes.contains_key(&lookup.0) {
            // Variable copy
            lookup.0 = "identity".to_owned();
            crate::operators::identity(&[lookup.2.clone()],
                                       lookup.2.as_slice())?
        }
        else if let Some(g) = custom_fns.get(&lookup) {
            g.clone()
        } else {
            let (ref name, ref in_shapes, ref out_shape) = lookup;
            crate::builtins::gather(name, in_shapes, out_shape, options.as_ref())
                .chain_err(|| ErrorKind::ParseError(toks[ident_pos].clone(), "valid arguments to call after"))?
        };
    let (mut name, in_shapes, _) = lookup;
    if let Some(m) = options {
        name.push('{');
        for (k, v) in &m {
            name.push_str(k);
            match v {
                Opt::Flag => (),
                Opt::Int(n) =>{ write!(&mut name, "={}", n).unwrap(); },
                Opt::Arr(a) => {
                    write!(&mut name, "=[{}]",
                           a.iter().map(|i| i.to_string()).join(",")).unwrap() ;
                },
            };
            name.push(',');
        }
        name.pop();
        name.push('}');
    }

    let prune = prune.unwrap_or_else(|| gathers.len() > 1 || fold_len.is_some());

    let stmt_idx = scopes.push_statement(
        Statement { var: var.to_owned(),
                    in_shapes, out_shape: out_shape.clone(),
                    args: args.clone(), name, prune,
                    used_at: BTreeSet::new(),
                    op: StmtType::Gathers(gathers, fold_len) });
    scopes.update_used_at(stmt_idx, &args);
    Ok(((), pos))
}

fn parse_annots(toks: &[Token], mut pos: usize) -> Result<(HashSet<String>, usize)> {
    let mut ret = HashSet::new();
    loop {
        if toks[pos].t != Annot {
            return Ok((ret, pos));
        } else {
            pos += 1;
        }

        if let Ident(ref s) = toks[pos].t {
            ret.insert(s.clone());
            pos += 1;
        } else {
            return error(&toks[pos], "identifier-like annotation after @")
        }
    }
}

fn parse_block(custom_fns: &mut DefsMap, scopes: &mut Scopes,
               goals: &mut Vec<ArrayD<Value>>,
               toks: &[Token], mut pos: usize) -> Result<((), usize)> {
    if toks[pos].t != LCurly {
        return error(&toks[pos], "{ (can't happen)");
    }
    pos += 1;
    loop {
        if toks[pos].t == EOF {
            return error(&toks[pos], "closing } for block");
        }
        else if toks[pos].t == RCurly {
            return Ok(((), pos + 1))
        }
        else {
            let (_, new_pos) = parse_statement(custom_fns, scopes, goals, toks, pos)?;
            pos = new_pos;
        }
    }
}

fn parse_statement(custom_fns: &mut DefsMap, scopes: &mut Scopes,
                   goals: &mut Vec<ArrayD<Value>>,
                   toks: &[Token], pos: usize) -> Result<((), usize)> {
    let (annots, pos) = parse_annots(toks, pos)?;
    match toks[pos].t {
        Define => {
            let ((name, in_shapes, out_shape, gathers), pos) = parse_define(toks, pos)?;
            custom_fns.insert((name, in_shapes, out_shape), gathers);
            Ok(((), pos))
        },
        Goal => {
            let (goal, pos) = parse_goal(toks, pos)?;
            goals.push(goal);
            Ok(((), pos))
        },
        Ident(ref s) => {
            let var = s.to_owned();
            let (_, pos) = recognize(Colon, ": [shape]")(toks, pos + 1)?;
            let (mut out_shape, pos) = parse_type(toks, pos)?;
            let (_, pos) = recognize(Equal, "=")(toks, pos)?;

            let prune = if annots.contains("prune") {
                Some(true)
            } else if annots.contains("noprune") {
                Some(false)
            } else {
                None
            };
            let n_folds = toks[pos..].iter().take_while(|t| t.t == Fold).count();
            if toks[pos+n_folds].t == LSquare || toks[pos+n_folds].t == Range {
                let (literal, pos) = parse_literal(Some(&out_shape), toks, pos)
                    .chain_err(|| ErrorKind::ParseError(toks[pos].clone(), "valid literal"))?;
                let name = format!{"init_{}", var};
                scopes.push_statement(Statement {
                    var, in_shapes: vec![], out_shape,
                    used_at: BTreeSet::new(), args: vec![], name,
                    prune: prune.unwrap_or(false),
                    op: StmtType::Initial(literal) });
                Ok(((), pos))
            }
            else if toks[pos+n_folds].t == LCurly {
                if n_folds > 0 {
                    error(&toks[pos], "thene not to be folds on a block")
                }
                else {
                    scopes.push_scope();

                    let (_, new_pos) =
                        parse_block(custom_fns, scopes, goals, toks, pos)?;
                    let (stmts, deps) = scopes.pop_scope();
                    if stmts.is_empty() {
                        error(&toks[pos].clone(), "non-empty block of statements")
                    }
                    else {
                        if stmts[stmts.len() - 1].out_shape != out_shape {
                            Err(Error::from(
                                ErrorKind::ShapeMismatch(stmts[stmts.len() - 1].out_shape.to_vec(),
                                                         out_shape.to_vec()))
                                .chain_err(|| ErrorKind::ParseError(toks[pos].clone(),
                                                                    "output to match variable type")))
                        }
                        else {
                            let args: Vec<_> = deps.iter().map(|d| d.parent_idx).collect();
                            let in_shapes = deps.iter().map(|d| d.shape.clone()).collect();
                            let name = format!("block_outputs_{}", var);
                            let block_idx = scopes.push_statement(
                                Statement {op: StmtType::Block { body: stmts, deps: deps, },
                                           var, args: args.clone(), in_shapes, out_shape,
                                           name, used_at: BTreeSet::new(),
                                           prune: prune.unwrap_or(true) });
                            scopes.update_used_at(block_idx, args.as_slice());
                            Ok(((), new_pos))
                        }
                    }
                }
            }
            else {
                parse_call(custom_fns, scopes, &var,
                           &mut out_shape, prune,
                           toks, pos)
            }
        },
        _ => error(&toks[pos], STATEMENT_START_ERR),
    }
}

pub fn parse(toks: &[Token]) -> Result<(Vec<Statement>, Vec<ArrayD<Value>>)> {
    let mut pos = 0;
    let mut custom_fns = DefsMap::new();
    let mut scopes = Scopes::new();
    scopes.push_scope();
    let mut goals = vec![];

    loop {
        if toks[pos].t == EOF {
            // Last statement doesn't prune
            let (stmts, deps) = scopes.pop_scope();
            if !deps.is_empty() {
                panic!("Top-level program depends on something, somehow: {:?}", deps);
            }
            return Ok((stmts, goals));
        }
        let (_, new_pos) = parse_statement(
            &mut custom_fns, &mut scopes, &mut goals,
            toks, pos)?;
        pos = new_pos;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn works() {
        let example = "x: [2] = [0, 1]
y:[2]=[2,3]
define permute([2]) -> [2] = {id=[0, 1], \"flip\"=[(0, (1)), (0, (0))],}
define stack([2], [2]) -> [2, 2] = [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]
@noprune
z: [2] = permute(x)
s: [2] = fold(stack{fold_len=2}(z, y))
g: [2] = ?permute(s)
h: [2] = g
goal: [2] = fold [[0, 3], [1, 2]]";
        let tokens = crate::lexer::lex(&example).unwrap();
        let (vars, goals) = super::parse(&tokens).unwrap();
        assert_eq!(vars.len(), 6);
        assert_eq!(goals.len(), 1);
    }
}
