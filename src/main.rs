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
extern crate swizzleflow;

use clap::clap_app;
use std::path::Path;
use std::io::BufReader;


use swizzleflow::problem_desc::ProblemDesc;
use swizzleflow::matrix_load;
use swizzleflow::synthesis::{Mode, synthesize};
use swizzleflow::misc::{open_file,parse_opt_arg};

use swizzleflow::errors::*;

const DEFAULT_MATRIX_DIR: &str = "matrices/";
const PRUNE_FUEL_ARG_REQS: &'static str = "a positive integer";
const PRUNE_FUEL_FRAC_ARG_REQS: &'static str = "a floating point number in (0, 1]";

fn run() -> Result<()> {
    let args =
        clap_app!(swizzleflow =>
                  (version: "0.5.0")
                  (author: "Krzysztof Drewniak <krzysd@cs.washington.edu> et al.")
                  (about: "Tool for synthesizing high-performance kernels from dataflow graph sketches")
                  (@arg matrix_dir: -m --("matrix-dir") +takes_value value_name("MATRIX_DIR")
                   default_value_os(DEFAULT_MATRIX_DIR.as_ref())
                   "Directory to store matrices in")
                  (@arg all: -a --all "Find all solutions")
                  (@arg print: -p --print "Print trace of valid solutions")
                  (@arg print_pruned: -P --("print-pruned") "Print pruned solutions")
                  (@arg prune_fuel: -f --("prune-fuel") [FUEL] "Number of terms to pair with every term during pruning")
                  (@arg prune_fuel_frac: -F --("prune-fuel-frac") [FUEL_FRAC] conflicts_with("prune_fuel") "Fraction of term count to pair with every term during pruning")
                  (@arg specs: ... value_name("SPEC") "Specification files (stdin if none specified)")
        ).setting(clap::AppSettings::TrailingVarArg).get_matches();

    let synthesis_mode = if args.is_present("all") { Mode::All } else { Mode::First };
    let print = args.is_present("print");
    let print_pruned = args.is_present("print_pruned");

    let prune_fuel = parse_opt_arg::<u64>(args.value_of("prune_fuel"),
                                          "--prune-fuel", PRUNE_FUEL_ARG_REQS)?
        .map(|v| v as usize);
    if !(prune_fuel.map(|v| v > 0).unwrap_or(true)) {
        return Err(ErrorKind::InvalidCmdArg("--prune-fuel",
                                            PRUNE_FUEL_ARG_REQS).into());
    }
    let prune_fuel_frac = parse_opt_arg::<f64>(args.value_of("prune_fuel_frac"),
                                               "--prune-fuel-frac",
                                               PRUNE_FUEL_FRAC_ARG_REQS)?;
    if !(prune_fuel_frac.map(|v| v >= 0.0 && v <= 1.0).unwrap_or(true)) {
        return Err(ErrorKind::InvalidCmdArg("--prune-fuel-frac",
                                            PRUNE_FUEL_FRAC_ARG_REQS).into());
    }

    let matrix_dir = Path::new(args.value_of_os("matrix_dir").unwrap()); // We have a default
    let specs: Vec<(ProblemDesc, String)> = match args.values_of_os("specs") {
        Some(iter) => {
            iter.map(
                |path| {
                    let name = path.to_string_lossy().into_owned();
                    let file = open_file(path)?;
                    serde_json::from_reader(BufReader::new(file))
                        .chain_err(|| ErrorKind::ParseError(path.into()))
                        .map(|v| (v, name))
                }).collect::<Result<Vec<_>>>()?
        },
        None => vec![(serde_json::from_reader(BufReader::new(std::io::stdin().lock()))?,
                      "stdin".into())]
    };

    for (desc, name) in specs {
        println!("spec:{}", name);
        let spec = desc.get_spec().chain_err(|| ErrorKind::BadSpec(desc.clone()))?;
        let domain = desc.make_domain(spec.view());
        let mut levels = desc.get_levels()
            .chain_err(|| ErrorKind::BadSpec(desc.clone()))?;
        let (initial, target, expected_syms) =
            desc.build_problem(&domain, &levels, spec)
            .chain_err(|| ErrorKind::BadSpec(desc.clone()))?;
        let max_lanes = initial.len();
        matrix_load::add_matrices(matrix_dir, &mut levels, max_lanes)?;
        let max_syms = expected_syms.iter().map(|l| l.len()).max().unwrap_or(1);
        let fuel_arg = if let Some(f) = prune_fuel {
            f
        } else if let Some(frac) = prune_fuel_frac {
            (frac * (max_syms as f64)).ceil() as u64 as usize
        } else {
            max_syms
        };
        let fuel = std::cmp::min(fuel_arg, max_syms);
        println!("Begin search");
        synthesize(initial.clone(), &target, &levels, &expected_syms, synthesis_mode,
                   print, print_pruned, fuel,  &name);
        matrix_load::remove_matrices(&mut levels);
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        use std::io::Write;
        use error_chain::ChainedError; // trait which holds `display_chain`
        let stderr = &mut std::io::stderr();

        writeln!(stderr, "{}", e.display_chain()).expect("Failed to write error message");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use crate::problem_desc::{trove, poly_mult};
    use crate::operators::swizzle::{xform,rotate};
    use crate::operators::load::{load_rep,broadcast};
    use crate::operators::{identity_gather, transpose_gather};
    use crate::state::{ProgState, Domain, Value};

    fn fixed_solution_from_scalar_8x3<'d>(d: &'d Domain) -> ProgState<'d> {
        let initial = ProgState::linear(d, 1, &[24]);
        let shape = [8, 3];
        let s0 = initial.gather_by(&load_rep(&[24], &[8, 3]).unwrap()
                                   .gathers().unwrap()[0]);
        let s1 = s0.gather_by(&xform(&shape, 1, 0, 1, 2, 1, 8, None, false));
        let s2 = s1.gather_by(&rotate(&shape, 1, 0, 1, 0, None));
        let s3 = s2.gather_by(&xform(&shape, 0, 1, 0, 3, 1, 3, None, false));
        let s4 = s3.gather_by(&rotate(&shape, 0, 1, 0, 0, None));
        s4
    }

    #[test]
    fn trove_solution_works() {
        let symbols: ndarray::Array1<Value> = (0u16..24u16).map(Value::Symbol).collect();
        let symbols = symbols.into_dyn();
        let domain = Domain::new(symbols.view());
        let spec = ProgState::new_from_spec(&domain, trove(8, 3), "trove").unwrap();
        let solution = fixed_solution_from_scalar_8x3(&domain);
        println!("spec {}\n solution {}", spec, solution);
        assert_eq!(spec, solution);
    }

    #[test]
    fn poly_mult_works() {
        use ndarray::ArrayD;
        use itertools::iproduct;
        use crate::operators::select::{cond_keep_gather,Op, BinOp};
        let f1 = xform(&[4, 4], 1, 0, 1, 1, 0, 4, None, false);
        let r1 = rotate(&[4, 4], 1, 0, 1, 0, None);
        let f2 = xform(&[4, 4], 0, 1, 1, 1, -1, 4, None, false);
        let r2 = rotate(&[4, 4], 1, 0, 1, 0, None);
        let broadcast = broadcast(&[4, 4], &[4, 2, 4], 1).unwrap().gathers().unwrap()[0].clone();
        let spec = poly_mult(4);
        let domain = Domain::new(spec.view());
        let arr1 = iproduct!(0..4, 0..4).map(|(_i, j)| crate::state::Value::Symbol(j)).collect();
        let arr1 = ArrayD::from_shape_vec(vec![4, 4], arr1).unwrap();
        let state = ProgState::new_from_spec(&domain, arr1, "init").unwrap();
        let s1 = state.gather_by(&f1);
        let s2 = s1.gather_by(&r1);

        let arr2 = iproduct!(0..4, 0..4).map(|(_i, j)| crate::state::Value::Symbol(4 + j)).collect();
        let arr2 = ArrayD::from_shape_vec(vec![4, 4], arr2).unwrap();
        let state2 = ProgState::new_from_spec(&domain, arr2, "init").unwrap();
        let s3 = state2.gather_by(&f2);
        let s4 = s3.gather_by(&r2);

        let m = ProgState::stack_folding(&[&s2, &s4]).unwrap();
        let b = m.gather_by(&broadcast);

        let mut retain = std::collections::BTreeMap::new();
        retain.insert(1, 0);
        let c1 = cond_keep_gather(&[4, 2, 4], 2, 0, 0,
                                  BinOp::Plus, Op::Leq, &retain);
        let d1 = b.gather_by(&c1);

        retain.insert(1, 1);
        let c2 = cond_keep_gather(&[4, 2, 4], 2, 0, 0,
                                  BinOp::Plus, Op::Gt, &retain);
        let d2 = d1.gather_fold_by(&c2).unwrap();

        let tr = transpose_gather(&[4, 2], &[2, 4]);
        let transposed = d2.gather_by(&tr);
        let reshape = identity_gather(&[8]);
        let result = transposed.gather_by(&reshape);
        let spec_state = ProgState::new_from_spec(&domain, spec, "spec").unwrap();
        assert_eq!(result, spec_state);
    }
}
