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
