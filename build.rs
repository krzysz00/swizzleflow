use std::env;

fn main() {
    let home_dir = env::var("HOME").unwrap();
    println!("cargo:rustc-link-search=native={}/progs/lib", home_dir);
    println!("cargo:rustc-link-search=native={}/.local/lib", home_dir);
    println!("cargo:rustc-link-lib=static=blis");
}
