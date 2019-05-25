fn main() {
    // It's not clear why this dosent' carry over, but that's life
    pkg_config::find_library("mkl-dynamic-lp64-iomp").unwrap();
}
