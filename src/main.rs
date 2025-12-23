use sidd_gpt::Data;
use std::process;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let _data = Data::parse(&args[0]).unwrap_or_else(|err| {
        eprintln!("problem parsing data: {err}");
        process::exit(1);
    });
}
