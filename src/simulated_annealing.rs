use ndarray::Array2;

use crate::solver::Solver;

pub struct SimulatedAnnealing {
    T0: f32,
    Tf: f32,
    P: f32,
    N: u32,
    state: Array2<f32>,
}

impl SimulatedAnnealing {}

impl Solver for SimulatedAnnealing {
    fn solve(&mut self) -> String {
        String::new()
    }
}
