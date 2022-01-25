pub mod annealing_scheduler;
pub mod simulated_annealing;
pub mod simulated_quantum_annealing;

use std::{fmt::Display, usize};

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::WeightedAliasIndex;
use rand::{prelude::SliceRandom, thread_rng, Rng};

use self::{
    simulated_annealing::SimulatedAnnealing, simulated_quantum_annealing::SimulatedQuantumAnnealing,
};

pub trait Solver {
    fn solve(&mut self) -> SolutionRecord;
    fn record(&self) -> Option<SolutionRecord>;
    fn clone_solver(&self) -> Self;
}

#[derive(Debug, Default, Clone)]
pub struct SolutionRecord {
    pub solver_name: String,
    pub bits: Array1<i8>,
    pub energy: f64,
    pub parameter: String,
}

#[derive(Debug, Default, Clone)]
pub struct StatisticsRecord {
    pub solver_name: String,
    pub best_energy: f64,
    pub average_energy: f64,
    pub worst_energy: f64,
    pub best_state: Array1<i8>,
    pub parameter: String,
    pub states: Vec<SolutionRecord>,
}

#[derive(Debug, Clone)]
pub enum SolverVariant {
    Sa(SimulatedAnnealing),
    Sqa(SimulatedQuantumAnnealing),
}

impl Display for SolutionRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim = self.bits.dim() as f64;
        let dim = f64::sqrt(dim) as usize;
        write!(
            f,
            "spins\n {}; energy {}; flags {}\nparameter {}",
            self.bits.to_shape((dim, dim)).unwrap(),
            self.energy,
            self.bits.sum(),
            self.parameter
        )
    }
}

impl Display for StatisticsRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\nbest {}; ave {};",
            self.parameter, self.best_energy, self.average_energy
        )
    }
}

impl SolverVariant {
    pub fn solve(&mut self) -> SolutionRecord {
        match self {
            Self::Sa(sa) => sa.solve(),
            Self::Sqa(sqa) => sqa.solve(),
        }
    }
}

pub fn probability_boolean<R>(p: f64, rng: &mut R) -> bool
where
    R: Rng,
{
    assert_eq!(true, p <= 1.);
    assert_eq!(true, p >= 0.);
    let choices = [(true, p), (false, 1. - p)];

    choices.choose_weighted(rng, |item| item.1).unwrap().0
}
