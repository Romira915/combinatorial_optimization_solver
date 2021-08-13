use std::{cmp::min, sync::Arc};

use getset::Getters;
use ndarray::{Array, Array1, Array2, MathCell};
use rand::{
    prelude::{SliceRandom, StdRng, ThreadRng},
    Rng, SeedableRng,
};

use crate::{
    model::{IsingModel, QuboModel},
    solver::{self, SolutionRecord, Solver},
};

#[derive(Debug)]
pub struct SimulatedAnnealing {
    T0: f64,
    Tf: f64,
    steps: usize,
    // P: f32,
    N: usize,
    model: Arc<IsingModel>,
    spins: Array1<i8>,
    rng: StdRng,
    record: Option<SolutionRecord>,
}

impl SimulatedAnnealing {
    pub fn new(
        T0: f64,
        Tf: f64,
        steps: usize,
        model: Arc<IsingModel>,
        mut rng: Option<StdRng>,
    ) -> Self {
        let N = model.J().dim().0;
        let mut rng = match rng {
            Some(rng) => rng,
            None => rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap(),
        };
        SimulatedAnnealing {
            T0,
            Tf,
            steps,
            N,
            model,
            spins: IsingModel::init_spins(N, &mut rng),
            rng,
            record: None,
        }
    }
}

impl Solver for SimulatedAnnealing {
    fn solve(&mut self) -> SolutionRecord {
        let T_array = Array::linspace(self.T0, self.Tf, self.steps);

        for T in &T_array {
            let flip_index = self.rng.gen_range(0..self.N);
            let delta_E = self.model.calculate_dE(self.spins.view(), flip_index) as f64;
            let p = f64::min(1., (-delta_E / T).exp());

            if solver::probability_boolean(p, &mut self.rng) {
                IsingModel::accept_flip(self.spins.view_mut(), flip_index);
            }
        }

        let record = SolutionRecord {
            solver_name: "Simulated Annealing".to_string(),
            bits: self.spins.map(QuboModel::BITS_FROM_SPINS),
            energy: self.model.calculate_energy(self.spins.view()) as f64,
            parameter: format!("[T0 {}; Tf {}; steps {};]", self.T0, self.Tf, self.steps),
        };
        self.record = Some(record.clone());

        record
    }

    fn record(&self) -> Option<SolutionRecord> {
        self.record.clone()
    }

    fn clone_solver(&self) -> Self {
        let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();
        let obj = SimulatedAnnealing {
            T0: self.T0.clone(),
            Tf: self.Tf.clone(),
            steps: self.steps.clone(),
            N: self.N.clone(),
            model: self.model.clone(),
            spins: IsingModel::init_spins(self.N, &mut rng),
            rng,
            record: self.record.clone(),
        };

        obj
    }
}

impl Clone for SimulatedAnnealing {
    fn clone(&self) -> Self {
        let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();
        SimulatedAnnealing {
            T0: self.T0.clone(),
            Tf: self.Tf.clone(),
            steps: self.steps.clone(),
            N: self.N.clone(),
            model: self.model.clone(),
            spins: IsingModel::init_spins(self.N, &mut rng),
            rng,
            record: self.record.clone(),
        }
    }
}
