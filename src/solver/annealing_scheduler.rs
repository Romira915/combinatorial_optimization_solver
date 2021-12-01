use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::{rc::Rc, sync::mpsc};

use crate::solver::{
    simulated_annealing::SimulatedAnnealing,
    simulated_quantum_annealing::SimulatedQuantumAnnealing, SolutionRecord, Solver, SolverVariant,
    StatisticsRecord,
};

pub struct AnnealingScheduler {
    solvers: Vec<SolverVariant>,
    try_number_of_times: usize,
}

impl AnnealingScheduler {
    pub fn new(solvers: Vec<SolverVariant>, try_number_of_times: usize) -> Self {
        AnnealingScheduler {
            solvers,
            try_number_of_times,
        }
    }

    pub fn run(&self) -> Vec<Vec<SolutionRecord>> {
        let cpus = num_cpus::get();
        ThreadPoolBuilder::new()
            .num_threads(cpus)
            .build_global()
            .unwrap();

        let mut records: Vec<Vec<SolutionRecord>> = Vec::new();

        for solver in &self.solvers {
            let mut solver_parallel = Vec::new();
            solver_parallel.resize_with(self.try_number_of_times, || solver.clone());

            let record: Vec<SolutionRecord> = solver_parallel
                .par_iter_mut()
                .map(|s| s.solve())
                .inspect(|s| println!("{}\n", &s))
                .collect();

            records.push(record);
        }

        records
    }

    pub fn analysis(records: &Vec<Vec<SolutionRecord>>) -> Vec<StatisticsRecord> {
        let mut statistics_records = Vec::new();

        for record in records {
            let best_record = record.iter().fold(
                {
                    let mut init = SolutionRecord::default();
                    init.energy = f64::MAX;
                    init
                },
                |acc, r| {
                    if acc.energy > r.energy {
                        r.clone()
                    } else {
                        acc
                    }
                },
            );
            let best_energy = best_record.energy;
            let best_state = best_record.bits;

            let worst_record = record.iter().fold(
                {
                    let mut init = SolutionRecord::default();
                    init.energy = f64::MIN;
                    init
                },
                |acc, r| {
                    if acc.energy < r.energy {
                        r.clone()
                    } else {
                        acc
                    }
                },
            );
            let worst_energy = worst_record.energy;

            let average_energy = record.iter().map(|r| r.energy).sum::<f64>() / record.len() as f64;

            statistics_records.push(StatisticsRecord {
                solver_name: record[0].solver_name.clone(),
                best_energy,
                average_energy,
                worst_energy,
                best_state,
                parameter: record[0].parameter.clone(),
                states: record.to_vec(),
            });
        }

        statistics_records
    }
}
