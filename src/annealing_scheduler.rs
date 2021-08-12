use rayon::{prelude::*, ThreadPoolBuilder};
use std::sync::mpsc;

use crate::solver::{SolutionRecord, Solver, StatisticsRecord};

pub struct AnnealingScheduler<T>
where
    T: Solver + Send + Sync + Sized,
{
    solvers: Vec<Box<T>>,
    try_number_of_times: usize,
}

impl<T> AnnealingScheduler<T>
where
    T: Solver + Send + Sync + Sized,
{
    pub fn new(solvers: Vec<Box<T>>, try_number_of_times: usize) -> Self {
        AnnealingScheduler {
            solvers,
            try_number_of_times,
        }
    }

    pub fn run(&mut self) -> Vec<Vec<SolutionRecord>> {
        let cpus = num_cpus::get();
        ThreadPoolBuilder::new()
            .num_threads(cpus)
            .build_global()
            .unwrap();

        let mut records: Vec<Vec<SolutionRecord>> = Vec::new();

        for solver in &self.solvers {
            let mut solver_parallel = Vec::new();
            solver_parallel.resize_with(self.try_number_of_times, || solver.clone_solver());

            let mut record: Vec<SolutionRecord> =
                solver_parallel.par_iter_mut().map(|s| s.solve()).collect();

            records.push(record);
        }

        records
    }

    fn analysis(records: &mut Vec<Vec<SolutionRecord>>) -> Vec<StatisticsRecord> {
        let mut statistics_records = Vec::new();

        for record in records {
            let best_record = record.iter().fold(SolutionRecord::default(), |acc, r| {
                if acc.energy > r.energy {
                    r.clone()
                } else {
                    acc
                }
            });
            let best_energy = best_record.energy;
            let best_state = best_record.bits;
            let average_energy = record.iter().map(|r| r.energy).sum::<f64>() / record.len() as f64;

            statistics_records.push(StatisticsRecord {
                solver_name: record[0].solver_name.clone(),
                best_energy,
                average_energy,
                best_state,
                parameter: record[0].parameter.clone(),
            });
        }

        statistics_records
    }
}
