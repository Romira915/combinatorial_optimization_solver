use core::num;
use std::sync::Arc;

use combinatorial_optimization_solver::annealing_scheduler::{self, AnnealingScheduler};
use combinatorial_optimization_solver::model::{IsingModel, QuboModel};
use combinatorial_optimization_solver::solver::simulated_annealing::SimulatedAnnealing;
use combinatorial_optimization_solver::solver::simulated_quantum_annealing::SimulatedQuantumAnnealing;
use combinatorial_optimization_solver::solver::Solver;
use ndarray::{array, Array1, Array2};
use ndarray_linalg::Scalar;
use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};

fn main() {
    let n = 1000;
    let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();
    let numbers = (&mut rng)
        .sample_iter(Uniform::new(0., 1.))
        .take(n)
        .collect::<Vec<f64>>();
    let sum: f64 = numbers.iter().sum();
    let m = sum * 0.5;
    let J = {
        let mut J = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                J[[i, j]] = (numbers[i] * numbers[j]) as f32;
            }
        }
        J /= 2.;
        J
    };

    let h = {
        let mut h = Array1::zeros(n);
        let k = sum / 2. - m;
        for i in 0..n {
            h[i] = (k * numbers[i]) as f32;
        }
        h
    };

    let ising = Arc::new(IsingModel::new(J, h));

    let steps = 3e5 as usize;
    let try_number_of_times = 300;
    let solvers: Vec<Box<dyn Solver>> = vec![
        Box::new(SimulatedAnnealing::new(
            3.,
            1e-06,
            steps,
            Arc::clone(&ising),
            None,
        )),
        Box::new(SimulatedAnnealing::new(
            3.,
            1e-06,
            steps,
            Arc::clone(&ising),
            None,
        )),
        Box::new(SimulatedAnnealing::new(
            3.,
            1e-06,
            steps,
            Arc::clone(&ising),
            None,
        )),
        Box::new(SimulatedQuantumAnnealing::new(
            3.,
            1e-06,
            1.,
            steps,
            1,
            Arc::clone(&ising),
            None,
        )),
        Box::new(SimulatedQuantumAnnealing::new(
            3.,
            1e-06,
            0.025,
            steps,
            40,
            Arc::clone(&ising),
            None,
        )),
        Box::new(SimulatedQuantumAnnealing::new(
            3.,
            1e-06,
            0.0125,
            steps,
            80,
            Arc::clone(&ising),
            None,
        )),
    ];

    // let mut scheduler = AnnealingScheduler::new(solvers, try_number_of_times);
    // let records = scheduler.run();

    // for record in &records {
    //     let cost = record
    //         .bits
    //         .map(|f| *f as f64)
    //         .dot(&Array1::from_iter(numbers.clone().into_iter()));

    //     println!("\n---- result ----");
    //     println!("param [m {}; sum {}]", &m, &sum);
    //     println!("{};\ncost {}", &record, (cost - m).abs());
    // }
}
