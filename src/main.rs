use core::num;
use std::env;
use std::sync::Arc;

use combinatorial_optimization_solver::model::{IsingModel, QuboModel};
use combinatorial_optimization_solver::opt::TspNode;
use combinatorial_optimization_solver::solver::annealing_scheduler::AnnealingScheduler;
use combinatorial_optimization_solver::solver::simulated_annealing::SimulatedAnnealing;
use combinatorial_optimization_solver::solver::simulated_quantum_annealing::SimulatedQuantumAnnealing;
use combinatorial_optimization_solver::solver::{Solver, SolverVariant};
use combinatorial_optimization_solver::webhook::Webhook;
use ndarray::{array, Array1, Array2};
use ndarray_linalg::Scalar;
use rand::distributions::Uniform;
use rand::{Rng, SeedableRng};
use serenity::model::channel::Embed;
use std::convert::TryFrom;

#[tokio::main]
async fn main() {
    dotenv::dotenv().expect("Not found .env file");
    let url = env::var("WEBHOOK_URL").unwrap();

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
    let try_number_of_times = 30;
    let range_param_start = 3.;
    let range_param_end = 1e-06;
    let solvers = vec![
        SolverVariant::Sa(SimulatedAnnealing::new(
            range_param_start,
            range_param_end,
            steps,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1.,
            steps,
            1,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.025,
            steps,
            40,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.0125,
            steps,
            80,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.00625,
            steps,
            160,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.003125,
            steps,
            320,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.0015625,
            steps,
            640,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.00078125,
            steps,
            1280,
            Arc::clone(&ising),
            None,
        )),
    ];

    let mut scheduler = AnnealingScheduler::new(solvers, try_number_of_times);
    let records = scheduler.run();
    let analysis_records = AnnealingScheduler::analysis(&records);

    let embed = Embed::fake(move |e| {
        let mut fields = Vec::new();
        for ar in &analysis_records {
            fields.push((
                format!("{}\nparameter {}", ar.solver_name, ar.parameter),
                format!(
                    "[best {}; ave {}; worst {};]\nbits {}",
                    ar.best_energy, ar.average_energy, ar.worst_energy, ar.best_state
                ),
                false,
            ));
        }
        e.title("Result")
            .description(format!(
                "sum {}; m {}; try_number_of_times {}",
                &sum, &m, &try_number_of_times
            ))
            .fields(fields)
    });

    let webhook = Webhook::new(&url);
    webhook.send(embed).await.unwrap();

    //     for record in &records {
    //         let cost = record
    //             .bits
    //             .map(|f| *f as f64)
    //             .dot(&Array1::from_iter(numbers.clone().into_iter()));

    //         println!("\n---- result ----");
    //         println!("param [m {}; sum {}]", &m, &sum);
    //         println!("{};\ncost {}", &record, (cost - m).abs());
    //     }
}
