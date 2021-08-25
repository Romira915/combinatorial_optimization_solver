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

    let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();

    let mut tsp = TspNode::try_from("./dataset/ulysses16.tsp").unwrap();
    tsp.set_bias(2.);
    let qubo = QuboModel::from(tsp.clone());
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);

    let steps = 3e5 as usize;
    let try_number_of_times = 1;
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
    ];

    let scheduler = AnnealingScheduler::new(solvers, try_number_of_times);
    let records = scheduler.run();
    let analysis_records = AnnealingScheduler::analysis(&records);

    let embed = Embed::fake(move |e| {
        let mut fields = Vec::new();
        for ar in &analysis_records {
            let best_len = tsp.len_from_state(ar.best_state.view());
            let best_len = match best_len {
                Ok(len) => len.to_string(),
                Err((len, message)) => format!("{} ({})", len, &message),
            };
            fields.push((
                format!("{}\nparameter {}", ar.solver_name, ar.parameter),
                format!(
                    "[best {}; ave {}; worst {}; best_len {}]\nbits {}",
                    ar.best_energy, ar.average_energy, ar.worst_energy, best_len, ar.best_state
                ),
                false,
            ));
        }
        let strict_solution = tsp.opt_len().unwrap_or_default();
        e.title("Result")
            .description(format!(
                "dataset {}; try_number_of_times {}; 厳密解 {}",
                tsp.data_name(),
                &try_number_of_times,
                strict_solution
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
