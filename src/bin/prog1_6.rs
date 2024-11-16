use gnuplot::{AxesCommon, Figure};
use nnlm::perceptron::Perceptron;
use nnlm::*;

fn main() {
    for dist in [-4.0, 1.0, 0.0, 4.0] {
        const NEPOCHS: usize = 50;
        let central_radius = 10.0;
        let radius_variation = 6.0;
        let (data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
        let data: Vec<_> = data
            .into_iter()
            .map(|[x, y]| [1.0, x, y]) // Add 1.0 - gives the perceptron a bias term
            .collect();
        let tr_data = &data[..1000];
        let tr_labels = &labels[..1000];
        let te_data = &data[1000..];
        let te_labels = &labels[1000..];
        let mut model = Perceptron::<3>::new();
        let mse: Vec<f64> = (0..NEPOCHS)
            .map(|_| model.train(tr_data, tr_labels))
            .collect();
        let correct: usize = te_data
            .iter()
            .zip(te_labels.iter())
            .map(|(x, d)| model.classify(x) == *d)
            .filter(|f| *f)
            .count();

        let title = format!("Learning curve - dist: {dist};");
        plot_mse(&mse, &title);
        let error = 100.0 * (te_data.len() - correct) as f64 / te_data.len() as f64;
        let title = format!(
            "Perceptron Classification with Half-Moon Data - dist: {dist}; Error: {error:.1}%"
        );
        plot(&te_data, &model.weights, &title);

        //let p_model = |input: &[f64; 3]| model.output(input);
        //plot_mesh(&te_data, p_model, &title);
    }
}
