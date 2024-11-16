use nnlm::perceptron::Perceptron;
use nnlm::*;
use stmc_rs::marsaglia::Marsaglia;

const IDIM: usize = 3;

fn dact(x: f64) -> f64 {
    //drelu(x)   // ReLU
    //1.0 - x.tanh().powi(2)  // tanh(x)
    2.0 - 2.0 * (2.0 * x).tanh().powi(2) // tanh(2x)
}

fn act(x: f64) -> f64 {
    //relu(x)        // does not work well on halfmoons
    //x.tanh()
    (2.0 * x).tanh()
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
fn drelu(x: f64) -> f64 {
    x.max(0.0)
}

fn mlp_output(hlayer: &[Perceptron<IDIM>], olayer: &Perceptron<21>, x: &[f64; IDIM]) -> f64 {
    let mut hd = vec![0.0; hlayer.len() + 1];
    hd[hlayer.len()] = 1.0; // bias
    for i in 0..hlayer.len() {
        hd[i] = act(hlayer[i].output(x));
    }
    act(olayer.output(&hd))
}

fn lr(ep: usize) -> f64 {
    1e-0 / (1000.0 + ep as f64)
    //1e-3
}

fn mlp(dist: f64) {
    let central_radius = 10.0;
    let radius_variation = 6.0;
    let (mut data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
    const NTRAIN: usize = 1000;

    let mean = nnlm::calc_mean(&data[..NTRAIN]);
    normalise_mean(&mut data[..NTRAIN], &mean);
    normalise_mean(&mut data[NTRAIN..], &mean);

    let max_vals = calc_max(&data[..NTRAIN]);
    normalise_max(&mut data[..NTRAIN], &max_vals);
    normalise_max(&mut data[NTRAIN..], &max_vals);

    let data: Vec<_> = data
        .into_iter()
        .map(|[x, y]| [1.0, x, y]) // add bias
        .collect();

    let mut rng = Marsaglia::new(12, 34, 56, 78);
    let mut hlayer = Vec::new();
    for _ in 0..20 {
        let mut p = Perceptron::<IDIM>::new();
        let norm = (2.0 / (1 + p.weights.len()) as f64).sqrt();
        p.weights
            .iter_mut()
            //.for_each(|w| *w = rng.uni() / 2.0 - 0.25);
            .for_each(|w| *w = rng.gauss() * norm);
        hlayer.push(p);
    }
    let mut olayer = Perceptron::<21>::new();
    let norm = (2.0 / (21.0f64)).sqrt();
    olayer
        .weights
        .iter_mut()
        //.for_each(|w| *w = rng.uni() / 2.0 - 0.25);
        .for_each(|w| *w = rng.gauss() * norm);

    let mse_thres = 1e-3;
    let mut err = Vec::new();

    const MAX_EPOCHS: usize = 50;
    for ep in 0..MAX_EPOCHS {
        let mut hd = vec![0.0; hlayer.len() + 1];
        hd[hlayer.len()] = 1.0; // bias
        let mut mse = 0.0;
        for (i, x) in data[..NTRAIN].iter().enumerate() {
            // Forward pass
            for z in 0..hlayer.len() {
                hd[z] = act(hlayer[z].output(x));
            }
            let o = act(olayer.output(&hd));
            let e = o - labels[i] as f64;
            mse += e * e;

            // Backpropagation
            let delta_o = e * dact(o);
            let mut delta_h = vec![0.0; hlayer.len()];
            for j in 0..delta_h.len() {
                delta_h[j] = olayer.weights[j] * delta_o * dact(hd[j]);
            }

            // Update weights
            for j in 0..olayer.weights.len() {
                olayer.weights[j] -= lr(ep) * delta_o * hd[j];
            }
            for j in 0..hlayer.len() {
                for k in 0..hlayer[j].weights.len() {
                    hlayer[j].weights[k] -= lr(ep) * delta_h[j] * x[k];
                }
            }
            // TODO: momentum
        }
        mse = mse / NTRAIN as f64;
        err.push(mse);
        println!("ep: {ep}; mse: {mse}");
        if mse < mse_thres {
            break;
        };
    }

    let correct: usize = data[NTRAIN..]
        .iter()
        .zip(labels[NTRAIN..].iter())
        .map(|(x, d)| *d == mlp_output(&hlayer, &olayer, x).signum() as i8)
        .filter(|f| *f)
        .count();

    let ntest = data.len() - NTRAIN;
    let errors = ntest - correct;
    let er = 100.0 * (errors as f64 / ntest as f64);
    println!("Errors: {correct}/{ntest} = {er:.2}%");

    let title = format!("Learning curve - dist: {dist};");
    plot_mse(&err, &title);
    let title = format!("MLP Classification with Half-Moon Data - dist: {dist}; Error: {er:.1}%");
    let mlp_model = |input: &[f64; 3]| mlp_output(&hlayer, &olayer, input);
    plot_mesh(&data[NTRAIN..], mlp_model, &title);
}

fn main() {
    //let dist = 0.0;
    let dist = -4.0;
    mlp(dist);
}