use gnuplot::{AxesCommon, Color, Figure, PointSymbol};
use nnlm::*;

fn plot(data: &[[f64; 3]], model: &Perceptron<3>, title: &str) {
    let (xmin, xmax) = data
        .iter()
        .map(|&v| v[1])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (f64::min(min, val), f64::max(max, val))
        });
    let (ymin, ymax) = data
        .iter()
        .map(|&v| v[2])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (f64::min(min, val), f64::max(max, val))
        });

    let mut grid_x = Vec::new();
    let mut grid_y = Vec::new();
    let mut grid_class = Vec::new();

    // Grid points - different colours and different sides of decision boundary
    let step_size = 0.2; // grid density 
    let mut x = xmin;
    while x <= xmax {
        let mut y = ymin;
        while y <= ymax {
            let input = [1.0, x, y]; // input with bias
            let z = model.classify(&input) as f64;
            grid_x.push(x);
            grid_y.push(y);
            grid_class.push(sign(z)); // predicted class
            y += step_size
        }
        x += step_size
    }

    let mut pos_x = Vec::new();
    let mut pos_y = Vec::new();
    let mut neg_x = Vec::new();
    let mut neg_y = Vec::new();

    for i in 0..grid_x.len() {
        if grid_class[i] == 1 {
            pos_x.push(grid_x[i]);
            pos_y.push(grid_y[i]);
        } else {
            neg_x.push(grid_x[i]);
            neg_y.push(grid_y[i]);
        }
    }

    // calcualte decision boundary
    let x_vals: Vec<f64> = (0..=100)
        .map(|i| xmin + (xmax - xmin) * (i as f64 / 100.0))
        .collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| -(model.weights[0] + model.weights[1] * x) / model.weights[2])
        .collect();

    let vx: Vec<f64> = data.iter().map(|v| v[1]).collect();
    let vy: Vec<f64> = data.iter().map(|v| v[2]).collect();
    // Plot using gnuplot
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(title, &[])
        .points(&vx, &vy, &[PointSymbol('O'), Color("blue")]) // Original data points
        .points(&pos_x, &pos_y, &[PointSymbol('.'), Color("green")]) // Positive classified points
        .points(&neg_x, &neg_y, &[PointSymbol('.'), Color("red")]) // Negative classified points
        .lines(&x_vals, &y_vals, &[Color("black")]);
    fg.show().unwrap();
}

fn plot_mse(mse: &[f64]) {
    let epochs: Vec<i32> = (0..mse.len()).map(|i| i as i32).collect();
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title("Learning Curve", &[])
        .set_x_label("Epoch", &[])
        .set_y_label("MSE", &[])
        .lines(
            &epochs,
            mse,
            &[gnuplot::Caption("MSE"), gnuplot::Color("black")],
        );
    fg.show().unwrap();
}

fn main() {
    for dist in [-4.0, 0.0, 4.0] {
        const NEPOCHS: usize = 50;
        let central_radius = 10.0;
        let radius_variation = 6.0;
        let (data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
        let tr_data = &data[..1000];
        let tr_labels = &labels[..1000];
        let te_data = &data[1000..];
        let te_labels = &labels[1000..];
        let mut model = Perceptron::<3>::new();
        let mse: Vec<f64> = (0..NEPOCHS).map(|_| model.train(tr_data, tr_labels)).collect();
        let correct: usize = te_data
            .iter()
            .zip(te_labels.iter())
            .map(|(x, d)| model.classify(x) == *d)
            .filter(|f| *f)
            .count();

        println!("{mse:?}");
        println!("Correct: {correct}/{}", te_data.len());
        plot_mse(&mse);
        let title = format!("Perceptron Classification with Half-Moon Data - dist {dist}");
        plot(&te_data, &model, &title);
    }
}
