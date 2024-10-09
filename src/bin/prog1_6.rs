use gnuplot::{AxesCommon, Color, Figure, PointSymbol};
use nnlm::*;

fn plot_results(data: &HalfMoonData, w: &Vec<f64>, title: &str) {
    let (xmin, xmax) = data
        .x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            (f64::min(min, val), f64::max(max, val))
        });
    let (ymin, ymax) = data
        .y
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
            (f64::min(min, val), f64::max(max, val))
        });

    let mut grid_x = Vec::new();
    let mut grid_y = Vec::new();
    let mut grid_class = Vec::new();

    // Generate grid points and classify them using the perceptron weights
    let step_size = 0.2; // Adjust this for a denser grid
    let mut x = xmin;
    while x <= xmax {
        let mut y = ymin;
        while y <= ymax {
            let input = [1.0, x, y]; // input with bias
            let z = w.iter().zip(&input).map(|(wi, xi)| wi * xi).sum::<f64>();
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
    let y_vals: Vec<f64> = x_vals.iter().map(|&x| -(w[0] + w[1] * x) / w[2]).collect();

    // Plot using gnuplot
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(title, &[])
        .points(&data.x, &data.y, &[PointSymbol('O'), Color("blue")]) // Original data points
        .points(&pos_x, &pos_y, &[PointSymbol('.'), Color("green")]) // Positive classified points
        .points(&neg_x, &neg_y, &[PointSymbol('.'), Color("red")]) // Negative classified points
        .lines(&x_vals, &y_vals, &[Color("black")]);
    fg.show().unwrap();
}

fn main() {
    for dist in [-4.0, 0.0, 4.0] {
        let radius = 10.0;
        let width = 6.0;
        let data = halfmoon(radius, width, dist, 3000);
        let num_tr = 1000;
        let num_te = 2000;
        let num_epochs = 50;

        let w = perceptron_train(&data, num_tr, num_epochs);
        let error_rate = perceptron_test(&data, &w, num_te, num_tr);
        println!("Test error rate: {:.2}%", error_rate);

        let title = format!("Perceptron Classification with Half-Moon Data - dist {dist}");
        plot_results(&data, &w, title.as_str());
    }
}
