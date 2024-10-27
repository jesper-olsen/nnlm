use clap::Parser;
use gnuplot::{AxesCommon, Caption, Color, Figure, LineStyle, LineWidth, Solid};
use nnlm::rbf::RBF;
use nnlm::{halfmoons, plot_mse};
use stmc_rs::marsaglia::Marsaglia;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long = "lms", default_value_t = false)]
    /// Weight training: Least Mean Squares (default: Recursive Least Squares)
    l: bool,
    #[arg(short, long = "kmeans", default_value_t = false)]
    /// Kernel training: k-means (default: Expectation Maximisation)
    k: bool,
    #[arg(short, long="dist", default_value_t = -5.0)]
    ///distance between halfmoons (e.g. -5.0 to 5.0
    d: f64,
    #[arg(short, long = "nkernels", default_value_t = 20)]
    ///number of RBF kernels
    n: usize,
}

fn main() {
    let args = Args::parse();
    let dist = args.d;
    let central_radius = 10.0;
    let radius_variation = 6.0;
    let (trdata, trlabels) = halfmoons::<3000>(central_radius, radius_variation, dist);
    let (tedata, telabels) = halfmoons::<2000>(central_radius, radius_variation, dist);
    let mut model = RBF::<2>::new(args.n);

    const MAX_ITER: usize = 100;
    let mut rng = Marsaglia::new(12, 34, 56, 78);
    if args.k {
        model.train_kernels_kmeans(&mut rng, &trdata, MAX_ITER);
    } else {
        model.train_kernels_em(&mut rng, &trdata, MAX_ITER);
    }
    let pdata: Vec<(f64, f64, f64)> = trdata
        .iter()
        .zip(trlabels.iter())
        .map(|(a, l)| (a[0], a[1], *l as f64))
        .collect();
    let centers: Vec<(f64, f64)> = model
        .kernels
        .iter()
        .map(|k| (k.mean[0], k.mean[1]))
        .collect();
    let vars: Vec<(f64, f64)> = model.kernels.iter().map(|k| (k.var[0], k.var[1])).collect();
    plot_rbf(&pdata, &centers, &vars);

    //model.weights.iter_mut().for_each(|w| *w = 0.5 * rng.uni() - 0.25);
    let mse = if args.l {
        model.train_weights_lms(&trdata, &trlabels, MAX_ITER)
    } else {
        model.train_weights_rhs(&trdata, &trlabels, MAX_ITER)
    };
    let title = format!("Training RBF network for dist: {dist}");
    plot_mse(&mse, &title);
    println!("{model}");
    model.eval(&trdata, &trlabels, "Errors - Training data:");
    model.eval(&tedata, &telabels, "Errors - Test data:");
}

fn plot_rbf(
    halfmoon_data: &[(f64, f64, f64)], // (x, y, label)
    centers: &[(f64, f64)],            // Kernel means
    variances: &[(f64, f64)],          // Kernel variances
) {
    let mut fg = Figure::new();

    let (x_pos, y_pos): (Vec<f64>, Vec<f64>) = halfmoon_data
        .iter()
        .filter(|(_, _, label)| *label > 0.0)
        .map(|(x, y, _)| (*x, *y))
        .unzip();

    let (x_neg, y_neg): (Vec<f64>, Vec<f64>) = halfmoon_data
        .iter()
        .filter(|(_, _, label)| *label < 0.0)
        .map(|(x, y, _)| (*x, *y))
        .unzip();

    let (x_c, y_c): (Vec<f64>, Vec<f64>) = centers.iter().cloned().unzip();

    // Plot the data points
    let axes = fg
        .axes2d()
        .points(
            &x_pos,
            &y_pos,
            &[Caption("Class 1"), Color("red"), gnuplot::PointSize(2.0)],
        )
        .points(
            &x_neg,
            &y_neg,
            &[Caption("Class -1"), Color("green"), gnuplot::PointSize(2.0)],
        )
        .points(
            &x_c,
            &y_c,
            &[
                Caption("RBF Centers"),
                Color("black"),
                gnuplot::PointSize(4.0),
            ],
        )
        .set_title("Data with RBF kernels indicated", &[])
        .set_x_label("x", &[])
        .set_y_label("y", &[]);

    // Plot ellipsoids around the RBF centers using the variances as axes
    for ((x, y), (x_var, y_var)) in centers.iter().zip(variances.iter()) {
        let ellipse_points: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
                let ellipse_x = x + (x_var.sqrt() * angle.cos());
                let ellipse_y = y + (y_var.sqrt() * angle.sin());
                (ellipse_x, ellipse_y)
            })
            .collect();

        let (ellipse_x, ellipse_y): (Vec<f64>, Vec<f64>) = ellipse_points.into_iter().unzip();

        axes.lines(
            &ellipse_x,
            &ellipse_y,
            &[Color("black"), LineWidth(1.0), LineStyle(Solid)],
        );
    }

    fg.show().unwrap();
}
