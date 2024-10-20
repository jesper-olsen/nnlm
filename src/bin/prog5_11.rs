use gnuplot::{AxesCommon, Caption, Color, Figure, LineStyle, LineWidth, Solid};
use na::DMatrix;
use nalgebra as na;
use nnlm::*;
use std::fmt;
use stmc_rs::marsaglia::Marsaglia;

fn clip_gradient(grad: f64, max_grad: f64) -> f64 {
    if grad > max_grad {
        max_grad
    } else if grad < -max_grad {
        -max_grad
    } else {
        grad
    }
}

#[derive(Copy, Clone)]
struct GKernal<const IDIM: usize> {
    mean: [f64; IDIM],
    var: [f64; IDIM],
}

impl<const IDIM: usize> fmt::Display for GKernal<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GKernal")?;
        write!(f, "mean: ")?;
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.mean[j])?;
        }
        writeln!(f)?;
        write!(f, "var:  ")?;
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.var[j])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<const IDIM: usize> GKernal<IDIM> {
    pub fn new() -> Self {
        Self {
            mean: [0.0; IDIM],
            var: [1.0; IDIM],
        }
    }

    pub fn estimate<'a, I>(&mut self, data: I)
    where
        I: IntoIterator<Item = &'a [f64; IDIM]> + Clone,
    {
        // Estimate mean and variance - Welford's method
        let mut count = 0;
        self.mean.fill(0.0);
        self.var.fill(0.0);
        let mut old_mean = [0.0f64; IDIM];
        for (i, item) in data.into_iter().enumerate() {
            let v = item.as_ref();
            count += 1;
            old_mean.copy_from_slice(&self.mean);
            self.mean
                .iter_mut()
                .zip(v.iter())
                .for_each(|(m, x)| *m += (*x - *m) / (i + 1) as f64);
            for j in 0..IDIM {
                self.var[j] += (v[j] - self.mean[j]) * (v[j] - old_mean[j]);
            }
        }
        self.var.iter_mut().for_each(|e| *e /= (count - 1) as f64);
    }

    /// Squared Euclidian distance between x and the Gaussian kernel.
    pub fn dist_euc(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .map(|(&m, &xi)| (xi - m).powi(2))
            .sum()
    }

    /// Squared Mahalanobis distance between x and the Gaussian kernel.
    pub fn dist(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .zip(self.var.iter())
            .map(|((&m, &xi), &v)| (xi - m).powi(2) / v)
            .sum()
    }

    /// Log probability of x
    pub fn logp(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        let s: f64 = self
            .var
            .iter()
            .map(|v| (2.0 * std::f64::consts::PI * v).ln())
            .sum();
        -0.5 * (self.dist(x) + s)
    }

    /// Probability of x
    pub fn p(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        let a: f64 = (2.0 * std::f64::consts::PI).powi(IDIM as i32);
        let vproduct = self.var.iter().fold(a, |acc, v| acc * v);
        (1.0 / vproduct.sqrt()) * (-0.5 * self.dist(x)).exp()
    }

    pub fn floor_variances(&mut self, floor: f64) {
        self.var.iter_mut().for_each(|v| *v = v.max(floor));
    }

    pub fn set_variances(&mut self, val: f64) {
        self.var.iter_mut().for_each(|v| *v = val);
    }
}

pub struct RBF<const IDIM: usize, const NKERNELS: usize> {
    kernels: [GKernal<IDIM>; NKERNELS],
    weights: [f64; NKERNELS],
}

impl<const IDIM: usize, const NKERNELS: usize> fmt::Display for RBF<IDIM, NKERNELS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RBF")?;

        write!(f, "Weights: ")?;
        for j in 0..NKERNELS {
            write!(f, "{:>8.2} ", self.weights[j])?;
        }
        writeln!(f)?;

        for i in 0..NKERNELS {
            write!(f, "{i}: {}", self.kernels[i])?;
        }
        Ok(())
    }
}

fn rbf(dist: f64) {
    let central_radius = 10.0;
    let radius_variation = 6.0;
    let (data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
    let mut model = RBF::<2, 20>::new();

    const MAX_ITER: usize = 100;
    model.train_kernels(&data[..1000], MAX_ITER);
    let pdata: Vec<(f64, f64, f64)> = data
        .iter()
        .zip(labels.iter())
        .map(|(a, l)| (a[0], a[1], *l as f64))
        .collect();
    let centers: Vec<(f64, f64)> = model
        .kernels
        .iter()
        .map(|k| (k.mean[0], k.mean[1]))
        .collect();
    let vars: Vec<(f64, f64)> = model.kernels.iter().map(|k| (k.var[0], k.var[1])).collect();
    plot_rbf(&pdata, &centers, &vars);

    //let mse = model.train_weights_lms(&data[..1000], &labels[..1000], MAX_ITER);
    let mse = model.train_weights_rhs(&data[..1000], &labels[..1000], MAX_ITER);
    let title = format!("Training RBF network for dist: {dist}");
    plot_mse(&mse, &title);
    println!("{model}");
    model.eval(&data[..1000], &labels[..1000], "Training data:");
    model.eval(&data[1000..], &labels[1000..], "Test data:");
}

impl<const IDIM: usize, const NKERNELS: usize> RBF<IDIM, NKERNELS> {
    pub fn new() -> Self {
        let mut rng = Marsaglia::new(12, 34, 56, 78);
        let mut weights = [0.0f64; NKERNELS];
        weights.iter_mut().for_each(|w| *w = 0.5 * rng.uni() - 0.25);
        Self {
            kernels: [GKernal::new(); NKERNELS],
            weights,
        }
    }

    pub fn output(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM, "Input length does not match expected IDIM");
        self.kernels
            .iter()
            .zip(self.weights.iter())
            .map(|(&k, &w)| w * k.p(x))
            .sum::<f64>()
            .signum()
    }

    fn eval(&self, data: &[[f64; IDIM]], labels: &[i8], title: &str) {
        let mut errors = 0;
        for (x, label) in data.iter().zip(labels) {
            let y = self.output(x) as i8;
            let d = label.signum();
            if d != y {
                errors += 1
            }
        }
        let errp = 100.0 * errors as f64 / data.len() as f64;
        println!("{title} errors: {errors}/{} = {errp:>6.2}%", data.len());
    }

    fn train_weights_rhs(
        &mut self,
        data: &[[f64; IDIM]],
        labels: &[i8],
        max_iter: usize,
    ) -> Vec<f64> {
        let sig = 0.01;
        let lambda = 1.0;
        let mut p = DMatrix::identity(NKERNELS, NKERNELS) / sig;

        let mut lmse = Vec::<f64>::new();
        for _ep in 0..max_iter {
            let mut mse = 0.0;
            for (_, (x, label)) in data.iter().zip(labels).enumerate() {
                let d = *label as f64; // 1 or -1
                let k: Vec<f64> = self.kernels.iter().map(|&k| k.p(x)).collect();

                let k_matrix = DMatrix::from_vec(k.len(), 1, k.clone());

                // Calculate pai and kk
                let pai = &p * &k_matrix;
                let g_t_pai = k
                    .iter()
                    .zip(pai.iter())
                    .map(|(gi, pi)| gi * pi)
                    .sum::<f64>();
                let kk = pai / (lambda + g_t_pai);

                let e = d - self
                    .weights
                    .iter()
                    .zip(&k)
                    .map(|(wi, gi)| wi * gi)
                    .sum::<f64>();
                let w_delta: Vec<f64> = kk.iter().map(|&kki| kki * e).collect();
                mse += e * e;

                for (wi, wd) in self.weights.iter_mut().zip(w_delta.iter()) {
                    *wi += wd;
                }

                let g_t_p = k_matrix.transpose() * &p;
                let kk_g_t_p = kk * g_t_p;
                p = (&p - kk_g_t_p) / lambda;
            }
            mse /= data.len() as f64;
            lmse.push(mse);
        }
        lmse
    }

    fn train_weights_lms(
        &mut self,
        data: &[[f64; IDIM]],
        labels: &[i8],
        max_iter: usize,
    ) -> Vec<f64> {
        let mut lmse = Vec::<f64>::new();
        for ep in 0..max_iter {
            let lr = 0.1 / (1.0 + ep as f64);
            let mut mse = 0.0;
            for (_, (x, label)) in data.iter().zip(labels).enumerate() {
                let k: Vec<f64> = self.kernels.iter().map(|&k| k.p(x)).collect();
                let d = *label as f64;
                let y: f64 = k
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(g, w)| g * w)
                    .sum::<f64>();
                let e = d - y;

                mse += e * e;
                self.weights
                    .iter_mut()
                    .zip(k.iter())
                    .map(|(w, ki)| (w, lr * e * ki))
                    .for_each(|(w, grad)| *w += grad);
                self.normalise_weights();
            }
            mse /= data.len() as f64;
            lmse.push(mse);
            //println!("ep {ep} {mse}");
        }
        lmse
    }

    fn normalise_weights(&mut self) {
        let nw = self.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        if nw > 0.0 {
            self.weights.iter_mut().for_each(|w| *w /= nw);
        }
    }

    fn train_kernels(&mut self, data: &[[f64; IDIM]], max_iter: usize) {
        const EPSILON: f64 = 0.0;
        assert!(
            data.len() >= NKERNELS,
            "Not enough data samples to initialize centroids."
        );

        // Initialize kernels from the first NKERNELS data samples (assume randomised)
        for (k, sample) in self.kernels.iter_mut().zip(data.iter().take(NKERNELS)) {
            k.mean.copy_from_slice(sample);
            k.var.fill(1.0);
        }

        let mut new_kernel = [GKernal::new(); NKERNELS];
        let mut sample2cluster: Vec<usize> = Vec::with_capacity(data.len());

        for ep in 0..max_iter {
            sample2cluster.clear();
            let mut gdist = 0.0;

            // assign samples to their nearest cluster
            let mut cluster_counts: [usize; NKERNELS] = [0; NKERNELS];
            for x in data {
                let (min_dist, min_pos) = self.nearest_kernel(x);
                sample2cluster.push(min_pos);
                cluster_counts[min_pos] += 1;
                gdist += min_dist;
            }
            gdist /= data.len() as f64;

            // re-estimate kernels
            for c in 0..NKERNELS {
                if cluster_counts[c] < 5 {
                    println!(
                        "Warning: kernel {c} only has {} samples - not updating",
                        cluster_counts[c]
                    );
                } else {
                    let dta: Vec<&[f64; IDIM]> = data
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| sample2cluster[*i] == c)
                        .map(|(_, v)| v)
                        .collect();
                    new_kernel[c].estimate(dta);
                }
            }

            // check for convergence - distance betwen old and new kernel estimates
            let cdist: f64 = (0..NKERNELS)
                .map(|c| self.kernels[c].dist(&new_kernel[c].mean))
                .sum();
            println!("Kmeans ep: {ep}; gdist: {gdist:>5.2}; cdist: {cdist:>5.2}");

            std::mem::swap(&mut self.kernels, &mut new_kernel);

            if cdist <= EPSILON {
                break;
            }
        }
    }

    fn nearest_kernel(&self, x: &[f64; IDIM]) -> (f64, usize) {
        self.kernels
            .iter()
            .enumerate()
            .map(|(i, k)| (k.dist(x), i))
            .fold((f64::INFINITY, 0), |(min_dist, min_pos), (dist, i)| {
                if dist < min_dist {
                    (dist, i)
                } else {
                    (min_dist, min_pos)
                }
            })
    }
}

fn main() {
    for dist in [-5.0] {
        rbf(dist);
    }
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
