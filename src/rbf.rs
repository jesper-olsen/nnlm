use log::{info, warn};
use na::DMatrix;
use nalgebra as na;
use std::fmt;
use crate::gmm::GMM;
use stmc_rs::marsaglia::Marsaglia;

//fn clip_gradient(grad: f64, max_grad: f64) -> f64 {
//    if grad > max_grad {
//        max_grad
//    } else if grad < -max_grad {
//        -max_grad
//    } else {
//        grad
//    }
//}

pub struct RBF<const IDIM: usize> {
    pub gmm: GMM<IDIM>,
}

impl<const IDIM: usize> fmt::Display for RBF<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RBF")?;

        write!(f, "Weights: ")?;
        for w in &self.gmm.weights {
            write!(f, "{:>8.2} ", w)?;
        }
        writeln!(f)?;

        for (i, k) in self.gmm.kernels.iter().enumerate() {
            write!(f, "{i}: {}", k)?;
        }
        Ok(())
    }
}


impl<const IDIM: usize> RBF<IDIM> {
    pub fn new(nkernels: usize) -> Self {
        Self {
            gmm: GMM::new(nkernels)
        }
    }

    pub fn output(&self, x: &[f64; IDIM]) -> f64 {
        self.gmm.output(x)
    }

    pub fn eval(&self, data: &[[f64; IDIM]], labels: &[i8], title: &str) {
        let errors: usize = data
            .iter()
            .zip(labels)
            .filter(|(x, label)| label.signum() != self.output(x).signum() as i8)
            .count();
        let errp = 100.0 * errors as f64 / data.len() as f64;
        info!("{title}: {errors}/{} = {errp:>6.2}%", data.len());
    }

    pub fn train_weights_rhs(
        &mut self,
        data: &[[f64; IDIM]],
        labels: &[i8],
        max_iter: usize,
    ) -> Vec<f64> {
        let sig = 0.01;
        let lambda = 1.0;
        let nkernels = self.gmm.kernels.len();
        let mut p = DMatrix::identity(nkernels, nkernels) / sig;

        let mut lmse = Vec::<f64>::new();
        for _ep in 0..max_iter {
            let mut mse = 0.0;
            for (_, (x, label)) in data.iter().zip(labels).enumerate() {
                let d = *label as f64; // 1 or -1
                let k: Vec<f64> = self.gmm.kernels.iter().map(|&k| k.p(x)).collect();

                let k_matrix = DMatrix::from_vec(k.len(), 1, k.clone());

                let pai = &p * &k_matrix;
                let g_t_pai = k
                    .iter()
                    .zip(pai.iter())
                    .map(|(gi, pi)| gi * pi)
                    .sum::<f64>();
                let kk = pai / (lambda + g_t_pai);

                let e = d - self.gmm
                    .weights
                    .iter()
                    .zip(&k)
                    .map(|(wi, gi)| wi * gi)
                    .sum::<f64>();
                let w_delta: Vec<f64> = kk.iter().map(|&kki| kki * e).collect();
                mse += e * e;

                for (wi, wd) in self.gmm.weights.iter_mut().zip(w_delta.iter()) {
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

    pub fn train_weights_lms(
        &mut self,
        data: &[[f64; IDIM]],
        labels: &[i8],
        max_iter: usize,
    ) -> Vec<f64> {
        let mut lmse = Vec::<f64>::new();
        for ep in 0..max_iter {
            let lr = 0.1 / (1.0 + ep as f64);
            let mut mse = 0.0;
            for (x, label) in data.iter().zip(labels) {
                let k: Vec<f64> = self.gmm.kernels.iter().map(|&k| k.p(x)).collect();
                let d = *label as f64;
                let y: f64 = k
                    .iter()
                    .zip(self.gmm.weights.iter())
                    .map(|(g, w)| g * w)
                    .sum::<f64>();
                let e = d - y;

                mse += e * e;
                self.gmm.weights
                    .iter_mut()
                    .zip(k.iter())
                    .map(|(w, ki)| (w, lr * e * ki))
                    .for_each(|(w, grad)| *w += grad);
            }
            self.normalise_weights();
            mse /= data.len() as f64;
            lmse.push(mse);
            //println!("ep {ep} {mse}");
        }
        lmse
    }

    fn normalise_weights(&mut self) {
        let nw = self.gmm.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        if nw > 0.0 {
            self.gmm.weights.iter_mut().for_each(|w| *w /= nw);
        }
    }
}
