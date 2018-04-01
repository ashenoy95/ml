# Machine Learning

* __speech-denoising__: MLP based speech denoiser implemented using TensorFlow that that takes a noisy speech spectrum (speech plus chip eating noise) and then produces a cleaned-up speech spectrum.

* __em.py__: EM algorithm for a Mixture of Gaussians on one-dimensional data.

* __mnist-shallow.py__: TensorFlow implementation of MLP (1024x5) using ReLU activation, He initialization and Adam optimization giving over 98% accuracy.

* __nw_compression.ipynb__: Network compression of 'mnist-shallow' using low rank approximation of trained weights (using the top 20 singular values after SVD), thereby using only 4% of the memory of the original n/w.

* __parity.py__: MLP (4-4-1) with backprop from scratch to solve the parity problem for a 4-bit input.


