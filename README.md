# Photonic Bandgap Suite

An interactive computational suite for calculating and visualizing the optical properties, transmission spectra, and dispersion relations of 1D and 2D photonic crystals. 

This application bridges advanced theoretical physics with accessible software engineering, utilizing **Python**, **Streamlit**, and fundamental numerical methods to provide real-time solving of Maxwell's equations for periodic dielectric media.

**Live Application:** [https://photonicbandgapsuite-cbo4zhgm2tlhbbpugujkqn.streamlit.app/]

---

## 🛠️ Features
* **1D Multilayer Stacks:** Computes the analytical band structure and layer-by-layer transmittance.
* **2D Triangular Lattices:** Solves the eigenvalue problem for $E$-polarization (TM) and $H$-polarization (TE) modes using the Plane Wave Expansion Method (PWEM).
* **2D Square Lattices:** Computes mode dispersions across the $\Gamma \to X \rightarrow M \rightarrow \Gamma$ irreducible Brillouin zone.
* **Dynamic Parameter Sweeping:** Real-time adjustments of lattice constants, dielectric contrasts, hole radii, and reciprocal lattice plane wave resolutions.

---

## 🔬 Physics Core & Methodology

This suite relies on two distinct numerical frameworks to extract the optical properties of periodic structures.

### 1. Transfer Matrix Method (TMM) for 1D Crystals
For one-dimensional photonic crystals (alternating layers of high and low refractive index materials), the optical response can be solved analytically using the Transfer Matrix Method. 

The electromagnetic field propagation through a single dielectric layer is represented by a characteristic matrix $M$. For a structure with $N$ layers, the total system matrix is the product of the individual matrices:
$$ \begin{bmatrix} E_{out} \\ H_{out} \end{bmatrix} = M_1 M_2 \dots M_N \begin{bmatrix} E_{in} \\ H_{in} \end{bmatrix} $$

By analyzing this matrix, we calculate the layer-by-layer **Transmittance (T)**. Furthermore, the infinite periodic **Photonic Band Structure (PBS)** is derived from the translational invariance of the Bloch wave vector $K$, governed by the transcendental dispersion equation:
$$ \cos(K\Lambda) = \cos(k_1b)\cos(k_2a) - \frac{1}{2}\left(\frac{k_1}{k_2} + \frac{k_2}{k_1}\right)\sin(k_1b)\sin(k_2a) $$
Where $\Lambda$ is the pitch, and $k_1, k_2$ are the wave vectors in their respective media.

### 2. Plane Wave Expansion Method (PWEM) for 2D Crystals
For two-dimensional structures (Square and Triangular lattices), there is no generalized analytical solution. This engine employs PWEM to cast Maxwell's equations into a scalable eigenvalue problem.

The periodic dielectric function $\epsilon(\mathbf{r})$ is expanded into a Fourier series over the reciprocal lattice vectors $\mathbf{G}$:
$$ \frac{1}{\epsilon(\mathbf{r})} = \sum_{\mathbf{G}} F(\mathbf{G}) e^{i\mathbf{G}\cdot\mathbf{r}} $$

Applying Bloch's Theorem, the magnetic/electric fields are expanded as a sum of plane waves. For instance, the Transverse Magnetic (TM) mode wave equation:
$$ \frac{1}{\epsilon(\mathbf{r})} \nabla^2 E_z + \left(\frac{\omega}{c}\right)^2 E_z = 0 $$

Translates into the following symmetric eigenvalue matrix equation in reciprocal space:
$$ \sum_{\mathbf{G}'} |\mathbf{k}+\mathbf{G}| F(\mathbf{G}-\mathbf{G}') |\mathbf{k}+\mathbf{G}'| C(\mathbf{k}|\mathbf{G}') = \left(\frac{\omega}{c}\right)^2 C(\mathbf{k}|\mathbf{G}) $$

By constructing and diagonalizing this matrix (up to $300 \times 300$ complex matrices depending on the chosen plane-wave resolution) across the high-symmetry points of the irreducible Brillouin zone, the engine extracts the normalized eigenfrequencies $\omega a / 2\pi c$ and isolates the complete Photonic Bandgaps (PBG).

---

## 💻 Tech Stack
* **Core Logic:** `Python`, `NumPy`, `SciPy` (Matrix Diagonalization & Bessel Functions)
* **Visualization:** `Matplotlib`
* **Frontend/UI:** `Streamlit`

---

## 🚀 Local Installation & Usage

To run this application locally on your machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/Tushar-Biswas06/Photonic_bandgap_suite.git](https://github.com/Tushar-Biswas06/Photonic_bandgap_suite.git)
   cd Photonic_bandgap_suite
