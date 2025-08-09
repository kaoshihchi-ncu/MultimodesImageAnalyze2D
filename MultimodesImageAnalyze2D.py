import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import jn, jvp
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
import pandas as pd
import os

# ------------- Constants -------------
PARAM_CSV = 'Parameters_MixingModes.csv'
lambda0 = 808e-9
nu = 1.453
k = 2 * np.pi / lambda0
v_sq_minus_1 = nu**2 - 1
u_11 = 2.405
u_12 = 5.52
pixel_size = 0.88e-6

# ------------- Field Functions -------------
def k_transverse(u_nm, a):
    return u_nm / a

def E_theta(n, r, theta, u_nm, a):
    k_tr = k_transverse(u_nm, a)
    kr = k_tr * r
    Jn_1 = jn(n - 1, kr)
    Jn_prime = jvp(n, kr, n=1)
    bracket = Jn_1 + 1j * u_nm**2 / (2 * n * k * a) * np.sqrt(v_sq_minus_1) * Jn_prime
    return bracket * np.cos(n * theta)

def E_r(n, r, theta, u_nm, a):
    k_tr = k_transverse(u_nm, a)
    kr = k_tr * r
    Jn_1 = jn(n - 1, kr)
    Jn = jn(n, kr)
    r_safe = np.where(r == 0, 1e-12, r)
    bracket = Jn_1 + 1j * u_nm / (2 * k * r_safe) * np.sqrt(v_sq_minus_1) * Jn
    return bracket * np.sin(n * theta)

def E_z(n, r, theta, u_nm, a):
    k_tr = k_transverse(u_nm, a)
    kr = k_tr * r
    Jn = jn(n, kr)
    return -1j * u_nm / (k * a) * Jn * np.sin(n * theta)

def E_mode_vector(n, r, theta, A, u_nm, a):
    Er = E_r(n, r, theta, u_nm, a)
    Etheta = E_theta(n, r, theta, u_nm, a)
    Ez = E_z(n, r, theta, u_nm, a)
    return A * np.stack((Er, Etheta, Ez), axis=-1)

def E_mix_vector(r, theta, phi, Amp11, Amp12, a):
    E11_vec = E_mode_vector(1, r, theta, Amp11, u_11, a)
    E12_vec = E_mode_vector(1, r, theta, Amp12, u_12, a)
    return E11_vec + E12_vec * np.exp(1j * phi)

def E_mode_magnitude_2D(r_2D, theta_2D, Amp11, Amp12, phi, a_fit):
    mask = r_2D <= a_fit
    Emix = np.zeros((*r_2D.shape, 3), dtype=complex)
    Emix[mask] = E_mix_vector(r_2D[mask], theta_2D[mask], phi, Amp11, Amp12, a_fit)
    intensity = np.sum(np.abs(Emix)**2, axis=-1)
    return intensity

def mode_power_ratio(Amp11, Amp12, a_fit):
    r_samples = np.linspace(0, a_fit, 500)
    theta = np.zeros_like(r_samples)
    E11 = E_mode_vector(1, r_samples, theta, Amp11, u_11, a_fit)
    E12 = E_mode_vector(1, r_samples, theta, Amp12, u_12, a_fit)
    I11 = np.sum(np.abs(E11)**2, axis=-1)
    I12 = np.sum(np.abs(E12)**2, axis=-1)
    P11 = trapezoid(I11 * r_samples, r_samples)
    P12 = trapezoid(I12 * r_samples, r_samples)
    return P12 / P11

# ------------- Classes & Functions -------------
class BeamImage:
    def __init__(self, filepath):
        self.filepath = filepath
        self._load_image()

    def _load_image(self):
        img = Image.open(self.filepath)
        self.data = np.array(img, dtype=np.float64)
        self.Ny, self.Nx = self.data.shape

    @staticmethod
    def cropImage(image, centroid, radius):
        x, y = int(centroid[0]), int(centroid[1])
        Ny, Nx = image.shape
        x0 = max(x - radius, 0)
        x1 = min(x + radius, Nx)
        y0 = max(y - radius, 0)
        y1 = min(y + radius, Ny)
        return image[y0:y1, x0:x1]
    
    @staticmethod
    def compute_gravity_center(image):
        total = np.sum(image)
        if total == 0:
            return np.array(image.shape[::-1]) / 2  # fallback: image center
        y, x = np.indices(image.shape)
        x_center = np.sum(x * image) / total
        y_center = np.sum(y * image) / total
        return x_center, y_center


def fit_intensity_2D(x_idx, y_idx, I_exp_2D, initial_guess):
    def residuals(params):
        Amp11, Amp12, phi, a_fit, x0, y0 = params
        x_phys = (x_idx - x0) * pixel_size
        y_phys = (y_idx - y0) * pixel_size
        r_2D = np.sqrt(x_phys**2 + y_phys**2)
        theta_2D = np.arctan2(y_phys, x_phys)
        I_model = E_mode_magnitude_2D(r_2D, theta_2D, Amp11, Amp12, phi, a_fit)
        weights = np.sqrt(I_exp_2D + 1e-6)
        return (I_model - I_exp_2D).ravel() * weights.ravel()

    bounds = (
        [float(row['Amp11_min']), float(row['Amp12_min']), float(row['phi_min']), float(row['a_min']), -np.inf, -np.inf],
        [float(row['Amp11_max']), float(row['Amp12_max']), float(row['phi_max']), float(row['a_max']), np.inf, np.inf]
    )

    result = least_squares(residuals, initial_guess, bounds=bounds)

    Amp11, Amp12, phi, a_fit, x0, y0 = result.x
    x_phys = (x_idx - x0) * pixel_size
    y_phys = (y_idx - y0) * pixel_size
    r_2D = np.sqrt(x_phys**2 + y_phys**2)
    theta_2D = np.arctan2(y_phys, x_phys)
    I_model_fit = E_mode_magnitude_2D(r_2D, theta_2D, Amp11, Amp12, phi, a_fit)

    ss_res = np.sum((I_exp_2D - I_model_fit)**2)
    ss_tot = np.sum((I_exp_2D - np.mean(I_exp_2D))**2)
    r_squared = 1 - ss_res / ss_tot
    print(f"2D Fit Iteration count: {result.nfev}, R²: {r_squared:.4f}")
    return result.x, r_squared, I_model_fit


# ------------- Main Loop -------------
param_df = pd.read_csv(PARAM_CSV)


for idx, row in param_df.iterrows():
    if int(row['enable']) != 1:
        continue

    filepath = row['file'] if 'file' in row else row['filename']
    centroid_x, centroid_y = float(row['centroid_x']), float(row['centroid_y'])
    crop_size = int(row['crop_size'])

    print(f"\nProcessing: {filepath}")
    beam = BeamImage(filepath)

    # --- Background subtraction using y=1100 row ---
    y_bg = 1100
    if y_bg < beam.data.shape[0]:
        x = np.arange(beam.data.shape[1])
        bg_row = beam.data[y_bg, :]
        coeffs = np.polyfit(x, bg_row, 1)  # Linear fit
        bg_line = np.polyval(coeffs, x)
        bg_matrix = np.tile(bg_line, (beam.data.shape[0], 1))
        data_bgsub = beam.data - bg_matrix
    else:
        print(f"Warning: y={y_bg} is out of image bounds. Skipping background subtraction.")
        data_bgsub = beam.data.copy()

    gravity_x, gravity_y = BeamImage.compute_gravity_center(data_bgsub)
    cropped_spot = BeamImage.cropImage(beam.data, [gravity_x, gravity_y], radius=crop_size//2)
    I_exp_2D = cropped_spot / np.max(cropped_spot)

    # Initial guess
    cx, cy = crop_size // 2, crop_size // 2
    initial_guess = [
        float(row['Amp11_init']),
        float(row['Amp12_init']),
        float(row['phi_init']),
        float(row['a_init']),
        cx,  # x0
        cy   # y0
    ]

    # Fitting
    y_idx, x_idx = np.indices(I_exp_2D.shape)
    fit_result, r_squared, I_fit_2D = fit_intensity_2D(x_idx, y_idx, I_exp_2D, initial_guess)
    Amp11_fit, Amp12_fit, phi_fit, a_fit, x0_fit, y0_fit = fit_result

    power_ratio = mode_power_ratio(Amp11_fit, Amp12_fit, a_fit)


    # Plotting (remove residual map, add color bars)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(I_exp_2D, cmap='gray', origin='lower')
    axs[0].set_title('Experimental Image')
    axs[0].axis('off')
    cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cbar0.set_label('Intensity')

    im1 = axs[1].imshow(I_fit_2D / np.max(I_fit_2D), cmap='hot', origin='lower')
    axs[1].set_title('Fitted Mode Intensity')
    axs[1].axis('off')
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity')

    fig.suptitle(
        f"{filepath}\n"
        f"Amp11={Amp11_fit:.2f}, Amp12={Amp12_fit:.2f}, φ={phi_fit:.2f}, "
        f"a={a_fit*1e6:.1f} µm, x0={x0_fit:.1f}, y0={y0_fit:.1f}, "
        f"R²={r_squared:.4f}, P12/P11={power_ratio:.3f}"
    )

    plt.tight_layout()
    svg_output_path = os.path.splitext(filepath)[0] + '_fit2D_result.svg'
    fig.savefig(svg_output_path, format='svg')
    plt.show()
