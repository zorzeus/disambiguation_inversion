import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from scipy.ndimage import zoom
from astropy.io import fits

def generate_flower_sunspot(x, y, params, surface='lower'):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    p = params[surface]
    umbra = p['B0'] * np.exp(-r**2 / p['a0']**2)
    petal1 = p['B1'] * (r**2 / p['a1']**2) * np.exp(-r**2 / p['a1']**2) * (1 + np.cos(p['n1'] * theta + p['phi1']))
    petal2 = p['B2'] * (r**2 / p['a2']**2) * np.exp(-r**2 / p['a2']**2) * (1 + np.cos(p['n2'] * theta + p['phi2']))
    return umbra + petal1 + petal2

def compute_potential_field_fourier(Bz_lower, Bz_upper, z_gap, dx, dy):
    nx, ny = Bz_lower.shape
    bz_lower_ft = fftn(Bz_lower)
    bz_upper_ft = fftn(Bz_upper)
    kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    ky = 2*np.pi * np.fft.fftfreq(ny, d=dy)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0,0] = 1e-10
    sinh_kz = np.sinh(k * z_gap)
    cosh_kz = np.cosh(k * z_gap)
    A = (bz_lower_ft * cosh_kz - bz_upper_ft) / sinh_kz
    Bx_ft = -1j * kx * A / k
    By_ft = -1j * ky * A / k
    Bx_ft[0,0] = 0
    By_ft[0,0] = 0
    Bx = np.real(ifftn(Bx_ft))
    By = np.real(ifftn(By_ft))
    return Bx, By

def apply_polar_projection(Bx, By, Bz, rotation_angle_deg):
    theta = np.radians(rotation_angle_deg)
    Bx_proj = Bx.copy()
    By_proj = By * np.cos(theta) - Bz * np.sin(theta)
    Bz_proj = By * np.sin(theta) + Bz * np.cos(theta)
    return Bx_proj, By_proj, Bz_proj

def apply_foreshortening(Bx, By, Bz, rotation_angle_deg):

    theta_rad = np.radians(rotation_angle_deg)
    
    Bx_r = Bx.copy()
    By_r = By * np.cos(theta_rad) - Bz * np.sin(theta_rad)
    Bz_r = By * np.sin(theta_rad) + Bz * np.cos(theta_rad)
    zf = np.round(np.cos(theta_rad), 4)

    if zf <= 1e-4:
        zf = 1e-4
  
    Bx_f = zoom(Bx_r, zoom=(zf, 1), order=1)
    By_f = zoom(By_r, zoom=(zf, 1), order=1)
    Bz_f = zoom(Bz_r, zoom=(zf, 1), order=1)
    
    return Bx_f, By_f, Bz_f

def create_base_flower_configuration(nx=512, ny=512, pixel_size=0.06):
    x = (np.arange(nx) - nx/2) * pixel_size
    y = (np.arange(ny) - ny/2) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')
    Bz_lower = np.zeros((nx, ny))
    Bz_upper = np.zeros((nx, ny))
    
    central_spot_params = {
        'lower': {'B0': -2500, 'a0': 3.0, 'B1': -800, 'a1': 4.5, 'n1': 6, 
                  'phi1': 0, 'B2': -400, 'a2': 6.0, 'n2': 12, 'phi2': np.pi/6},
        'upper': {'B0': -2500, 'a0': 3.0, 'B1': -800, 'a1': 4.5, 'n1': 6,
                  'phi1': np.pi/4, 'B2': -400, 'a2': 6.0, 'n2': 12, 'phi2': np.pi/3}}
    positive_spot_params = {
        'lower': {'B0': 2000, 'a0': 2.0, 'B1': 600, 'a1': 3.0, 'n1': 8,
                  'phi1': 0, 'B2': 300, 'a2': 4.0, 'n2': 16, 'phi2': 0},
        'upper': {'B0': 2000, 'a0': 2.0, 'B1': 600, 'a1': 3.0, 'n1': 8,
                  'phi1': np.pi/5, 'B2': 300, 'a2': 4.0, 'n2': 16, 'phi2': np.pi/4}}
    small_negative_params = {
        'lower': {'B0': -1500, 'a0': 1.5, 'B1': -400, 'a1': 2.0, 'n1': 4,
                  'phi1': 0, 'B2': -200, 'a2': 2.5, 'n2': 8, 'phi2': 0},
        'upper': {'B0': -1500, 'a0': 1.5, 'B1': -400, 'a1': 2.0, 'n1': 4,
                  'phi1': np.pi/3, 'B2': -200, 'a2': 2.5, 'n2': 8, 'phi2': np.pi/6}}
    upper_positive_params = {
        'lower': {'B0': 1800, 'a0': 1.8, 'B1': 500, 'a1': 2.5, 'n1': 6,
                  'phi1': 0, 'B2': 250, 'a2': 3.0, 'n2': 12, 'phi2': 0},
        'upper': {'B0': 1800, 'a0': 1.8, 'B1': 500, 'a1': 2.5, 'n1': 6,
                  'phi1': np.pi/6, 'B2': 250, 'a2': 3.0, 'n2': 12, 'phi2': np.pi/4}}

    spots = [
        {'params': central_spot_params, 'center': (-8, 8)},
        {'params': positive_spot_params, 'center': (6, -6)},
        {'params': small_negative_params, 'center': (8, 2)},
        {'params': upper_positive_params, 'center': (-2, -8)}]

    for spot in spots:
        xc, yc = spot['center']
        X_shifted = X - xc
        Y_shifted = Y - yc
        Bz_lower += generate_flower_sunspot(X_shifted, Y_shifted, spot['params'], 'lower')
        Bz_upper += generate_flower_sunspot(X_shifted, Y_shifted, spot['params'], 'upper')

    np.random.seed(42) 
    for i in range(50):
        px = np.random.uniform(-6, 6)
        py = np.random.uniform(-6, 6)
        strength = np.random.choice([-500, 500]) * np.random.uniform(0.5, 1.5)
        size = np.random.uniform(0.3, 0.8)
        X_plage = X - px
        Y_plage = Y - py
        plage_contribution = strength * np.exp(-(X_plage**2 + Y_plage**2) / size**2)
        Bz_lower += plage_contribution
        Bz_upper += plage_contribution * np.random.uniform(0.8, 1.2)

    Bz_lower -= np.mean(Bz_lower)
    Bz_upper -= np.mean(Bz_upper)
    
    z_gap = 5 * pixel_size  
    Bx, By = compute_potential_field_fourier(Bz_lower, Bz_upper, z_gap, pixel_size, pixel_size)
    
    return Bx, By, Bz_lower, X, Y, spots, x, y

def get_polar_projection_with_foreshortening(Bx_base, By_base, Bz_base, rotation_angle, apply_foreshortening_effect=True):

    if apply_foreshortening_effect:
        return apply_foreshortening(Bx_base, By_base, Bz_base, rotation_angle)
    else:
        return apply_polar_projection(Bx_base, By_base, Bz_base, rotation_angle)

def plot_flower_field(Bx, By, Bz, title='', fontsize=14):
    azimuth = np.arctan2(By, Bx) * 180 / np.pi
    azimuth = (azimuth + 360) % 360
    B_total = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_tr = np.sqrt(Bx**2 + By**2)

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(231)
    im1 = plt.imshow(Bx, origin='lower', cmap='bwr',  vmin=-500, vmax=500)
    plt.title(r'$B_x\,\rm [G]$')
    plt.colorbar(im1)

    plt.subplot(232)
    im2 = plt.imshow(By, origin='lower', cmap='bwr', vmin=-1000, vmax=1000)
    plt.title(r'$B_y\,\rm [G]$')
    plt.colorbar(im2)
    
    plt.subplot(233)
    im3 = plt.imshow(Bz, origin='lower', cmap='bwr', vmin=-700, vmax=700)
    plt.title(r'$B_z\,\rm [G]$')
    plt.colorbar(im3)

    plt.subplot(234)
    im4 = plt.imshow(B_total, origin='lower', cmap='plasma')
    plt.title(r'$|B|\,\rm [G]$')
    plt.colorbar(im4)

    plt.subplot(235)
    im5 = plt.imshow(B_tr, origin='lower', cmap='plasma')
    plt.colorbar(im5)
    plt.title(r'$B_{\perp}\,\rm [G]$')

    plt.subplot(236)
    im6 = plt.imshow(azimuth, origin='lower', cmap='twilight')
    plt.title(r'$\varphi\,[^{\circ}]$')
    plt.colorbar(im6)

#   plt.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating base magnetic field configuration...")
    Bx_base, By_base, Bz_base, X, Y, spots, x_coord, y_coord = create_base_flower_configuration(
        nx=512, ny=512, pixel_size=0.06)
    
    angles = [0]
    hdus = [fits.PrimaryHDU()]
    all_results = []
    all_results_no_foreshortening = []

    print("Computing polar projections with foreshortening for all angles...")
    for ang in angles:
        Bx, By, Bz = get_polar_projection_with_foreshortening(
            Bx_base, By_base, Bz_base, ang, apply_foreshortening_effect=True)

        hdu_bx = fits.ImageHDU(Bx.astype(np.float32), name=f'BX_{ang}')
        hdu_by = fits.ImageHDU(By.astype(np.float32), name=f'BY_{ang}')
        hdu_bz = fits.ImageHDU(Bz.astype(np.float32), name=f'BZ_{ang}')
        
        for hdu in [hdu_bx, hdu_by, hdu_bz]:
            hdu.header['ROTATION'] = ang
            hdu.header['FORESHORT'] = True
        
        hdus.extend([hdu_bx, hdu_by, hdu_bz])
        all_results.append((Bx, By, Bz, ang))

    hdul = fits.HDUList(hdus)
    hdul.writeto('flower_field_polar_projections.fits', overwrite=True)
    for Bx, By, Bz, ang in all_results:
        fig = plot_flower_field(Bx, By, Bz, 
                               f'Flower magnetic field with foreshortening: θ={ang}°')
        plt.show()