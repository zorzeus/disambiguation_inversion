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
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-10
    sinh_kz = np.sinh(k * z_gap)
    cosh_kz = np.cosh(k * z_gap)
    A = (bz_lower_ft * cosh_kz - bz_upper_ft) / sinh_kz
    Bx_ft = -1j * kx * A / k
    By_ft = -1j * ky * A / k
    Bx_ft[0, 0] = 0
    By_ft[0, 0] = 0
    Bx = np.real(ifftn(Bx_ft))
    By = np.real(ifftn(By_ft))
    return Bx, By


def create_base_flower_configuration(nx=512, ny=512, pixel_size=0.06):
    x = (np.arange(nx) - nx / 2) * pixel_size
    y = (np.arange(ny) - ny / 2) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')
    Bz_lower = np.zeros((nx, ny))
    Bz_upper = np.zeros((nx, ny))

    central_spot_params = {
        'lower': {'B0': -2500, 'a0': 3.0, 'B1': -800, 'a1': 4.5, 'n1': 6,
                  'phi1': 0, 'B2': -400, 'a2': 6.0, 'n2': 12, 'phi2': np.pi / 6},
        'upper': {'B0': -2500, 'a0': 3.0, 'B1': -800, 'a1': 4.5, 'n1': 6,
                  'phi1': np.pi / 4, 'B2': -400, 'a2': 6.0, 'n2': 12, 'phi2': np.pi / 3}}
    positive_spot_params = {
        'lower': {'B0': 2000, 'a0': 2.0, 'B1': 600, 'a1': 3.0, 'n1': 8,
                  'phi1': 0, 'B2': 300, 'a2': 4.0, 'n2': 16, 'phi2': 0},
        'upper': {'B0': 2000, 'a0': 2.0, 'B1': 600, 'a1': 3.0, 'n1': 8,
                  'phi1': np.pi / 5, 'B2': 300, 'a2': 4.0, 'n2': 16, 'phi2': np.pi / 4}}
    small_negative_params = {
        'lower': {'B0': -1500, 'a0': 1.5, 'B1': -400, 'a1': 2.0, 'n1': 4,
                  'phi1': 0, 'B2': -200, 'a2': 2.5, 'n2': 8, 'phi2': 0},
        'upper': {'B0': -1500, 'a0': 1.5, 'B1': -400, 'a1': 2.0, 'n1': 4,
                  'phi1': np.pi / 3, 'B2': -200, 'a2': 2.5, 'n2': 8, 'phi2': np.pi / 6}}
    upper_positive_params = {
        'lower': {'B0': 1800, 'a0': 1.8, 'B1': 500, 'a1': 2.5, 'n1': 6,
                  'phi1': 0, 'B2': 250, 'a2': 3.0, 'n2': 12, 'phi2': 0},
        'upper': {'B0': 1800, 'a0': 1.8, 'B1': 500, 'a1': 2.5, 'n1': 6,
                  'phi1': np.pi / 6, 'B2': 250, 'a2': 3.0, 'n2': 12, 'phi2': np.pi / 4}}

    spots = [
        {'params': central_spot_params, 'center': (-8, 8)},
        {'params': positive_spot_params, 'center': (6, -6)},
        {'params': small_negative_params, 'center': (8, 2)},
        {'params': upper_positive_params, 'center': (-2, -8)}]

    for spot in spots:
        xc, yc = spot['center']
        Xs = X - xc
        Ys = Y - yc
        Bz_lower += generate_flower_sunspot(Xs, Ys, spot['params'], 'lower')
        Bz_upper += generate_flower_sunspot(Xs, Ys, spot['params'], 'upper')

    np.random.seed(42)
    for _ in range(50):
        px, py = np.random.uniform(-6, 6, 2)
        strength = np.random.choice([-500, 500]) * np.random.uniform(0.5, 1.5)
        size = np.random.uniform(0.3, 0.8)
        Xp = X - px
        Yp = Y - py
        plage = strength * np.exp(-(Xp**2 + Yp**2) / size**2)
        Bz_lower += plage
        Bz_upper += plage * np.random.uniform(0.8, 1.2)

    Bz_lower -= np.mean(Bz_lower)
    Bz_upper -= np.mean(Bz_upper)

    z_gap = 5 * pixel_size
    Bx, By = compute_potential_field_fourier(Bz_lower, Bz_upper, z_gap, pixel_size, pixel_size)
    Bz = Bz_lower
    return Bx, By, Bz, X, Y, x, y, pixel_size


def apply_foreshortening_transformation(Bx, By, Bz, theta_deg, pixel_size=0.06, use_xi_curvature=False, use_eta_curvature=False, L_deg=0):
    theta_rad = np.radians(theta_deg)
    
    Bx_r = Bx.copy()
    By_r = By * np.cos(theta_rad) - Bz * np.sin(theta_rad)
    Bz_r = By * np.sin(theta_rad) + Bz * np.cos(theta_rad)
    
    if use_xi_curvature:
        Bx_r, By_r, Bz_r = apply_xi_curvature_correction(Bx_r, By_r, Bz_r, L_deg, pixel_size)
    
    if use_eta_curvature:
        Bx_r, By_r, Bz_r = apply_eta_curvature_correction(Bx_r, By_r, Bz_r, theta_deg, pixel_size)
    
    zf = np.round(np.cos(theta_rad), 4)
    if zf <= 1e-4:
        zf = 1e-4
    
    Bx_f = zoom(Bx_r, zoom=(zf, 1), order=1)
    By_f = zoom(By_r, zoom=(zf, 1), order=1)
    Bz_f = zoom(Bz_r, zoom=(zf, 1), order=1)
    
    return Bx_f, By_f, Bz_f, zf


def apply_xi_curvature_correction(Bx, By, Bz, L_deg, pixel_size):
    nx, ny = Bx.shape
    x = (np.arange(nx) - nx/2) * pixel_size
    y = (np.arange(ny) - ny/2) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    R = 960.0
    L_rad = np.radians(L_deg)
    
    delta_xi = X * np.cos(L_rad) - (X**2 / (2*R)) * np.sin(L_rad)
    
    from scipy.ndimage import map_coordinates
    
    xi_grid = (X + delta_xi) / pixel_size + nx/2
    eta_grid = Y / pixel_size + ny/2
    
    Bx_corr = map_coordinates(Bx, [xi_grid.T, eta_grid.T], order=1, mode='nearest')
    By_corr = map_coordinates(By, [xi_grid.T, eta_grid.T], order=1, mode='nearest')
    Bz_corr = map_coordinates(Bz, [xi_grid.T, eta_grid.T], order=1, mode='nearest')
    
    return Bx_corr.T, By_corr.T, Bz_corr.T


def apply_eta_curvature_correction(Bx, By, Bz, B_deg, pixel_size):
    nx, ny = Bx.shape
    x = (np.arange(nx) - nx/2) * pixel_size
    y = (np.arange(ny) - ny/2) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    R = 960.0
    B_rad = np.radians(B_deg)
    
    delta_eta = -R * (B_rad**3) / 6.0
    
    from scipy.ndimage import map_coordinates
    
    xi_grid = X / pixel_size + nx/2
    eta_grid = (Y + delta_eta) / pixel_size + ny/2
    
    Bx_corr = map_coordinates(Bx, [xi_grid.T, eta_grid.T], order=1, mode='nearest')
    By_corr = map_coordinates(By, [xi_grid.T, eta_grid.T], order=1, mode='nearest')
    Bz_corr = map_coordinates(Bz, [xi_grid.T, eta_grid.T], order=1, mode='nearest')
    
    return Bx_corr.T, By_corr.T, Bz_corr.T


def apply_simple_rotation(Bx, By, Bz, theta_deg):
    theta_rad = np.radians(theta_deg)
    Bx_r = Bx.copy()
    By_r = By * np.cos(theta_rad) - Bz * np.sin(theta_rad)
    Bz_r = By * np.sin(theta_rad) + Bz * np.cos(theta_rad)
    return Bx_r, By_r, Bz_r


def plot_flower_field(Bx, By, Bz, title='Flower Magnetic Field', fontsize=14):
    azimuth = np.arctan2(By, Bx) * 180 / np.pi
    azimuth = (azimuth + 360) % 360
    B_total = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_tr = np.sqrt(Bx**2 + By**2)

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(231)
    im1 = plt.imshow(Bx, origin='lower', cmap='bwr', vmin=-500, vmax=500)
    plt.title(r'$B_x\,\rm [G]$', fontsize=fontsize)
    plt.colorbar(im1)

    plt.subplot(232)
    im2 = plt.imshow(By, origin='lower', cmap='bwr', vmin=-1000, vmax=1000)
    plt.title(r'$B_y\,\rm [G]$', fontsize=fontsize)
    plt.colorbar(im2)
    
    plt.subplot(233)
    im3 = plt.imshow(Bz, origin='lower', cmap='bwr', vmin=-700, vmax=700)
    plt.title(r'$B_z\,\rm [G]$', fontsize=fontsize)
    plt.colorbar(im3)

    plt.subplot(234)
    im4 = plt.imshow(B_total, origin='lower', cmap='plasma')
    plt.title(r'$|B|\,\rm [G]$', fontsize=fontsize)
    plt.colorbar(im4)

    plt.subplot(235)
    im5 = plt.imshow(B_tr, origin='lower', cmap='plasma')
    plt.colorbar(im5)
    plt.title(r'$B_{\perp}\,\rm [G]$', fontsize=fontsize)

    plt.subplot(236)
    im6 = plt.imshow(azimuth, origin='lower', cmap='twilight')
    plt.title(r'$\varphi\,[^{\circ}]$', fontsize=fontsize)
    plt.colorbar(im6)

    plt.suptitle(title, fontsize=fontsize+2)
    plt.tight_layout()
    return fig


def generate_tilted_viewpoint_field(theta_deg, nx=512, ny=512, pixel_size=0.06, apply_foreshortening_effect=True, use_xi_curvature=False, use_eta_curvature=False, L_deg=0):
    Bx_h, By_h, Bz_h, X, Y, x, y, psize = create_base_flower_configuration(nx, ny, pixel_size)
    
    if apply_foreshortening_effect:
        Bx_img, By_img, Bz_img, zoom_factor = apply_foreshortening_transformation(Bx_h, By_h, Bz_h, theta_deg, pixel_size, use_xi_curvature, use_eta_curvature, L_deg)
    else:
        Bx_img, By_img, Bz_img = apply_simple_rotation(Bx_h, By_h, Bz_h, theta_deg)
        zoom_factor = 1.0

    geometry = {
        'L_minus_L0': L_deg,
        'B_field': theta_deg,
        'B0': 0,
        'P0': 0
    }

    return {
        'Bx_image': Bx_img,
        'By_image': By_img,
        'Bz_image': Bz_img,
        'geometry': geometry,
        'zoom_factor': zoom_factor,
        'theta_deg': theta_deg,
        'use_xi_curvature': use_xi_curvature,
        'use_eta_curvature': use_eta_curvature
    }


if __name__ == "__main__":
    angles = [0, 15, 30, 45, 60, 75]
    hdus = [fits.PrimaryHDU()]

    for theta in angles:
        result = generate_tilted_viewpoint_field(theta, apply_foreshortening_effect=True, use_xi_curvature=False, use_eta_curvature=False)
        Bx = result['Bx_image']
        By = result['By_image']
        Bz = result['Bz_image']
        geom = result['geometry']
        zoom_factor = result['zoom_factor']

        for name, arr in zip(['BX', 'BY', 'BZ'], [Bx, By, Bz]):
            hdu = fits.ImageHDU(arr.astype(np.float32), name=f"{name}_{theta}")
            hdu.header['B_FIELD'] = geom['B_field']
            hdu.header['B0'] = geom['B0']
            hdu.header['P0'] = geom['P0']
            hdu.header['THETA'] = theta
            hdu.header['ZOOMFAC'] = zoom_factor
            hdu.header['FORESHRT'] = True
            hdu.header['XI_CURV'] = result['use_xi_curvature']
            hdu.header['ETA_CURV'] = result['use_eta_curvature']
            hdus.append(hdu)

        fig = plot_flower_field(Bx, By, Bz, f'Flower magnetic field with foreshortening: θ={theta}°')
        plt.show()

    hdul = fits.HDUList(hdus)
    hdul.writeto('flower_field_polar_projections_corrected.fits', overwrite=True)
    
    print("Flux conservation check:")
    base_result = generate_tilted_viewpoint_field(0, apply_foreshortening_effect=False)
    base_flux = np.sum(np.abs(base_result['Bz_image']))
    
    for theta in [30, 60]:
        result_flat = generate_tilted_viewpoint_field(theta, apply_foreshortening_effect=True, use_xi_curvature=False, use_eta_curvature=False)
        flat_flux = np.sum(np.abs(result_flat['Bz_image']))
        print(f"θ={theta}° (flat): Flux ratio = {flat_flux/base_flux:.3f}")
        
        result_xi = generate_tilted_viewpoint_field(theta, apply_foreshortening_effect=True, use_xi_curvature=True, use_eta_curvature=False, L_deg=10)
        xi_flux = np.sum(np.abs(result_xi['Bz_image']))
        print(f"θ={theta}° (ξ-curvature, L=10°): Flux ratio = {xi_flux/base_flux:.3f}")
        
        result_eta = generate_tilted_viewpoint_field(theta, apply_foreshortening_effect=True, use_xi_curvature=False, use_eta_curvature=True)
        eta_flux = np.sum(np.abs(result_eta['Bz_image']))
        print(f"θ={theta}° (η-curvature): Flux ratio = {eta_flux/base_flux:.3f}")
        
        result_both = generate_tilted_viewpoint_field(theta, apply_foreshortening_effect=True, use_xi_curvature=True, use_eta_curvature=True, L_deg=10)
        both_flux = np.sum(np.abs(result_both['Bz_image']))
        print(f"θ={theta}° (both curvatures, L=10°): Flux ratio = {both_flux/base_flux:.3f}")
        print()