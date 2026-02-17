import numpy as np
from numba import jit
from scipy.fft import fft2, ifft2, fftfreq
import time


def compute_dBz_dz_spectral(Bz, dx=1.0, dy=1.0, pad_size=8):
    ny, nx = Bz.shape
    Bz_padded = np.pad(Bz, pad_size, mode='edge')
    Bz_padded = Bz_padded - np.mean(Bz_padded)
    ny_pad, nx_pad = Bz_padded.shape
    
    kx = 2 * np.pi * fftfreq(nx_pad, dx) 
    ky = 2 * np.pi * fftfreq(ny_pad, dy)  
    KX, KY = np.meshgrid(kx, ky, indexing='xy') 
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1e-12
    
    Bz_fft = fft2(Bz_padded)
    dBz_dz_fft = -K * Bz_fft
    dBz_dz_fft[0, 0] = 0.0
    dBz_dz_padded = np.real(ifft2(dBz_dz_fft))
    dBz_dz = dBz_dz_padded[pad_size:pad_size+ny, pad_size:pad_size+nx]
    
    return dBz_dz


@jit(nopython=True, fastmath=True, inline='always')
def compute_4point_average(array, i, j):
    return 0.25 * (array[i, j] + array[i+1, j] + array[i, j+1] + array[i+1, j+1])


@jit(nopython=True, fastmath=True, inline='always')
def compute_single_stencil(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy):

    dBx_dx = (Bx[i+1, j] - Bx[i, j] + Bx[i+1, j+1] - Bx[i, j+1]) * inv_2dx
    dBy_dx = (By[i+1, j] - By[i, j] + By[i+1, j+1] - By[i, j+1]) * inv_2dx
    dBx_dy = (Bx[i, j+1] - Bx[i, j] + Bx[i+1, j+1] - Bx[i+1, j]) * inv_2dy
    dBy_dy = (By[i, j+1] - By[i, j] + By[i+1, j+1] - By[i+1, j]) * inv_2dy
    
    dBz_dz_avg = compute_4point_average(dBz_dz, i, j)
    
    div_B = dBx_dx + dBy_dy + dBz_dz_avg
    Jz = dBy_dx - dBx_dy
    
    return div_B**2 + lambda_w * Jz**2


@jit(nopython=True, fastmath=True)
def compute_flip_delta(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy):
   
    ny, nx = Bx.shape
    
    E_old = 0.0
    if i > 0 and j > 0:
        E_old += compute_single_stencil(Bx, By, dBz_dz, i-1, j-1, lambda_w, inv_2dx, inv_2dy)
    if i > 0 and j < nx-1:
        E_old += compute_single_stencil(Bx, By, dBz_dz, i-1, j, lambda_w, inv_2dx, inv_2dy)
    if i < ny-1 and j > 0:
        E_old += compute_single_stencil(Bx, By, dBz_dz, i, j-1, lambda_w, inv_2dx, inv_2dy)
    if i < ny-1 and j < nx-1:
        E_old += compute_single_stencil(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy)
    
    Bx[i, j] = -Bx[i, j]
    By[i, j] = -By[i, j]
    
    E_new = 0.0
    if i > 0 and j > 0:
        E_new += compute_single_stencil(Bx, By, dBz_dz, i-1, j-1, lambda_w, inv_2dx, inv_2dy)
    if i > 0 and j < nx-1:
        E_new += compute_single_stencil(Bx, By, dBz_dz, i-1, j, lambda_w, inv_2dx, inv_2dy)
    if i < ny-1 and j > 0:
        E_new += compute_single_stencil(Bx, By, dBz_dz, i, j-1, lambda_w, inv_2dx, inv_2dy)
    if i < ny-1 and j < nx-1:
        E_new += compute_single_stencil(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy)
    
    Bx[i, j] = -Bx[i, j]
    By[i, j] = -By[i, j]
    
    return E_new - E_old


@jit(nopython=True, fastmath=True)
def anneal_iteration(Bx, By, dBz_dz, tvar, lambda_w, inv_2dx, inv_2dy, jump):
 
    ny, nx = Bx.shape
    n_accepted = 0
    total_dE = 0.0
    
    i_start = int(np.random.random() * jump)
    j_start = int(np.random.random() * jump)
    
    for i in range(i_start, ny, jump):
        for j in range(j_start, nx, jump):
            dE = compute_flip_delta(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy)
            
            temp = tvar[i, j]
            accept = False
            
            if dE < 0.0:
                accept = True
            elif temp > 1e-30:
                if np.random.random() < np.exp(-dE / temp):
                    accept = True
            
            if accept:
                Bx[i, j] = -Bx[i, j]
                By[i, j] = -By[i, j]
                n_accepted += 1
                total_dE += dE
    
    return n_accepted, total_dE


@jit(nopython=True, fastmath=True)
def compute_total_energy(Bx, By, dBz_dz, lambda_w, inv_2dx, inv_2dy):
    
    ny, nx = Bx.shape
    E_total = 0.0
    
    for i in range(ny-1):
        for j in range(nx-1):
            E_total += compute_single_stencil(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy)
    
    return E_total

@jit(nopython=True, fastmath=True)
def compute_divB_and_jz_totals(Bx, By, dBz_dz, lambda_w, inv_2dx, inv_2dy):

    ny, nx = Bx.shape
    total_divB = 0.0
    total_jz = 0.0
    
    for i in range(ny-1):
        for j in range(nx-1):
            dBx_dx = (Bx[i+1, j] - Bx[i, j] + Bx[i+1, j+1] - Bx[i, j+1]) * inv_2dx
            dBy_dx = (By[i+1, j] - By[i, j] + By[i+1, j+1] - By[i, j+1]) * inv_2dx
            dBx_dy = (Bx[i, j+1] - Bx[i, j] + Bx[i+1, j+1] - Bx[i+1, j]) * inv_2dy
            dBy_dy = (By[i, j+1] - By[i, j] + By[i+1, j+1] - By[i+1, j]) * inv_2dy
            
            dBz_dz_avg = compute_4point_average(dBz_dz, i, j)
            
            divB = dBx_dx + dBy_dy + dBz_dz_avg
            jz = dBy_dx - dBx_dy
            
            total_divB += np.abs(divB)
            total_jz += np.abs(jz)
    
    return total_divB, total_jz

@jit(nopython=True, fastmath=True)
def initialize_temperature(Bx, By, dBz_dz, lambda_w, inv_2dx, inv_2dy, tfac0, n_samples):
  
    ny, nx = Bx.shape
    tvar = np.zeros((ny, nx))
    sample_count = np.zeros((ny, nx))
    
    for _ in range(n_samples):
        i = int(np.random.random() * ny)
        j = int(np.random.random() * nx)
        
        dE = compute_flip_delta(Bx, By, dBz_dz, i, j, lambda_w, inv_2dx, inv_2dy)
        
        if np.abs(dE) > tvar[i, j]:
            tvar[i, j] = np.abs(dE)
        
        sample_count[i, j] += 1
    
    for i in range(ny):
        for j in range(nx):
            if sample_count[i, j] == 0:
                sum_t = 0.0
                count = 0
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ii = i + di
                        jj = j + dj
                        if 0 <= ii < ny and 0 <= jj < nx and sample_count[ii, jj] > 0:
                            sum_t += tvar[ii, jj]
                            count += 1
                if count > 0:
                    tvar[i, j] = sum_t / count
    
    tvar = tfac0 * tvar
    
    max_temp = np.max(tvar)
    min_temp = max_temp * 1e-10
    for i in range(ny):
        for j in range(nx):
            if tvar[i, j] < min_temp:
                tvar[i, j] = min_temp
    
    return tvar


class Disambiguator:
  
    
    def __init__(self, lambda_weight=1.0, dx=1.0, dy=1.0, tfac0=2.0,
                 tfactr=0.95, jump=1, neq=20, seed=42,
                 conv_check_window=10, conv_threshold=0.001):
        
        self.lambda_weight = lambda_weight
        self.dx = dx
        self.dy = dy
        self.tfac0 = tfac0
        self.tfactr = tfactr
        self.jump = jump
        self.neq = neq
        self.conv_check_window = conv_check_window
        self.conv_threshold = conv_threshold
        
        np.random.seed(seed)
        
        self.inv_2dx = 1.0 / (2.0 * dx)
        self.inv_2dy = 1.0 / (2.0 * dy)
    
    def disambiguate(self, Bx, By, Bz, verbose=False, max_iterations=100):
        
        start_time = time.time()
        ny, nx = Bx.shape
        
        dBz_dz = compute_dBz_dz_spectral(Bz, self.dx, self.dy)
    
        Bx_work = Bx.copy()
        By_work = By.copy()
        
        n_samples = min(20000, max(5000, 10 * ny * nx))
        tvar = initialize_temperature(Bx_work, By_work, dBz_dz, 
                                     self.lambda_weight, self.inv_2dx, self.inv_2dy,
                                     self.tfac0, n_samples)
        
        total_energy = compute_total_energy(Bx_work, By_work, dBz_dz,
                                           self.lambda_weight, self.inv_2dx, self.inv_2dy)
        
        if verbose:
            print(f"SA | Initial E: {total_energy:.2e}")
    
        divB_init, jz_init = compute_divB_and_jz_totals(Bx_work, By_work, dBz_dz,
                                                        self.lambda_weight, self.inv_2dx, self.inv_2dy)

        energy_history = [total_energy]
        divB_history = [divB_init]
        jz_history = [jz_init]
        neq_actual = max(10, self.neq)
        
        iteration = 0
        converged = False
        t_max = np.max(tvar)
        t_stop = 1e-7 * t_max
        
        while not converged:
            n_accepted = 0
            
            for _ in range(neq_actual):
                n_acc, dE = anneal_iteration(Bx_work, By_work, dBz_dz, tvar,
                                            self.lambda_weight, self.inv_2dx, self.inv_2dy, 
                                            self.jump)
                n_accepted += n_acc
                total_energy += dE
            
            tvar *= self.tfactr

            iteration += 1
            divB_iter, jz_iter = compute_divB_and_jz_totals(Bx_work, By_work, dBz_dz,
                                                            self.lambda_weight, self.inv_2dx, self.inv_2dy)

            energy_history.append(total_energy)
            divB_history.append(divB_iter)
            jz_history.append(jz_iter)
            
            if verbose and iteration % 20 == 0:
                print(f"  Iter {iteration:3d} | E: {total_energy:.3e} | T: {np.max(tvar):.2e} | Accepted: {n_accepted}")
         
            if len(energy_history) >= self.conv_check_window + 1:
                E_old = energy_history[-self.conv_check_window-1]
                E_new = energy_history[-1]
                if E_old > 0:
                    rel_change = abs(E_new - E_old) / E_old
                    if rel_change < self.conv_threshold:
                        converged = True
            
            if n_accepted == 0 or np.max(tvar) < t_stop or iteration >= max_iterations:
                converged = True
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"SA complete | {total_time:.1f}s | Final E: {total_energy:.2e} | {iteration} iterations")
        
        np.savetxt('SA_behaviour.txt', 
                np.column_stack([energy_history, divB_history, jz_history]),
                header='Iteration  Energy  divB  Jz',
                fmt='%.6e')

        info = {
            'energy_history': energy_history,
            'divB_history': divB_history,
            'jz_history': jz_history,
            'final_energy': total_energy,
            'total_time': total_time,
            'n_iterations': iteration}

        return Bx_work, By_work, info
