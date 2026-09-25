import os
import time
from datetime import datetime
import numpy as np
strtimenow = lambda: datetime.now().strftime("%H:%M:%S")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
#from uncertainties import ufloat
#from uncertainties import unumpy
#from dmx_hom import *
import math

def trova_picchi(bin_edges, bin_values, bin_centrale, distanza_bin=7):
    """
    Trova i picchi in un istogramma e li separa in:
    - sinistra
    - destra
    - centrale (quello più vicino a bin_centrale)

    Parametri
    ---------
    bin_edges : array
        Bordo dei bin dell'istogramma
    bin_values : array
        Valori dei bin
    bin_centrale : float
        Valore atteso del picco centrale (nella stessa unità di bin_edges)
    distanza_bin : int
        Numero minimo di bin tra due picchi (default=7)

    Ritorna
    -------
    pos_sx, val_sx, pos_dx, val_dx, pos_centr, val_centr , pos_noise_sx, val_noise_sx, pos_noise_dx, val_noise_dx

    """
    # calcolo i centri dei bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # trovo tutti i picchi
    picchi, _ = find_peaks(bin_values, distance=distanza_bin)
    
    if len(picchi) == 0:
        return [], [], [], [], None, None , [], [], [], []
    
    # individuo il picco "centrale" come quello più vicino a bin_centrale
    idx_centrale = picchi[np.argmin(np.abs(bin_centers[picchi] - bin_centrale))]
    
    # separo picchi a sinistra e a destra
    picchi_sx = picchi[picchi < idx_centrale]
    picchi_dx = picchi[picchi > idx_centrale]
    
    # preparo gli array richiesti
    pos_sx = bin_centers[picchi_sx]
    val_sx = bin_values[picchi_sx]
    pos_dx = bin_centers[picchi_dx]
    val_dx = bin_values[picchi_dx]
    pos_centr = bin_centers[idx_centrale]
    val_centr = bin_values[idx_centrale]
    pos_noise_sx = bin_centers[((picchi_sx[1:] + picchi_sx[:-1]) // 2).astype(int)]
    val_noise_sx = bin_values[((picchi_sx[1:] + picchi_sx[:-1]) // 2).astype(int)]
    pos_noise_dx = bin_centers[((picchi_dx[1:] + picchi_dx[:-1]) // 2).astype(int)]
    val_noise_dx = bin_values[((picchi_dx[1:] + picchi_dx[:-1]) // 2).astype(int)]
    
    return pos_sx, val_sx, pos_dx, val_dx, pos_centr, val_centr , pos_noise_sx, val_noise_sx, pos_noise_dx, val_noise_dx

def noise_estimation(bin_values, picchi_sx, picchi_dx):
    picchi_sx = picchi_sx.astype(int)
    picchi_dx = picchi_dx.astype(int)

    noise_sx = np.array([
        picchi_sx[i] + np.argmin(bin_values[picchi_sx[i]:picchi_sx[i+1]])
        for i in range(len(picchi_sx) - 1)
    ])

    noise_dx = np.array([
        picchi_dx[i] + np.argmin(bin_values[picchi_dx[i]:picchi_dx[i+1]])
        for i in range(len(picchi_dx) - 1)
    ])

    return noise_sx, noise_dx

       
def plot_picchi(bin_edges, bin_values, bin_centrale):
    # calcolo i centri dei bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # trovo tutti i picchi
    picchi, _ = find_peaks(bin_values, distance=7)
    
    if len(picchi) == 0:
        print("Nessun picco trovato.")
        return
    
    # individuo il picco "centrale" come quello più vicino a bin_centrale
    idx_centrale = picchi[np.argmin(np.abs(bin_centers[picchi] - bin_centrale))]
    
    # separo picchi a sinistra e a destra
    picchi_sx = picchi[picchi < idx_centrale]
    picchi_dx = picchi[picchi > idx_centrale]
    
    # plotto istogramma
    plt.figure(figsize=(8,5))
    plt.bar(bin_centers, bin_values, width=bin_edges[1]-bin_edges[0], alpha=0.5, color="skyblue", edgecolor="k")
    
    # plotto i picchi laterali (gialli)
    plt.scatter(bin_centers[picchi_sx], bin_values[picchi_sx], color="yellow", marker="x", s=100, label="Picchi SX")
    plt.scatter(bin_centers[picchi_dx], bin_values[picchi_dx], color="yellow", marker="x", s=100, label="Picchi DX")
    
    # plotto il picco centrale (rosso)
    plt.scatter(bin_centers[idx_centrale], bin_values[idx_centrale], color="red", marker="x", s=120, label="Picco centrale")
    
    plt.xlabel("x")
    plt.ylabel("Conteggi")
    plt.title("Istogramma con picchi identificati")
    plt.legend()
    plt.show()


def fit_retta(x, y):
    """
    Fit lineare y = m*x + q ai punti (x, y).
    Restituisce m, q, rmse e la matrice di covarianza.
    """
    coeffs, pcov = np.polyfit(x, y, 1, cov=True)
    m, q = coeffs
    
    # valori fittati
    y_fit = m * x + q
    
    # scarto quadratico medio
    rmse = np.sqrt(np.mean((y - y_fit)**2))
    
    return m, q, rmse, pcov
def intersezione_rette(m1, q1, m2, q2):
    """
    Trova l'intersezione tra due rette y = m1*x + q1 e y = m2*x + q2.
    Restituisce (x_int, y_int).
    """
    if np.isclose(m1, m2):
        raise ValueError("Le rette sono parallele o quasi parallele, niente intersezione.")
    
    x_int = (q2 - q1) / (m1 - m2)
    y_int = m1 * x_int + q1
    
    return x_int, y_int
def distanze_picco(m1, q1, m2, q2, x_int, y_int, x_centr, y_centr):
    """
    Calcola le distanze verticali:
    - tra il picco centrale e l'intersezione delle due rette
    - tra il picco centrale e ciascuna retta (alla stessa x del picco centrale)
    
    Restituisce (d_int, d_r1, d_r2).
    """
    # distanza dal punto di intersezione
    d_int = y_int - y_centr
    
    # valori delle rette in corrispondenza di x_centr
    y_r1 = m1 * x_centr + q1
    y_r2 = m2 * x_centr + q2
    
    d_r1 = y_r1 - y_centr
    d_r2 = y_r2 - y_centr
    
    return d_int, d_r1, d_r2

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fit_retta(x, y):
    coeffs, pcov = np.polyfit(x, y, 1, cov=True)
    m, q = coeffs
    y_fit = m * x + q
    rmse = np.sqrt(np.mean((y - y_fit) ** 2))
    return m, q, rmse, pcov

def errore_intersezione_y(m1, q1, pcov1, m2, q2, pcov2):
    """
    Calcola l'errore sulla coordinata y del punto di intersezione tra due rette.
    """
    denom = m1 - m2
    if np.isclose(denom, 0):
        return np.inf
    dq = q2 - q1
    dydm1 = dq * m2 / (denom**2)
    dydq1 = m2 / denom
    dydm2 = -m1 * dq / (denom**2)
    dydq2 = m1 / denom
    sigma_m1_sq = pcov1[0, 0]
    sigma_q1_sq = pcov1[1, 1]
    cov_m1q1 = pcov1[0, 1]
    sigma_m2_sq = pcov2[0, 0]
    sigma_q2_sq = pcov2[1, 1]
    cov_m2q2 = pcov2[0, 1]
    sigma_y_sq = (
        (dydm1**2) * sigma_m1_sq +
        (dydq1**2) * sigma_q1_sq +
        2 * dydm1 * dydq1 * cov_m1q1 +
        (dydm2**2) * sigma_m2_sq +
        (dydq2**2) * sigma_q2_sq +
        2 * dydm2 * dydq2 * cov_m2q2
    )
    return np.sqrt(sigma_y_sq)

def intersezione_rette(m1, q1, m2, q2):
    if np.isclose(m1, m2):
        #raise ValueError("Le rette sono parallele o quasi parallele.")
        x_int = 0
    else:
        x_int = (q2 - q1) / (m1 - m2)
    y_int = m1 * x_int + q1
    return x_int, y_int

def plot_picchi_con_fit(bin_edges, bin_values, bin_centrale, distanza_bin=7):
    # centri bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # trova picchi
    picchi, _ = find_peaks(bin_values, distance=distanza_bin)
    if len(picchi) == 0:
        print("Nessun picco trovato.")
        return
    
    # picco centrale
    idx_centrale = picchi[np.argmin(np.abs(bin_centers[picchi] - bin_centrale))]
    picchi_sx = picchi[picchi < idx_centrale]
    picchi_dx = picchi[picchi > idx_centrale]
    
    # fit rette sx e dx (solo se esistono almeno 2 punti)
    m_sx = q_sx = m_dx = q_dx = None
    x_int = y_int = None
    
    if len(picchi_sx) >= 2:
        m_sx, q_sx, _ = fit_retta(bin_centers[picchi_sx], bin_values[picchi_sx])
    if len(picchi_dx) >= 2:
        m_dx, q_dx, _ = fit_retta(bin_centers[picchi_dx], bin_values[picchi_dx])
    
    if (m_sx is not None) and (m_dx is not None):
        try:
            x_int, y_int = intersezione_rette(m_sx, q_sx, m_dx, q_dx)
        except ValueError:
            pass
    
    # plot istogramma
    plt.figure(figsize=(9,6))
    plt.bar(bin_centers, bin_values, width=bin_edges[1]-bin_edges[0],
            alpha=0.5, color="skyblue", edgecolor="k", label="Istogramma")
    
    # picchi
    plt.scatter(bin_centers[picchi_sx], bin_values[picchi_sx],
                color="yellow", marker="x", s=100, label="Picchi SX")
    plt.scatter(bin_centers[picchi_dx], bin_values[picchi_dx],
                color="yellow", marker="x", s=100, label="Picchi DX")
    plt.scatter(bin_centers[idx_centrale], bin_values[idx_centrale],
                color="red", marker="x", s=120, label="Picco centrale")
    
    # rette
    if m_sx is not None:
        xfit = np.linspace(min(bin_centers[picchi_sx]), max(bin_centers[picchi_sx]), 100)
        plt.plot(xfit, m_sx*xfit + q_sx, 'g--', label="Fit SX")
    if m_dx is not None:
        xfit = np.linspace(min(bin_centers[picchi_dx]), max(bin_centers[picchi_dx]), 100)
        plt.plot(xfit, m_dx*xfit + q_dx, 'm--', label="Fit DX")
    
    # intersezione
    if x_int is not None and y_int is not None:
        plt.plot(x_int, y_int, 'ro', label="Intersezione")
    
    plt.xlabel("x")
    plt.ylabel("Conteggi")
    plt.title("Istogramma con picchi e rette fittate")
    plt.legend()
    plt.show()


def fit_inverse(x, y):
    """
    Fit dei dati (x, y) con la funzione y = a/x + b
    Restituisce i parametri del fit (a, b) e la funzione fittata.
    """
    # modello
    def model(x, a, b):
        return a / x + b

    # fit
    popt, pcov = curve_fit(model, x, y, p0=(1.0, 0.0))  # guess iniziale
    a, b = popt
    perr = np.sqrt(np.diag(pcov))  # incertezze sui parametri
    
    # curve fittata
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = model(x_fit, *popt)

    # plot
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color="blue", label="Dati")
    plt.plot(x_fit, y_fit, "r-", label=f"Fit: y = {a:.3f}/x + {b:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fit con funzione 1/x")
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    return (a, b), perr

def idx_to_pair(n, iii):
    # Calcolo di i risolvendo l'equazione quadratica
    i = int((2*n - 1 - math.sqrt((2*n - 1)**2 - 8*iii)) // 2)
    
    # Verifica che i sia corretto, altrimenti aggiusta
    while iii < i*n - (i*(i+1))//2:
        i -= 1
    while iii >= (i+1)*n - ((i+1)*(i+2))//2:
        i += 1

    # Calcolo offset
    offset = i * n - (i * (i + 1)) // 2
    
    # Calcolo ii
    ii = (iii - offset) + (i + 1)
    
    return i, ii

def integra_picchi(bin_edges, bin_values, pos_picchi, finestra):
    """
    Integra i picchi in un istogramma attorno alle loro posizioni.

    Parametri
    ---------
    bin_edges : array
        Bordo dei bin dell'istogramma.
    bin_values : array
        Valori dei bin dell'istogramma.
    pos_picchi : array
        Posizioni (x) dei picchi da integrare.
    finestra : float
        Semi-ampiezza della finestra di integrazione (stessa unità di bin_edges).
        L'integrazione sarà fatta su [pos - finestra, pos + finestra].

    Ritorna
    -------
    pos_out : array
        Posizioni dei picchi.
    integrali : array
        Valori degli integrali calcolati.
    """
    # centri dei bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    larghezza_bin = bin_edges[1] - bin_edges[0]

    pos_out = []
    integrali = []

    for pos in np.atleast_1d(pos_picchi):
        # seleziono i bin dentro la finestra
        mask = (bin_centers >= pos - finestra) & (bin_centers <= pos + finestra)
        x_sel = bin_centers[mask]
        y_sel = bin_values[mask]

        if len(x_sel) > 1:
            # integrazione con regola del trapezio
            area = np.trapz(y_sel, x_sel)
            pos_out.append(int(pos))
            integrali.append(area)

    return np.array(pos_out, dtype=np.int64), np.array(integrali)

def integra_picchi_normalizzati(bin_edges, bin_values, pos_picchi, finestra):
    """
    Integra i picchi attorno alle loro posizioni e normalizza
    dividendo per la larghezza totale della finestra (2*finestra).
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pos_out, integrali_norm = [], []

    for pos in np.atleast_1d(pos_picchi):
        mask = (bin_centers >= pos - finestra) & (bin_centers <= pos + finestra)
        x_sel = bin_centers[mask]
        y_sel = bin_values[mask]

        if len(x_sel) > 1:
            area = np.trapz(y_sel, x_sel)
            area_norm = area / (2 * finestra)   # normalizzazione
            pos_out.append(pos)
            integrali_norm.append(area_norm)

    return np.array(pos_out), np.array(integrali_norm)

def plot_picchi_con_fit_integrali_normalizzati(bin_edges, bin_values, bin_centrale, distanza_bin=7, finestra=2500):
    """
    Trova picchi, calcola integrali normalizzati, fitta con rette e mostra tutto.
    """
    # calcolo i centri dei bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # trova picchi
    picchi, _ = find_peaks(bin_values, distance=distanza_bin)
    if len(picchi) == 0:
        print("Nessun picco trovato.")
        return

    idx_centrale = picchi[np.argmin(np.abs(bin_centers[picchi] - bin_centrale))]
    picchi_sx = picchi[picchi < idx_centrale]
    picchi_dx = picchi[picchi > idx_centrale]

    # integrazione normalizzata
    pos_sx, int_sx = integra_picchi_normalizzati(bin_edges, bin_values, bin_centers[picchi_sx], finestra)
    pos_dx, int_dx = integra_picchi_normalizzati(bin_edges, bin_values, bin_centers[picchi_dx], finestra)
    pos_c, int_c = integra_picchi_normalizzati(bin_edges, bin_values, bin_centers[idx_centrale], finestra)

    # fit delle rette
    m_sx, q_sx, rms_sx = fit_retta(pos_sx, int_sx) if len(pos_sx) > 1 else (None, None, None)
    m_dx, q_dx, rms_dx = fit_retta(pos_dx, int_dx) if len(pos_dx) > 1 else (None, None, None)

    # intersezione
    x_int, y_int = None, None
    if m_sx is not None and m_dx is not None:
        x_int, y_int = intersezione_rette(m_sx, q_sx, m_dx, q_dx)

    # --- plot ---
    plt.figure(figsize=(9,6))
    plt.bar(bin_centers, bin_values, width=bin_edges[1]-bin_edges[0], alpha=0.5, color="skyblue", edgecolor="k")

    # picchi (x gialle)
    plt.scatter(bin_centers[picchi_sx], bin_values[picchi_sx], color="yellow", marker="x", s=100, label="Picchi SX")
    plt.scatter(bin_centers[picchi_dx], bin_values[picchi_dx], color="yellow", marker="x", s=100, label="Picchi DX")
    plt.scatter(bin_centers[idx_centrale], bin_values[idx_centrale], color="red", marker="x", s=120, label="Picco centrale")

    # integrali normalizzati (o blu)
    plt.scatter(pos_sx, int_sx, color="blue", marker="o", s=80, label="Integrali norm. SX")
    plt.scatter(pos_dx, int_dx, color="blue", marker="o", s=80, label="Integrali norm. DX")
    plt.scatter(pos_c, int_c, color="blue", marker="o", s=100, label="Integrale norm. centrale")

    # rette fittate
    if m_sx is not None:
        x_fit = np.linspace(min(pos_sx), max(pos_sx), 200)
        plt.plot(x_fit, m_sx*x_fit + q_sx, "g--", label="Fit SX")
    if m_dx is not None:
        x_fit = np.linspace(min(pos_dx), max(pos_dx), 200)
        plt.plot(x_fit, m_dx*x_fit + q_dx, "m--", label="Fit DX")

    # intersezione (^ rosso)
    if x_int is not None:
        plt.scatter(x_int, y_int, color="red", marker="^", s=120, label="Intersezione")

    plt.xlabel("x")
    plt.ylabel("Conteggi / Integrali normalizzati")
    plt.title("Istogramma con picchi e fit sugli integrali normalizzati")
    plt.legend()
    plt.show()

    return (pos_sx, int_sx), (pos_dx, int_dx), (pos_c, int_c), (m_sx, q_sx, rms_sx), (m_dx, q_dx, rms_dx), (x_int, y_int)

def plot_picchi_con_fit_integrali(bin_edges, bin_values, bin_centrale, distanza_bin=7, finestra=2500):
    """
    Trova picchi, calcola gli integrali, fitta con rette e mostra tutto.
    """
    # calcolo i centri dei bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # trova picchi
    picchi, _ = find_peaks(bin_values, distance=distanza_bin)
    if len(picchi) == 0:
        print("Nessun picco trovato.")
        return

    idx_centrale = picchi[np.argmin(np.abs(bin_centers[picchi] - bin_centrale))]
    picchi_sx = picchi[picchi < idx_centrale]
    picchi_dx = picchi[picchi > idx_centrale]

    # integrazione dei picchi
    pos_sx, int_sx = integra_picchi(bin_edges, bin_values, bin_centers[picchi_sx], finestra)
    pos_dx, int_dx = integra_picchi(bin_edges, bin_values, bin_centers[picchi_dx], finestra)
    pos_c, int_c = integra_picchi(bin_edges, bin_values, bin_centers[idx_centrale], finestra)

    # fit delle rette
    m_sx, q_sx, rms_sx = fit_retta(pos_sx, int_sx) if len(pos_sx) > 1 else (None, None, None)
    m_dx, q_dx, rms_dx = fit_retta(pos_dx, int_dx) if len(pos_dx) > 1 else (None, None, None)

    # intersezione
    x_int, y_int = None, None
    if m_sx is not None and m_dx is not None:
        x_int, y_int = intersezione_rette(m_sx, q_sx, m_dx, q_dx)

    # --- plot ---
    plt.figure(figsize=(9,6))
    plt.bar(bin_centers, bin_values, width=bin_edges[1]-bin_edges[0], alpha=0.5, color="skyblue", edgecolor="k")

    # picchi (x gialle)
    plt.scatter(bin_centers[picchi_sx], bin_values[picchi_sx], color="yellow", marker="x", s=100, label="Picchi SX")
    plt.scatter(bin_centers[picchi_dx], bin_values[picchi_dx], color="yellow", marker="x", s=100, label="Picchi DX")
    plt.scatter(bin_centers[idx_centrale], bin_values[idx_centrale], color="red", marker="x", s=120, label="Picco centrale")

    # integrali (o blu)
    plt.scatter(pos_sx, int_sx, color="blue", marker="o", s=80, label="Integrali SX")
    plt.scatter(pos_dx, int_dx, color="blue", marker="o", s=80, label="Integrali DX")
    plt.scatter(pos_c, int_c, color="blue", marker="o", s=100, label="Integrale centrale")

    # rette fittate
    if m_sx is not None:
        x_fit = np.linspace(min(pos_sx), max(pos_sx), 200)
        plt.plot(x_fit, m_sx*x_fit + q_sx, "g--", label="Fit SX")
    if m_dx is not None:
        x_fit = np.linspace(min(pos_dx), max(pos_dx), 200)
        plt.plot(x_fit, m_dx*x_fit + q_dx, "m--", label="Fit DX")

    # intersezione (^ rosso)
    if x_int is not None:
        plt.scatter(x_int, y_int, color="red", marker="^", s=120, label="Intersezione")

    plt.xlabel("x")
    plt.ylabel("Conteggi / Integrali")
    plt.title("Istogramma con picchi e fit sugli integrali")
    plt.legend()
    plt.show()

    return (pos_sx, int_sx), (pos_dx, int_dx), (pos_c, int_c), (m_sx, q_sx, rms_sx), (m_dx, q_dx, rms_dx), (x_int, y_int)
