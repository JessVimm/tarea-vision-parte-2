import cv2
import numpy as np
from tkinter import Tk, filedialog


# -------------------------------------------------
# Seleccionar imagen
# -------------------------------------------------
def seleccionar_imagen():
    # Abre un cuadro de diálogo para que el usuario seleccione una imagen del sistema
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")]
    )


# -------------------------------------------------
# Clase principal
# -------------------------------------------------
class ContornoAF8:

    def __init__(self, ruta):
        # Inicializa la configuración y carga la imagen en escala de grises, aplicando binarización
        self.config = {
        "T": 0.3,
        "T_curvas": 0.1,         
        "alphas": [8, 4, 2, 1],
        "lambda_penalty": 1.0,
        "max_p": None,
        "max_q": None,
        "max_r": None
    }

        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Error al cargar imagen")

        # binarización
        _, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.img = binaria


    # -------------------------------------------------
    # 1. CONTORNO CON OPENCV
    # -------------------------------------------------
    def obtener_contorno(self):
        # Extrae el contorno externo más grande de la imagen binaria

        contours, _ = cv2.findContours(
            self.img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return None, 0

        cnt = max(contours, key=cv2.contourArea)
        contorno = cnt.reshape(-1, 2)

        return contorno, len(contorno)


    # -------------------------------------------------
    # 2. DIRECCIÓN F8
    # -------------------------------------------------
    def direccion(self, p1, p2):
        # Calcula la dirección entre dos puntos usando el esquema de 8 direcciones

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        dx = int(np.sign(dx))
        dy = int(np.sign(dy))

        direcciones = {
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
            (-1, 1): 3,
            (-1, 0): 4,
            (-1, -1): 5,
            (0, -1): 6,
            (1, -1): 7
        }

        return direcciones[(dx, dy)]


    # -------------------------------------------------
    # 3. F8  AF8
    # -------------------------------------------------
    def obtener_AF8(self, contorno):
        # Convierte el contorno en una cadena de direcciones y su representación diferencial (AF8)

        F8 = []

        for i in range(len(contorno)):
            p1 = contorno[i]
            p2 = contorno[(i + 1) % len(contorno)]
            F8.append(self.direccion(p1, p2))

        AF8 = []
        for i in range(len(F8)):
            diff = (F8[i] - F8[i - 1]) % 8
            AF8.append(diff)

        simbolos = ['a','b','c','d','e','f','g','h']
        cadena = [simbolos[d] for d in AF8]

        return cadena, AF8


    # -------------------------------------------------
    # 5. DETECTAR BREAK POINTS
    # -------------------------------------------------
    def detectar_breakpoints_greedy(self, AF8, max_p=None, max_q=None, max_r=None):
        # Detecta puntos de quiebre en la secuencia AF8 usando un enfoque voraz basado en patrones

        """
        AF8: lista de enteros en [0..7]
        max_p, max_q, max_r: límites opcionales (None = sin límite)
        """

        n = len(AF8)
        breakpoints = []

        if n == 0:
            return breakpoints

        i = 0
        start_global = None

        while True:

            if start_global is not None and i == start_global:
                break

            if start_global is None:
                start_global = i

            x = AF8[i]
            a = x
            b = (x + 1) % 8
            h = (x - 1) % 8

            j = 1

            p = 0
            while j < n:
                if AF8[(i + j) % n] != a:
                    break
                p += 1
                j += 1
                if max_p is not None and p >= max_p:
                    break

            def intentar_extender(j0, orient):
                # Intenta extender el patrón según la orientación dada (bh o hb)
                j_loc = j0
                r_loc = 0

                while j_loc + 1 < n:
                    s0 = AF8[(i + j_loc) % n]
                    s1 = AF8[(i + j_loc + 1) % n]

                    if orient == 'bh':
                        if not (s0 == b and s1 == h):
                            break
                    else:
                        if not (s0 == h and s1 == b):
                            break

                    j_loc += 2

                    q = 0
                    while j_loc < n and AF8[(i + j_loc) % n] == a:
                        q += 1
                        j_loc += 1
                        if max_q is not None and q >= max_q:
                            break

                    r_loc += 1
                    if max_r is not None and r_loc >= max_r:
                        break

                return j_loc, r_loc

            j_bh, r_bh = intentar_extender(j, 'bh')
            j_hb, r_hb = intentar_extender(j, 'hb')

            if j_bh > j_hb:
                j_final, r = j_bh, r_bh
            else:
                j_final, r = j_hb, r_hb

            if p > 0 or r > 0:
                breakpoints.append(i)
                avance = j_final if j_final > 0 else 1
                i = (i + avance) % n
            else:
                i = (i + 1) % n

            if len(breakpoints) > n:
                break

        return breakpoints
    

    def distancia_punto_segmento2(self, p, p1, p2):
        # Calcula la distancia al cuadrado de un punto a un segmento de línea

        x, y = p
        x1, y1 = p1
        x2, y2 = p2

        num = ((x - x1)*(y2 - y1) - (y - y1)*(x2 - x1))**2
        den = (x2 - x1)**2 + (y2 - y1)**2

        if den == 0:
            return 0

        return num / den
    

    def calcular_ISE(self, contorno, i, j):
        # Calcula el error cuadrático integrado entre dos puntos del contorno

        n = len(contorno)

        p1 = contorno[i]
        p2 = contorno[j]

        ISE = 0
        s = 0

        k = i

        while True:
            k = (k + 1) % n
            if k == j:
                break

            p = contorno[k]
            ISE += self.distancia_punto_segmento2(p, p1, p2)

            x1, y1 = contorno[k]
            x0, y0 = contorno[k - 1]
            s += np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        d = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        return ISE, s, d


    def refinar_breakpoints(self, contorno, breakpoints, T=None):
        # Refina los breakpoints dividiendo segmentos que no cumplen el criterio de error

        if T is None:
            T = self.config["T"]

        breakpoints = sorted(breakpoints)
        n = len(contorno)

        cambios = True

        while cambios:
            cambios = False
            nuevos_bp = []

            for i in range(len(breakpoints)):
                i1 = breakpoints[i]
                i2 = breakpoints[(i + 1) % len(breakpoints)]

                ISE, s, d = self.calcular_ISE(contorno, i1, i2)

                if d == 0:
                    continue

                criterio = ISE / (s * d) if s > 0 else 0

                if criterio >= T:

                    max_error = -1
                    max_idx = None

                    k = i1
                    while True:
                        k = (k + 1) % n
                        if k == i2:
                            break

                        p = contorno[k]
                        err = self.distancia_punto_segmento2(p, contorno[i1], contorno[i2])

                        if err > max_error:
                            max_error = err
                            max_idx = k

                    if max_idx is not None:
                        nuevos_bp.append(i1)
                        nuevos_bp.append(max_idx)
                        cambios = True
                    else:
                        nuevos_bp.append(i1)

                else:
                    nuevos_bp.append(i1)

            breakpoints = list(set(nuevos_bp))
            breakpoints.sort()

        return breakpoints
    

    def eliminar_puntos(self, contorno, breakpoints, T=None):
        # Elimina puntos innecesarios si el segmento resultante sigue cumpliendo el criterio

        if T is None:
            T = self.config["T"]

        n = len(contorno)
        bp = breakpoints.copy()

        i = 0

        while i < len(bp):

            if len(bp) <= 3:
                break

            i_prev = bp[i - 1]
            i_curr = bp[i]
            i_next = bp[(i + 1) % len(bp)]

            ISE, s, d = self.calcular_ISE(contorno, i_prev, i_next)

            if d == 0:
                i += 1
                continue

            criterio = ISE / (s * d) if s > 0 else 0

            if criterio < T:
                bp.pop(i)
            else:
                i += 1

        return bp
    

    def construir_rejilla(self, alpha):
        # Reduce la resolución de la imagen agrupando píxeles en bloques de tamaño alpha

        h, w = self.img.shape

        h2 = h // alpha
        w2 = w // alpha

        grid = np.zeros((h2, w2), dtype=np.uint8)

        for i in range(h2):
            for j in range(w2):
                bloque = self.img[
                    i*alpha:(i+1)*alpha,
                    j*alpha:(j+1)*alpha
                ]

                if np.any(bloque == 255):
                    grid[i, j] = 255

        return grid
    

    def reescalar_puntos(self, contorno_small, alpha):
        # Lleva los puntos de una rejilla reducida de vuelta a la escala original

        contorno_big = []

        for (x, y) in contorno_small:
            xb = int(x * alpha + alpha // 2)
            yb = int(y * alpha + alpha // 2)
            contorno_big.append((xb, yb))

        return np.array(contorno_big)
    

    def multiresolucion_rejilla(self, T=None):
        # Aplica el algoritmo en distintas resoluciones para encontrar una aproximación eficiente

        if T is None:
            T = self.config["T"]

        alphas = self.config["alphas"]
        alphas = [8, 4, 2, 1]

        for alpha in alphas:

            print(f"\nAlpha = {alpha}")

            img_small = self.construir_rejilla(alpha)

            if img_small.size == 0:
                continue

            temp = ContornoAF8.__new__(ContornoAF8)
            temp.img = img_small
            temp.h, temp.w = img_small.shape

            contorno_s, _ = temp.obtener_contorno()

            if contorno_s is None or len(contorno_s) < 3:
                continue

            _, AF8_num = temp.obtener_AF8(contorno_s)

            bps = temp.detectar_breakpoints_greedy(AF8_num)
            bps_ref = temp.refinar_breakpoints(contorno_s, bps, T)
            bps_final = temp.eliminar_puntos(contorno_s, bps_ref, T)

            contorno_big = self.reescalar_puntos(contorno_s, alpha)

            error_total = 0

            for i in range(len(bps_final)):
                i1 = bps_final[i]
                i2 = bps_final[(i + 1) % len(bps_final)]

                ISE, s, d = self.calcular_ISE(contorno_big, i1, i2)

                if d > 0 and s > 0:
                    error_total += ISE / (s * d)

            print("Error:", error_total)

            if error_total < T:
                print("Aceptado con alpha =", alpha)
                return contorno_big, bps_final

        print("fallback a original")

        contorno, _ = self.obtener_contorno()
        _, AF8 = self.obtener_AF8(contorno)

        bps = self.detectar_breakpoints_greedy(AF8)
        bps_ref = self.refinar_breakpoints(contorno, bps, T)
        bps_final = self.eliminar_puntos(contorno, bps_ref, T)
        contorno, bps_final = proc.multiresolucion_rejilla(T=None)

        return contorno, bps_final
    

    def calcular_ISE_total(self, contorno, breakpoints):
        # Suma el error total de todos los segmentos definidos por los breakpoints

        total_ISE = 0
        n = len(contorno)

        for i in range(len(breakpoints)):
            i1 = breakpoints[i]
            i2 = breakpoints[(i + 1) % len(breakpoints)]

            k = i1

            while True:
                k = (k + 1) % n
                if k == i2:
                    break

                p = contorno[k]
                total_ISE += self.distancia_punto_segmento2(
                    p,
                    contorno[i1],
                    contorno[i2]
                )

        return total_ISE
    

    def calcular_FOM(self, contorno, breakpoints):
        # Calcula una métrica de calidad basada en número de puntos y error total

        n = len(contorno)
        DP = len(breakpoints)

        ISE = self.calcular_ISE_total(contorno, breakpoints)

        if DP == 0 or ISE == 0:
            return 0

        FOM = n / (DP * ISE)

        return FOM, ISE, DP
    

    def obtener_poligono(self, contorno, breakpoints):
        # Construye el polígono final a partir de los breakpoints ordenados

        breakpoints = sorted(breakpoints)

        poligono = []

        for idx in breakpoints:
            x, y = contorno[idx]
            poligono.append((x, y))

        return np.array(poligono, dtype=np.int32)
    

    def dibujar_completo(self, contorno, breakpoints):
        # Dibuja el contorno original, los breakpoints y el polígono resultante

        img_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        contorno_cv = contorno.reshape(-1, 1, 2)
        cv2.drawContours(img_color, [contorno_cv], -1, (211, 211, 211), 1)

        for idx in breakpoints:
            x, y = contorno[idx]
            cv2.circle(img_color, (x, y), 4, (250, 128, 114), -1)

        poligono = self.obtener_poligono(contorno, breakpoints)

        for i in range(len(poligono)):
            x1, y1 = poligono[i]
            x2, y2 = poligono[(i + 1) % len(poligono)]

            cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 1)

        return img_color
    

    def shortest_path_dp(self, contorno, T=None, lambda_penalty=None):
        # Encuentra un camino óptimo usando programación dinámica considerando error y penalización

        if T is None:
            T = self.config["T"]

        if lambda_penalty is None:
            lambda_penalty = self.config["lambda_penalty"]

        n = len(contorno)

        dp = [float('inf')] * n
        prev = [-1] * n

        dp[0] = 0

        for i in range(n):

            if dp[i] == float('inf'):
                continue

            for j in range(i + 1, n):

                ISE, s, d = self.calcular_ISE(contorno, i, j)

                if d == 0 or s == 0:
                    continue

                criterio = ISE / (s * d)

                if criterio < T:

                    costo = ISE + lambda_penalty

                    if dp[i] + costo < dp[j]:
                        dp[j] = dp[i] + costo
                        prev[j] = i

        path = []
        j = n - 1

        while j != -1:
            path.append(j)
            j = prev[j]

        path.reverse()

        return path
    

    def shortest_path_circular(self, contorno, T=None):
        # Adapta el método de camino más corto para contornos cerrados

        contorno_ext = np.concatenate([contorno, contorno], axis=0)

        path = self.shortest_path_dp(contorno_ext, T)

        n = len(contorno)
        path = [p % n for p in path if p < 2*n]

        path = sorted(list(set(path)))

        return path

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    # Solicita al usuario una imagen desde el sistema
    ruta = seleccionar_imagen()
    if not ruta:
        exit()

    # Crea el objeto principal con la imagen seleccionada
    proc = ContornoAF8(ruta)

    # Obtiene el contorno inicial y su tamaño
    contorno, n = proc.obtener_contorno()

    # Guarda n en la configuración
    proc.config["n"] = n

    # Toma el valor actual del umbral
    T = proc.config["T"]

    # Ajusta el umbral manualmente
    proc.config["T"] = 0.1

    # Calcula límites para p, q y r en función del tamaño del contorno
    proc.config["max_p"] = int(0.02 * n / T)
    proc.config["max_q"] = int(0.005 * n / T)
    proc.config["max_r"] = int(0.01 * n / T)
        
    # 1. vuelve a obtener el contorno 
    contorno, n = proc.obtener_contorno()

    # 2. calcula la representación AF8 del contorno
    AF8_simbolos, AF8_num = proc.obtener_AF8(contorno)

    # 3. detecta breakpoints iniciales usando el método basado en CFG
    bps = proc.detectar_breakpoints_greedy(
        AF8_num,
        proc.config["max_p"],
        proc.config["max_q"],
        proc.config["max_r"]
    )

    # 4. refina los breakpoints dividiendo segmentos que no cumplen el criterio
    bps_ref = proc.refinar_breakpoints(contorno, bps)

    # 5. elimina puntos innecesarios manteniendo la calidad de aproximación
    bps_final = proc.eliminar_puntos(contorno, bps_ref)

    # (OPCIONAL) alternativa usando camino más corto en lugar de eliminación
    #bps_final = proc.shortest_path_circular(contorno)

    # 7. calcula métricas de calidad del resultado
    FOM, ISE, DP = proc.calcular_FOM(contorno, bps_final)

    # Muestra resultados en consola
    print("\n--- RESULTADOS ---")
    print("n:", n)
    print("DP:", DP)
    print("ISE:", ISE)
    print("FOM:", FOM)

    # 8. genera la imagen final con contorno, puntos y polígono
    img = proc.dibujar_completo(contorno, bps_final)

    # Muestra la imagen en una ventana
    cv2.imshow("Poligono final", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
