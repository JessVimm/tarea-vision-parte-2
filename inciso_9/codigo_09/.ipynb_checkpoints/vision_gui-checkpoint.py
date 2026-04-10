import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from tkinter import messagebox
import cv2
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Contornos")

        # -----------------------------------------------
        # MENÚ SUPERIOR
        # -----------------------------------------------
        menubar = tk.Menu(root)

        # Menú Archivo
        menu_archivo = tk.Menu(menubar, tearoff=0)
        menu_archivo.add_command(label="Cargar imagen", command=self.cargar_imagen)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=root.quit)

        menubar.add_cascade(label="Archivo", menu=menu_archivo)

        # Menú Contorno
        menu_contorno = tk.Menu(menubar, tearoff=0)
        menu_contorno.add_command(label="Detectar contorno", command=self.detectar_contorno)

        menubar.add_cascade(label="Contorno", menu=menu_contorno)

        # Menú Cadenas
        menu_cadenas = tk.Menu(menubar, tearoff=0)
        menu_cadenas.add_command(label="F4", command=self.generar_f4)
        menu_cadenas.add_command(label="F8", command=self.generar_f8)
        menu_cadenas.add_command(label="AF8", command=self.generar_af8)
        menu_cadenas.add_command(label="VCC", command=self.generar_vcc)
        menu_cadenas.add_command(label="3OT", command=self.generar_3ot)

        menubar.add_cascade(label="Cadenas", menu=menu_cadenas)

        root.config(menu=menubar)

        # -----------------------------------------------
        # BARRA DE BOTONES DE CADENAS
        # -----------------------------------------------
        self.frame_cadenas = tk.Frame(root)
        self.frame_cadenas.pack(pady=5)

        self.btn_f4 = tk.Button(self.frame_cadenas, text="F4", width=8, command=self.generar_f4)
        self.btn_f4.pack(side=tk.LEFT, padx=5)

        self.btn_f8 = tk.Button(self.frame_cadenas, text="F8", width=8, command=self.generar_f8)
        self.btn_f8.pack(side=tk.LEFT, padx=5)

        self.btn_af8 = tk.Button(self.frame_cadenas, text="AF8", width=8, command=self.generar_af8)
        self.btn_af8.pack(side=tk.LEFT, padx=5)

        self.btn_vcc = tk.Button(self.frame_cadenas, text="VCC", width=8, command=self.generar_vcc)
        self.btn_vcc.pack(side=tk.LEFT, padx=5)

        self.btn_3ot = tk.Button(self.frame_cadenas, text="3OT", width=8, command=self.generar_3ot)
        self.btn_3ot.pack(side=tk.LEFT, padx=5)

        #-----------------------------------------------
        # ZONA IMAGEN
        #-----------------------------------------------
        self.drop_label = tk.Label(root, text="Pegue aquí la imagen binaria del objeto a analizar",
                                   width=50, height=5, bg="lightgray")
        self.drop_label.pack(pady=10)

        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop_imagen)

        self.label_img = tk.Label(root)
        self.label_img.pack()

        self.img = None
        self.img_binaria = None

        # -----------------------------------------------
        # ÁREA DE RESULTADOS
        # -----------------------------------------------
        frame_texto = tk.Frame(root)
        frame_texto.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = tk.Scrollbar(frame_texto)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_resultado = tk.Text(frame_texto, height=30, width=70, yscrollcommand=scrollbar.set)
        self.text_resultado.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.text_resultado.yview)

        # Fuente más clara
        self.text_resultado.config(font=("Consolas", 10))

        root.configure(bg="#f5f5f5")

        self.frame_cadenas.configure(bg="#f5f5f5")
        self.drop_label.configure(bg="#dcdcdc")

    #-----------------------------
    # VALIDAR IMAGEN
    #-----------------------------
    def validar_imagen(self):
        if self.img_binaria is None:
            messagebox.showwarning("Imagen requerida", "Primero carga una imagen.")
            print("Primero carga una imagen")
            return False
        return True


    # ----------------------------
    # CARGA IMAGEN
    # ----------------------------
    def procesar_ruta(self, ruta):
        try:
            img_pil = Image.open(ruta).convert("L")
            img = np.array(img_pil)

            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            self.img = img
            self.img_binaria = img_bin

            img_pil_bin = Image.fromarray(img_bin)
            self.img_tk = ImageTk.PhotoImage(img_pil_bin)

            self.label_img.config(image=self.img_tk)
            self.label_img.image = self.img_tk

        except Exception as e:
            print("Error:", e)

    def cargar_imagen(self):
        ruta = filedialog.askopenfilename()
        if ruta:
            self.procesar_ruta(ruta)

    def drop_imagen(self, event):
        ruta = event.data
        if ruta.startswith("{") and ruta.endswith("}"):
            ruta = ruta[1:-1]
        self.procesar_ruta(ruta)

    # ----------------------------
    # CONTORNO
    # ----------------------------
    def detectar_contorno(self):
        if not self.validar_imagen():
            return

        # Usamos la misma lógica que F8
        binary = self.img_binaria.copy()

        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            messagebox.showerror("Error", "No se encontraron contornos.")
            print("No se encontraron contornos")
            return

        # Tomar el contorno más grande
        cnt = max(contours, key=cv2.contourArea)

        # Reordenar para que empiece en top-left y sea horario
        cnt = self.rotate_contour_to_start(cnt)

        # Convertir imagen a color para dibujar
        img_color = cv2.cvtColor(self.img_binaria, cv2.COLOR_GRAY2BGR)

        # Dibujar contorno en rojo
        for i in range(len(cnt)):
            x, y = cnt[i][0]
            img_color[y, x] = [255, 0, 0]  # rojo (BGR)

        # Punto inicial
        x0, y0 = cnt[0][0]

        # Dibujar punto inicial grande y visible
        cv2.circle(img_color, (x0, y0), 4, (0, 255, 0), -1)  # verde

        # Mostrar imagen
        img_pil = Image.fromarray(img_color)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        self.label_img.config(image=self.img_tk)
        self.label_img.image = self.img_tk

        print(f"Contorno detectado con {len(cnt)} puntos")
        print(f"Punto inicial: ({x0}, {y0})")

    # ----------------------------
    # FUNCIONES AUXILIARES
    # ----------------------------
    # ----------------------------
    # F4
    # ----------------------------
    def find_start_point(self, binary):
        padded = np.pad(binary, 1, 'constant', constant_values=0)

        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if padded[y, x] == 255:
                    if padded[y - 1, x] == 0 or padded[y, x - 1] == 0:
                        return x, y
        return None

    def trace_boundary(self, binary):
        directions_4 = {
            (1, 0): 0,
            (0, 1): 1,
            (-1, 0): 2,
            (0, -1): 3
        }
        inv_dir_4 = {v: k for k, v in directions_4.items()}

        padded = np.pad(binary, 1, 'constant', constant_values=0)

        start_point = self.find_start_point(binary)
        if start_point is None:
            return []

        start_x, start_y = start_point

        vx, vy = start_x, start_y
        d = 0

        chain = []

        for _ in range(10000):
            dx, dy = inv_dir_4[d]
            vx += dx
            vy += dy

            chain.append(d)

            if (vx, vy) == (start_x, start_y) and len(chain) > 0:
                break

            d = (d + 3) % 4  # giro izquierda

            for _ in range(4):
                dx, dy = inv_dir_4[d]

                if d == 0:
                    px, py = vx, vy
                elif d == 1:
                    px, py = vx - 1, vy
                elif d == 2:
                    px, py = vx - 1, vy - 1
                else:
                    px, py = vx, vy - 1

                if padded[py, px] == 255:
                    break

                d = (d + 1) % 4

        return chain

    # ----------------------------
    # FUNCIONES AUXILIARES
    # ----------------------------

    def rotate_contour_to_start(self, contour):
        if contour.size == 0:
            return contour

        points = contour.reshape(-1, 2)

        # Top-left: menor y, luego menor x
        start_idx = min(range(len(points)), key=lambda i: (points[i][1], points[i][0]))

        rotated = np.concatenate((points[start_idx:], points[:start_idx]), axis=0)

        # Hacerlo horario
        rotated = rotated[::-1]

        return rotated.reshape((-1, 1, 2)).astype(contour.dtype)

    # ----------------------------
    # F4
    # ----------------------------
    #  3
    # 2 0
    #  1
    def generar_f4(self):
        if not self.validar_imagen():
            return

        cadena_f4 = self.trace_boundary(self.img_binaria)

        if not cadena_f4:
            messagebox.showerror("Error", "No se pudo generar la cadena F4.")
            print("No se pudo generar F4")
            return

        cadena_str = ''.join(map(str, cadena_f4))

        # Guardar cadena F4
        self.cadena_f4 = cadena_f4

        self.text_resultado.delete("1.0", tk.END)
        self.text_resultado.insert(tk.END, f"Cadena F4:\n{cadena_str}\n")
        self.text_resultado.insert(tk.END, f"\nLongitud: {len(cadena_f4)}")

    # ----------------------------
    # F8
    # ----------------------------
    # 5 6 7
    # 4   0
    # 3 2 1
    def generar_f8(self):
        if not self.validar_imagen():
            return

        # IMPORTANTE: invertir (igual que tu código de referencia)
        binary = self.img_binaria.copy()

        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            messagebox.showerror("Error", "No se encontraron contornos.")
            print("No se encontraron contornos.")
            return

        # Contorno más grande
        cnt = max(contours, key=cv2.contourArea)

        # Reordenar
        cnt = self.rotate_contour_to_start(cnt)

        # Direcciones F8 (igual que tu referencia)
        directions_8 = {
            (0, 1): 0,
            (1, 1): 1,
            (1, 0): 2,
            (1, -1): 3,
            (0, -1): 4,
            (-1, -1): 5,
            (-1, 0): 6,
            (-1, 1): 7
        }

        chain_code_8 = []

        for i in range(len(cnt)):
            curr = cnt[i][0]
            nxt = cnt[(i + 1) % len(cnt)][0]

            dy = nxt[1] - curr[1]
            dx = nxt[0] - curr[0]

            if (dy, dx) in directions_8:
                chain_code_8.append(directions_8[(dy, dx)])

        # Ajuste final (IMPORTANTE)
        if chain_code_8:
            chain_code_8 = chain_code_8[-1:] + chain_code_8[:-1]

        cadena_str = ''.join(map(str, chain_code_8))

        # Guardar para futuro uso
        self.cadena_f8 = chain_code_8

        self.text_resultado.delete("1.0", tk.END)
        self.text_resultado.insert(tk.END, f"Cadena F8:\n{cadena_str}\n")
        self.text_resultado.insert(tk.END, f"\nLongitud: {len(chain_code_8)}")

    # ----------------------------
    # AF8
    # ----------------------------
    def generar_af8(self):
        if not hasattr(self, 'cadena_f8'):
            messagebox.showwarning("Advertencia", "Primero genera F8.")
            print("Primero genera F8")
            return

        f8 = self.cadena_f8

        if len(f8) < 2:
            messagebox.showwarning("Advertencia", "Cadena F8 demasiado corta")
            print("Cadena F8 demasiado corta")
            return

        # Tabla (si no la pusiste en __init__)
        F8toAF8 = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5, (0, 6): 6, (0, 7): 7,
            (1, 0): 7, (1, 1): 0, (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 5, (1, 7): 6,
            (2, 0): 6, (2, 1): 7, (2, 2): 0, (2, 3): 1, (2, 4): 2, (2, 5): 3, (2, 6): 4, (2, 7): 5,
            (3, 0): 5, (3, 1): 6, (3, 2): 7, (3, 3): 0, (3, 4): 1, (3, 5): 2, (3, 6): 3, (3, 7): 4,
            (4, 0): 4, (4, 1): 5, (4, 2): 6, (4, 3): 7, (4, 4): 0, (4, 5): 1, (4, 6): 2, (4, 7): 3,
            (5, 0): 3, (5, 1): 4, (5, 2): 5, (5, 3): 6, (5, 4): 7, (5, 5): 0, (5, 6): 1, (5, 7): 2,
            (6, 0): 2, (6, 1): 3, (6, 2): 4, (6, 3): 5, (6, 4): 6, (6, 5): 7, (6, 6): 0, (6, 7): 1,
            (7, 0): 1, (7, 1): 2, (7, 2): 3, (7, 3): 4, (7, 4): 5, (7, 5): 6, (7, 6): 7, (7, 7): 0
        }

        af8 = []

        # Conversión
        for i in range(len(f8)):
            prev = f8[i - 1]  # circular
            curr = f8[i]

            af8.append(F8toAF8[(prev, curr)])

        cadena_str = ''.join(map(str, af8))

        self.text_resultado.delete("1.0", tk.END)
        self.text_resultado.insert(tk.END, f"Cadena AF8:\n{cadena_str}\n")
        self.text_resultado.insert(tk.END, f"\nLongitud: {len(af8)}")

    # ----------------------------
    # VCC
    # ----------------------------
    def generar_vcc(self):
        if not hasattr(self, 'cadena_f4'):
            messagebox.showwarning("Advertencia", "Primero genera F4.")
            print("Primero genera F4")
            return

        f4 = self.cadena_f4

        if len(f4) < 2:
            messagebox.showwarning("Advertencia", "Cadena F4 demasiado corta")
            print("Cadena F4 demasiado corta")
            return

        F4toVCC = {
            (0, 0): 0,
            (0, 1): 1,
            (0, 3): 2,
            (1, 0): 2,
            (1, 1): 0,
            (1, 2): 1,
            (2, 1): 2,
            (2, 2): 0,
            (2, 3): 1,
            (3, 0): 1,
            (3, 2): 2,
            (3, 3): 0
        }

        vcc = []

        # Conversión circular
        for i in range(len(f4)):
            prev = f4[i - 1]
            curr = f4[i]

            if (prev, curr) in F4toVCC:
                vcc.append(F4toVCC[(prev, curr)])
            else:
                # Por seguridad (no debería pasar)
                vcc.append(0)

        cadena_str = ''.join(map(str, vcc))

        self.text_resultado.delete("1.0", tk.END)
        self.text_resultado.insert(tk.END, f"Cadena VCC:\n{cadena_str}\n")
        self.text_resultado.insert(tk.END, f"\nLongitud: {len(vcc)}")

    # ----------------------------
    # 3OT
    # ----------------------------
    def generar_3ot(self):
        if not hasattr(self, 'cadena_f4'):
            messagebox.showwarning("Advertencia", "Primero genera F4.")
            print("Primero genera F4")
            return

        f4 = self.cadena_f4
        n = len(f4)

        if n < 2:
            messagebox.showwarning("Advertencia", "Cadena F4 demasiado corta")
            print("Cadena 3OT demasiado corta")
            return

        c3ot = []

        ref = f4[0]
        support = f4[0]
        primer_cambio_detectado = False

        # Recorrido principal
        for i in range(1, n):
            change = f4[i]

            if change == support:
                c3ot.append(0)
            else:
                if not primer_cambio_detectado:
                    c3ot.append(2)
                    primer_cambio_detectado = True
                elif change == ref:
                    c3ot.append(1)
                    ref = support
                elif (change - ref) % 4 == 2:
                    c3ot.append(2)
                    ref = support
                else:
                    c3ot.append(1)
                    ref = support

            support = change

        # Cierre circular
        change = f4[0]

        if change == support:
            c3ot.append(0)
        elif not primer_cambio_detectado:
            c3ot.append(2)
        elif change == ref:
            c3ot.append(1)
        elif (change - ref) % 4 == 2:
            c3ot.append(2)
        else:
            c3ot.append(1)

        cadena_str = ''.join(map(str, c3ot))

        self.text_resultado.delete("1.0", tk.END)
        self.text_resultado.insert(tk.END, f"Cadena 3OT:\n{cadena_str}\n")
        self.text_resultado.insert(tk.END, f"\nLongitud: {len(c3ot)}")

root = TkinterDnD.Tk()
app = App(root)
root.mainloop()