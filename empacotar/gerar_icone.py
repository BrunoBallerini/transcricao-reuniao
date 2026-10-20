"""Desenha o ícone (bolinha vermelha de gravação num quadrado escuro) e grava icone.ico com PNG embutido."""

import struct
import sys
import tkinter as tk
from pathlib import Path

LADO = 256
saida = Path(sys.argv[1])
png = saida.with_suffix(".png")

raiz = tk.Tk()
raiz.withdraw()
img = tk.PhotoImage(width=LADO, height=LADO)
centro, raio, margem, canto = LADO / 2, 64, 16, 48
for y in range(LADO):
    linha = []
    for x in range(LADO):
        dx = max(margem + canto - x, x - (LADO - margem - canto), 0)
        dy = max(margem + canto - y, y - (LADO - margem - canto), 0)
        dentro_fundo = dx * dx + dy * dy <= canto * canto and margem <= x < LADO - margem and margem <= y < LADO - margem
        dentro_bola = (x - centro) ** 2 + (y - centro) ** 2 <= raio * raio
        anel = raio + 14 <= ((x - centro) ** 2 + (y - centro) ** 2) ** 0.5 <= raio + 24
        if dentro_bola:
            linha.append("#e5484d")
        elif anel and dentro_fundo:
            linha.append("#e8e8ec")
        elif dentro_fundo:
            linha.append("#1f2430")
        else:
            linha.append("#000000")
    img.put("{" + " ".join(linha) + "}", to=(0, y))
# Fundo fora do quadrado arredondado fica transparente.
for y in range(LADO):
    for x in range(LADO):
        if img.get(x, y) == (0, 0, 0):
            img.transparency_set(x, y, True)
img.write(str(png), format="png")
raiz.destroy()

dados = png.read_bytes()
cabecalho = struct.pack("<HHH", 0, 1, 1)
entrada = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(dados), 6 + 16)
saida.write_bytes(cabecalho + entrada + dados)
png.unlink()
print("icone:", saida)
