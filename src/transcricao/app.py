"""Janela do app: gravar, ver a transcrição ao vivo e revisar ao final."""

import bisect
import gc
import json
import os
import re
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

from .ao_vivo import TranscritorAoVivo
from .falantes import Vozes
from .gravador import Gravador
from .importar import EXTENSOES, importar, pasta_da_gravacao
from .transcritor import MODELOS, hora, remover_eco, salvar, transcrever

PASTA_BASE = Path(os.environ.get("TRANSCRICAO_PASTA", Path.home() / "Transcricoes"))
LOG = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Transcricao" / "transcricao.log"
IDIOMA = "pt"

# Rótulo que o app gera sozinho; nome diferente destes, dado pelo usuário, vira voz salva.
ROTULO_AUTOMATICO = re.compile(r"(Falante|Outros)( \d+| \?)?")

TITULOS = {"transcricao": "Transcrição", "transcricao_revisada": "Transcrição revisada"}


class App:
    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.eventos: queue.Queue = queue.Queue()
        self.gravador: Gravador | None = None
        self.ao_vivo: TranscritorAoVivo | None = None
        self.ocupado = False
        self.pasta: Path | None = None
        self.arquivo: Path | None = None
        self.segmentos = []
        self._inicio = 0.0
        self._ultima_pasta: Path | None = None

        raiz.title("Transcrição de reunião")
        raiz.geometry("820x600")
        raiz.minsize(640, 420)
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)

        botoes = ttk.Frame(raiz, padding=(10, 10, 10, 4))
        botoes.pack(fill="x")
        self.b_gravar = ttk.Button(botoes, text="● Gravar", command=self.gravar)
        self.b_parar = ttk.Button(botoes, text="■ Parar", command=self.parar, state="disabled")
        self.b_abrir = ttk.Button(botoes, text="Abrir pasta", command=self.abrir_pasta)
        self.b_anterior = ttk.Button(botoes, text="Transcrever arquivo de áudio...", command=self.transcrever_arquivo)
        self.b_nomear = ttk.Button(botoes, text="Nomear falantes...", command=self.nomear, state="disabled")
        for b in (self.b_gravar, self.b_parar, self.b_abrir, self.b_anterior, self.b_nomear):
            b.pack(side="left", padx=(0, 6))

        opcoes = ttk.Frame(raiz, padding=(10, 2))
        opcoes.pack(fill="x")
        ttk.Label(opcoes, text="Modelo ao vivo:").pack(side="left")
        self.modelo_vivo = ttk.Combobox(opcoes, values=MODELOS, width=15, state="readonly")
        self.modelo_vivo.set("small")
        self.modelo_vivo.pack(side="left", padx=(4, 16))
        self.revisar = tk.BooleanVar(value=True)
        ttk.Checkbutton(opcoes, text="Revisar ao final com", variable=self.revisar).pack(side="left")
        self.modelo_revisao = ttk.Combobox(opcoes, values=MODELOS, width=15, state="readonly")
        # Medição de 16/10/2026, 1 h de áudio num Ryzen 5 7535U: large-v3-turbo 22 min, medium 33 min.
        self.modelo_revisao.set("large-v3-turbo")
        self.modelo_revisao.pack(side="left", padx=4)

        pessoas = ttk.Frame(raiz, padding=(10, 2))
        pessoas.pack(fill="x")
        self.na_sala = tk.IntVar(value=1)
        self.no_teams = tk.IntVar(value=1)
        self.campos_pessoas = []
        for rotulo, var in (("Pessoas na sala (com você):", self.na_sala),
                            ("Pessoas do outro lado da chamada:", self.no_teams)):
            ttk.Label(pessoas, text=rotulo).pack(side="left")
            campo = ttk.Spinbox(pessoas, from_=1, to=10, width=4, textvariable=var, state="readonly")
            campo.pack(side="left", padx=(4, 16))
            self.campos_pessoas.append(campo)

        medidores = ttk.Frame(raiz, padding=(10, 6))
        medidores.pack(fill="x")
        self.niveis = {}
        for nome, rotulo in (("mic", "Microfone"), ("sistema", "Sistema")):
            ttk.Label(medidores, text=rotulo).pack(side="left")
            barra = ttk.Progressbar(medidores, length=110, maximum=100)
            barra.pack(side="left", padx=(4, 14))
            self.niveis[nome] = barra
        self.relogio = ttk.Label(medidores, text="00:00:00", font=("Segoe UI", 11, "bold"))
        self.relogio.pack(side="left", padx=(0, 14))
        self.fila = ttk.Label(medidores, text="")
        self.fila.pack(side="left")

        self.texto = ScrolledText(raiz, wrap="word", font=("Segoe UI", 10), state="disabled", padx=8, pady=6)
        self.texto.pack(fill="both", expand=True, padx=10, pady=4)

        rodape = ttk.Frame(raiz, padding=(10, 4, 10, 10))
        rodape.pack(fill="x")
        self.status = ttk.Label(rodape, text=f"Pronto. As gravações ficam em {PASTA_BASE}")
        self.status.pack(side="left", fill="x", expand=True)
        self.progresso = ttk.Progressbar(rodape, length=180, maximum=1.0)
        self.progresso.pack(side="right")

        self.raiz.after(100, self._atualizar)

    # Eventos das threads chegam pela fila e só a thread da janela mexe nos widgets.
    def _atualizar(self):
        while True:
            try:
                tipo, valor = self.eventos.get_nowait()
            except queue.Empty:
                break
            if tipo == "segmento":
                bisect.insort(self.segmentos, valor, key=lambda s: s.inicio)
                self._mostrar(self.segmentos)
            elif tipo == "transcricao":
                self.segmentos = list(valor)
                self._mostrar(self.segmentos)
            elif tipo == "renomear":
                self._mostrar(self.segmentos)
            elif tipo == "status":
                self.status.config(text=valor)
            elif tipo == "progresso":
                self.progresso["value"] = valor
            elif tipo == "falha":
                self.status.config(text=valor)
                messagebox.showerror("Falha", valor)
            elif tipo == "fim":
                self._ao_terminar(valor)
        if self.gravador:
            for nome, barra in self.niveis.items():
                barra["value"] = min(100, self.gravador.niveis[nome] ** 0.5 * 300)
            self.relogio.config(text=hora(time.perf_counter() - self._inicio))
        if self.ao_vivo:
            n = self.ao_vivo.pendentes
            self.fila.config(text=f"Fila: {n} bloco(s) de 30 s" if n else "Fila: em dia")
        self.raiz.after(100, self._atualizar)

    def _mostrar(self, segmentos):
        # As faixas chegam fora de ordem, então a caixa é redesenhada inteira em ordem de horário.
        self.texto.config(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.insert("end", "".join(f"[{hora(s.inicio)}] {s.falante}: {s.texto}\n" for s in segmentos))
        self.texto.see("end")
        self.texto.config(state="disabled")

    def _botoes(self, gravando: bool):
        estado = "disabled" if gravando or self.ocupado else "normal"
        self.b_gravar.config(state=estado)
        self.b_anterior.config(state=estado)
        self.b_parar.config(state="normal" if gravando else "disabled")
        self.modelo_vivo.config(state="disabled" if gravando or self.ocupado else "readonly")
        for campo in self.campos_pessoas:
            campo.config(state="disabled" if gravando or self.ocupado else "readonly")
        self.b_nomear.config(state="normal" if not gravando and not self.ocupado and self._falantes() else "disabled")

    def _pessoas(self) -> dict[str, int]:
        return {"mic": self.na_sala.get(), "sistema": self.no_teams.get()}

    def _falantes(self) -> dict:
        """Assinaturas de voz da revisão atual (falantes.json), só dos rótulos que estão na tela."""
        if not self.pasta or not self.segmentos:
            return {}
        try:
            dados = json.loads((self.pasta / "falantes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        na_tela = {s.falante for s in self.segmentos}
        return {r: v for r, v in dados.items() if r in na_tela}

    def gravar(self):
        self.pasta = PASTA_BASE / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.arquivo = None
        self.segmentos = []
        self._mostrar(self.segmentos)
        self.progresso["value"] = 0
        self.gravador = Gravador(self.pasta)
        self.ao_vivo = TranscritorAoVivo(
            self.modelo_vivo.get(), IDIOMA, self.gravador.atraso, self._pessoas(),
            ao_segmento=lambda s: self.eventos.put(("segmento", s)),
            ao_falhar=lambda m: self.eventos.put(("falha", m)),
            ao_renomear=lambda nome, trocas: self.eventos.put(("renomear", trocas)),
        )
        self.gravador.ao_receber = self.ao_vivo.alimentar
        try:
            self.gravador.iniciar()
        except Exception as erro:
            self.gravador = None
            self.ao_vivo.finalizar()
            self.ao_vivo = None
            messagebox.showerror("Falha", f"Falha ao abrir o áudio: {erro}")
            return
        self._inicio = time.perf_counter()
        self.status.config(text=f"Gravando ({self.pasta.name}). O texto aparece com 30 a 60 s de atraso.")
        self._botoes(gravando=True)

    def parar(self):
        self.ocupado = True
        self._botoes(gravando=False)
        gravador, ao_vivo = self.gravador, self.ao_vivo
        self.gravador = None
        revisar, modelo, pessoas = self.revisar.get(), self.modelo_revisao.get(), self._pessoas()
        for barra in self.niveis.values():
            barra["value"] = 0
        threading.Thread(target=self._finalizar, args=(gravador, ao_vivo, revisar, modelo, pessoas),
                         daemon=True).start()

    def _revisar(self, pasta: Path, modelo: str, pessoas: dict[str, int], evento):
        return transcrever(pasta, modelo, IDIOMA, progresso=lambda f: evento("progresso", f),
                           status=lambda m: evento("status", m), pessoas=pessoas)

    def _finalizar(self, gravador: Gravador, ao_vivo: TranscritorAoVivo, revisar: bool, modelo: str,
                   pessoas: dict[str, int]):
        evento = lambda tipo, valor: self.eventos.put((tipo, valor))
        try:
            evento("status", "Gravação encerrada. Transcrevendo o último trecho...")
            pasta = gravador.parar()
            for nome, msg in gravador.falhas.items():
                evento("falha", f"Falha na gravação de {nome}: {msg}")
            segmentos = remover_eco(pasta, ao_vivo.finalizar())
            self.ao_vivo = None
            evento("transcricao", segmentos)
            arquivo = salvar(pasta, segmentos)
            if revisar:
                gc.collect()
                evento("status", "Transcrição ao vivo salva. Revisando...")
                revisados = self._revisar(pasta, modelo, pessoas, evento)
                evento("transcricao", revisados)
                arquivo = salvar(pasta, revisados, base="transcricao_revisada",
                                 titulo=TITULOS["transcricao_revisada"])
            evento("fim", arquivo)
        except Exception as erro:
            evento("falha", f"Falha ao finalizar: {erro}")
            evento("fim", None)

    def transcrever_arquivo(self):
        """Transcreve uma gravação do app ou qualquer arquivo de áudio ou vídeo."""
        escolhido = filedialog.askopenfilename(
            initialdir=self._ultima_pasta or (PASTA_BASE if PASTA_BASE.exists() else Path.home()),
            title="Arquivo de áudio ou vídeo",
            filetypes=[("Áudio e vídeo", EXTENSOES), ("Todos os arquivos", "*.*")])
        if not escolhido:
            return
        arquivo = Path(escolhido)
        self._ultima_pasta = arquivo.parent
        pasta = pasta_da_gravacao(arquivo)
        self.pasta = pasta
        self.arquivo = None
        self.segmentos = []
        self._mostrar(self.segmentos)
        self.ocupado = True
        self._botoes(gravando=False)
        self.progresso["value"] = 0
        modelo = self.modelo_revisao.get()
        # Arquivo de fora vira uma faixa só, com todas as pessoas da reunião nela.
        pessoas = self._pessoas() if pasta else {"mic": self.na_sala.get() + self.no_teams.get()}

        def tarefa():
            nonlocal pasta
            evento = lambda tipo, valor: self.eventos.put((tipo, valor))
            try:
                if pasta is None:
                    evento("status", f"Convertendo {arquivo.name} para 16 kHz...")
                    pasta = importar(arquivo, PASTA_BASE, progresso=lambda f: evento("progresso", f))
                    self.pasta = pasta
                    evento("progresso", 0)
                base = "transcricao_revisada" if (pasta / "transcricao.md").exists() else "transcricao"
                segmentos = self._revisar(pasta, modelo, pessoas, evento)
                evento("transcricao", segmentos)
                evento("fim", salvar(pasta, segmentos, base=base, titulo=TITULOS[base]))
            except Exception as erro:
                evento("falha", f"Falha ao transcrever: {erro}")
                evento("fim", None)

        threading.Thread(target=tarefa, daemon=True).start()

    def _ao_terminar(self, arquivo: Path | None):
        self.ocupado = False
        self.arquivo = arquivo
        self._botoes(gravando=False)
        self.fila.config(text="")
        if arquivo:
            self.progresso["value"] = 1.0
            self.status.config(text=f"Pronto: {arquivo.parent.name}\\{arquivo.name}")

    def nomear(self):
        falantes = self._falantes()
        if not falantes or not self.arquivo:
            return
        ordem = list(dict.fromkeys(s.falante for s in self.segmentos if s.falante in falantes))
        janela = tk.Toplevel(self.raiz)
        janela.title("Nomear falantes")
        janela.transient(self.raiz)
        janela.grab_set()
        quadro = ttk.Frame(janela, padding=12)
        quadro.pack(fill="both", expand=True)
        ttk.Label(quadro, text="Dê um nome a cada voz. O app reconhece o nome nas próximas reuniões.\n"
                               "Use o mesmo nome em duas linhas para juntar duas vozes numa pessoa.",
                  foreground="#555").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        campos = {}
        for i, rotulo in enumerate(ordem, start=1):
            falas = [s for s in self.segmentos if s.falante == rotulo]
            minutos = sum(s.fim - s.inicio for s in falas) / 60
            exemplo = " / ".join(s.texto for s in sorted(falas, key=lambda s: len(s.texto), reverse=True)[:2])
            var = tk.StringVar(value=rotulo)
            ttk.Entry(quadro, textvariable=var, width=18).grid(row=i, column=0, sticky="nw", padx=(0, 10), pady=4)
            ttk.Label(quadro, text=f"{rotulo}, {minutos:.1f} min: \"{exemplo[:160]}\"", wraplength=460).grid(
                row=i, column=1, sticky="w", pady=4)
            campos[rotulo] = var

        def salvar_nomes():
            trocas = {r: v.get().strip() for r, v in campos.items() if v.get().strip() and v.get().strip() != r}
            janela.destroy()
            if trocas:
                self._renomear(trocas, falantes)

        rodape = ttk.Frame(quadro)
        rodape.grid(row=len(ordem) + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(rodape, text="Cancelar", command=janela.destroy).pack(side="right")
        ttk.Button(rodape, text="Salvar", command=salvar_nomes).pack(side="right", padx=6)

    def _renomear(self, trocas: dict[str, str], falantes: dict):
        for s in self.segmentos:
            s.falante = trocas.get(s.falante, s.falante)
        self._mostrar(self.segmentos)
        base = self.arquivo.stem
        self.arquivo = salvar(self.pasta, self.segmentos, base=base, titulo=TITULOS.get(base, "Transcrição"))
        # Duas vozes com o mesmo nome viram uma só: a assinatura é a soma normalizada pela Vozes.
        vozes, novos = Vozes(), {}
        for rotulo, assinatura in falantes.items():
            nome = trocas.get(rotulo, rotulo)
            v = np.array(assinatura, dtype=np.float32)
            novos[nome] = (novos[nome] + v) if nome in novos else v
            if rotulo in trocas and not ROTULO_AUTOMATICO.fullmatch(nome):
                vozes.guardar(nome, v)
        dados = {n: [round(float(x), 5) for x in v / np.linalg.norm(v)] for n, v in novos.items()}
        (self.pasta / "falantes.json").write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        self.status.config(text=f"Nomes salvos em {self.arquivo.name}. Vozes guardadas: {', '.join(vozes.vozes)}")
        self._botoes(gravando=False)

    def abrir_pasta(self):
        alvo = self.pasta or PASTA_BASE
        alvo.mkdir(parents=True, exist_ok=True)
        os.startfile(alvo)

    def fechar(self):
        if self.gravador:
            if not messagebox.askyesno("Gravação em curso", "Parar a gravação e fechar? O áudio fica salvo, a transcrição ao vivo se perde."):
                return
            self.gravador.parar()
        elif self.ocupado:
            if not messagebox.askyesno("Transcrição em curso", "Fechar agora interrompe a transcrição. Fechar mesmo assim?"):
                return
        self.raiz.destroy()


def _preparar_saida():
    """No .exe sem console, stdout e stderr valem None e o tqdm do download do modelo falha."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def main():
    _preparar_saida()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    raiz = tk.Tk()
    icone = Path(getattr(sys, "_MEIPASS", "")) / "icone.ico"
    if icone.exists():
        raiz.iconbitmap(default=str(icone))
    try:
        App(raiz)
    except Exception:
        traceback.print_exc()
        messagebox.showerror("Falha", f"Falha ao abrir o app. Detalhes em {LOG}")
        raise
    raiz.mainloop()
