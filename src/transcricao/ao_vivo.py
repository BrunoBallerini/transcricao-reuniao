"""Transcrição durante a gravação: corta cada faixa em blocos num silêncio e transcreve em fila."""

import queue
import threading

import numpy as np
from faster_whisper import WhisperModel

from .falantes import AgrupadorAoVivo, ExtratorVoz, Vozes, modelos
from .gravador import FAIXAS, TAXA
from .transcritor import PREFIXO, SILENCIO, Segmento, confiavel, threads_cpu

# O corte cai no trecho de 100 ms mais silencioso entre MINIMO e MAXIMO segundos do bloco.
MINIMO = 20
MAXIMO = 30
JANELA = TAXA // 10


class TranscritorAoVivo:
    def __init__(self, modelo: str, idioma: str, atraso: dict[str, float], pessoas: dict[str, int] | None = None,
                 ao_segmento=None, ao_falhar=None, ao_renomear=None):
        self.modelo = modelo
        self.idioma = idioma
        self.atraso = atraso
        self.pessoas = pessoas or {}
        self.ao_segmento = ao_segmento
        self.ao_falhar = ao_falhar
        self.ao_renomear = ao_renomear
        self._agrupadores: dict[str, AgrupadorAoVivo] = {}
        self._da_faixa: dict[str, list[Segmento]] = {nome: [] for nome in FAIXAS}
        self.segmentos: list[Segmento] = []
        self._buffer = {nome: [] for nome in FAIXAS}
        self._tamanho = {nome: 0 for nome in FAIXAS}
        self._posicao = {nome: 0 for nome in FAIXAS}
        self._ultimo_texto = {nome: "" for nome in FAIXAS}
        self._fila: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._processar, daemon=True)
        self._worker.start()

    @property
    def pendentes(self) -> int:
        return self._fila.qsize()

    def alimentar(self, nome: str, dados: np.ndarray):
        self._buffer[nome].append(dados)
        self._tamanho[nome] += len(dados)
        if self._tamanho[nome] >= MAXIMO * TAXA:
            audio = np.concatenate(self._buffer[nome])
            corte = self._ponto_de_corte(audio)
            self._enfileirar(nome, audio[:corte])
            resto = audio[corte:]
            self._buffer[nome] = [resto]
            self._tamanho[nome] = len(resto)

    @staticmethod
    def _ponto_de_corte(audio: np.ndarray) -> int:
        trecho = audio[MINIMO * TAXA:MAXIMO * TAXA]
        n = len(trecho) // JANELA
        energia = (trecho[: n * JANELA].reshape(n, JANELA) ** 2).mean(axis=1)
        return MINIMO * TAXA + int(energia.argmin()) * JANELA + JANELA // 2

    def _enfileirar(self, nome: str, audio: np.ndarray):
        inicio = self._posicao[nome] / TAXA
        self._posicao[nome] += len(audio)
        if len(audio) and float(np.sqrt(np.mean(audio**2))) > SILENCIO:
            self._fila.put((nome, inicio, audio))

    def _processar(self):
        try:
            whisper = WhisperModel(self.modelo, device="cpu", compute_type="int8", cpu_threads=threads_cpu())
        except Exception as erro:
            if self.ao_falhar:
                self.ao_falhar(f"Falha ao carregar o modelo {self.modelo}: {erro}")
            whisper = None
        separar = {nome: n for nome, n in self.pessoas.items() if n > 1}
        if whisper is not None and separar:
            try:
                _, modelo_voz = modelos()
                extrator, vozes = ExtratorVoz(modelo_voz, 1), Vozes()
                self._agrupadores = {nome: AgrupadorAoVivo(extrator, vozes, n, PREFIXO[nome])
                                     for nome, n in separar.items()}
            except Exception as erro:
                if self.ao_falhar:
                    self.ao_falhar(f"Falha ao carregar o modelo de voz; o texto ao vivo sai sem separar falantes: {erro}")
        while True:
            item = self._fila.get()
            if item is None:
                break
            if whisper is None:
                continue
            nome, inicio, audio = item
            try:
                partes, _ = whisper.transcribe(
                    audio, language=self.idioma, vad_filter=True,
                    initial_prompt=self._ultimo_texto[nome] or None,
                )
                agrupador = self._agrupadores.get(nome)
                novos = []
                for p in partes:
                    texto = p.text.strip()
                    if not texto or not confiavel(p):
                        continue
                    deslocamento = inicio + self.atraso.get(nome, 0.0)
                    s = Segmento(p.start + deslocamento, p.end + deslocamento, FAIXAS[nome], texto, nome)
                    if agrupador:
                        agrupador.adicionar(audio[int(p.start * TAXA):int(p.end * TAXA)])
                        self._da_faixa[nome].append(s)
                        novos.append(s)
                        continue
                    self.segmentos.append(s)
                    self._ultimo_texto[nome] = texto[-200:]
                    if self.ao_segmento:
                        self.ao_segmento(s)
                if agrupador and novos:
                    mudou, ids_novos = False, {id(s) for s in novos}
                    for s, rotulo in zip(self._da_faixa[nome], agrupador.reagrupar()):
                        mudou |= id(s) not in ids_novos and s.falante != rotulo
                        s.falante = rotulo
                    if mudou and self.ao_renomear:
                        self.ao_renomear(nome, None)
                    for s in novos:
                        self.segmentos.append(s)
                        self._ultimo_texto[nome] = s.texto[-200:]
                        if self.ao_segmento:
                            self.ao_segmento(s)
            except Exception as erro:
                if self.ao_falhar:
                    self.ao_falhar(f"Falha ao transcrever um bloco de {nome}: {erro}")

    def finalizar(self) -> list[Segmento]:
        """Transcreve o resto dos buffers e espera a fila esvaziar."""
        for nome in FAIXAS:
            if self._buffer[nome]:
                self._enfileirar(nome, np.concatenate(self._buffer[nome]))
                self._buffer[nome] = []
        self._fila.put(None)
        self._worker.join()
        return sorted(self.segmentos, key=lambda s: s.inicio)
