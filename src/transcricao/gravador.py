"""Captura do microfone e do áudio do sistema (WASAPI loopback) em duas faixas WAV."""

import json
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

TAXA = 16000
BLOCO = TAXA // 10

FAIXAS = {"mic": "Você", "sistema": "Outros"}


def dispositivos():
    mic = sc.default_microphone()
    alto_falante = sc.default_speaker()
    loopback = sc.get_microphone(id=str(alto_falante.name), include_loopback=True)
    return {"mic": mic, "sistema": loopback}


class Gravador:
    """Grava as faixas em disco e repassa cada bloco a `ao_receber(nome, dados)`."""

    def __init__(self, pasta: Path, ao_receber=None):
        self.pasta = pasta
        self.ao_receber = ao_receber
        self.niveis = {nome: 0.0 for nome in FAIXAS}
        self.falhas: dict[str, str] = {}
        self.atraso: dict[str, float] = {}
        self._parar = threading.Event()
        self._threads: list[threading.Thread] = []

    def iniciar(self):
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.inicio = datetime.now()
        self._salvar_meta()
        self._t0 = time.perf_counter()
        for nome, dispositivo in dispositivos().items():
            t = threading.Thread(target=self._gravar, args=(nome, dispositivo), daemon=True)
            t.start()
            self._threads.append(t)

    def _gravar(self, nome, dispositivo):
        caminho = self.pasta / f"{nome}.wav"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sf.SoundFile(caminho, "w", TAXA, 1, "PCM_16") as arquivo, \
                        dispositivo.recorder(samplerate=TAXA, channels=1, blocksize=BLOCO) as rec:
                    self.atraso[nome] = round(time.perf_counter() - self._t0, 3)
                    while not self._parar.is_set():
                        dados = rec.record(numframes=BLOCO)
                        if dados.ndim > 1:
                            dados = dados.mean(axis=1)
                        dados = dados.astype(np.float32)
                        arquivo.write(dados)
                        self.niveis[nome] = float(np.sqrt(np.mean(dados**2)))
                        if self.ao_receber:
                            self.ao_receber(nome, dados)
        except Exception as erro:
            self.falhas[nome] = str(erro)

    def _salvar_meta(self):
        meta = {
            "inicio": self.inicio.isoformat(timespec="seconds"),
            "taxa": TAXA,
            "atraso": self.atraso,
            "falhas": self.falhas,
        }
        (self.pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def parar(self) -> Path:
        self._parar.set()
        for t in self._threads:
            t.join(timeout=5)
        self._salvar_meta()
        return self.pasta
