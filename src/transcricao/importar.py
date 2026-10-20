"""Importação de um arquivo de áudio ou vídeo qualquer para uma pasta de gravação do app."""

import json
import re
from datetime import datetime
from pathlib import Path

import soundfile as sf

from .gravador import TAXA

EXTENSOES = ("*.mp3 *.m4a *.mp4 *.wav *.ogg *.opus *.flac *.webm *.aac *.wma *.mkv *.mov *.avi *.3gp *.amr")


def pasta_da_gravacao(arquivo: Path) -> Path | None:
    """Pasta criada pelo app, quando o arquivo escolhido é uma faixa ou o meta.json dela."""
    if arquivo.name in ("mic.wav", "sistema.wav", "mic.flac", "sistema.flac", "meta.json") \
            and (arquivo.parent / "meta.json").exists():
        return arquivo.parent
    return None


def importar(arquivo: Path, pasta_base: Path, progresso=None) -> Path:
    """Converte o áudio para mic.wav mono em 16 kHz numa pasta nova e grava o meta.json."""
    import av

    criado = datetime.fromtimestamp(arquivo.stat().st_mtime)
    nome = re.sub(r"[^\w\- ]", "", arquivo.stem).strip()[:40] or "audio"
    pasta = pasta_base / f"{criado:%Y-%m-%d_%H-%M-%S}_{nome}"
    pasta.mkdir(parents=True, exist_ok=True)

    with av.open(str(arquivo)) as entrada:
        if not entrada.streams.audio:
            raise ValueError(f"{arquivo.name} não tem faixa de áudio.")
        fluxo = entrada.streams.audio[0]
        duracao = entrada.duration / av.time_base if entrada.duration else 0.0
        conversor = av.AudioResampler(format="s16", layout="mono", rate=TAXA)
        with sf.SoundFile(pasta / "mic.wav", "w", TAXA, 1, "PCM_16") as saida:
            for quadro in entrada.decode(fluxo):
                for convertido in conversor.resample(quadro):
                    saida.write(convertido.to_ndarray().reshape(-1))
                if progresso and duracao and quadro.time is not None:
                    progresso(min(1.0, quadro.time / duracao))
            for convertido in conversor.resample(None):
                saida.write(convertido.to_ndarray().reshape(-1))

    meta = {"inicio": criado.isoformat(timespec="seconds"), "taxa": TAXA, "atraso": {}, "falhas": {},
            "origem": str(arquivo)}
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return pasta
