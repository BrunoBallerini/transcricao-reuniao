"""Separação de falantes (diarização) com sherpa-onnx e reconhecimento das vozes que o usuário nomeou."""

import json
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .gravador import TAXA

PASTA_DADOS = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Transcricao"
PASTA_MODELOS = PASTA_DADOS / "modelos"
ARQUIVO_VOZES = PASTA_DADOS / "vozes.json"

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
URL_SEGMENTACAO = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
URL_VOZ = f"{_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

# Cosseno entre assinaturas de voz, medido em 20:00-25:00 da reunião presencial de 16/10/2026.
# Trechos de 2 s: mesmo falante mediana 0,60, falantes diferentes mediana 0,24.
# Voz acumulada (1 min ou mais): mesmo falante 0,91 a 0,97, falantes diferentes 0,34.
LIMIAR_TRECHO = 0.40
LIMIAR_VOZ_SALVA = 0.60

# Trecho mais curto que isto não gera assinatura confiável e herda o falante anterior.
TRECHO_MINIMO = 1.0
# Para a assinatura de cada falante, a revisão usa até este tanto de fala, nos turnos mais longos.
AMOSTRA_FALANTE = 90


def modelos(status=None) -> tuple[Path, Path]:
    """Caminhos dos dois modelos; baixa no primeiro uso (cerca de 47 MB)."""
    segmentacao = PASTA_MODELOS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    voz = PASTA_MODELOS / Path(URL_VOZ).name
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    if not segmentacao.exists():
        if status:
            status("Baixando o modelo de segmentação de falantes (7 MB)...")
        pacote = PASTA_MODELOS / Path(URL_SEGMENTACAO).name
        urllib.request.urlretrieve(URL_SEGMENTACAO, pacote)
        with tarfile.open(pacote, "r:bz2") as tar:
            tar.extractall(PASTA_MODELOS, filter="data")
        pacote.unlink()
    if not voz.exists():
        if status:
            status("Baixando o modelo de voz (40 MB)...")
        parcial = voz.with_suffix(".parcial")
        urllib.request.urlretrieve(URL_VOZ, parcial)
        parcial.replace(voz)
    return segmentacao, voz


def _normalizar(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


class ExtratorVoz:
    def __init__(self, modelo: Path, threads: int):
        import sherpa_onnx

        self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(modelo), num_threads=threads))

    def assinatura(self, audio: np.ndarray) -> np.ndarray:
        fluxo = self._ext.create_stream()
        fluxo.accept_waveform(sample_rate=TAXA, waveform=audio)
        fluxo.input_finished()
        return _normalizar(np.array(self._ext.compute(fluxo), dtype=np.float32))


class Vozes:
    """Vozes que o usuário nomeou, em vozes.json, para rotular as próximas reuniões."""

    def __init__(self, arquivo: Path = ARQUIVO_VOZES):
        self.arquivo = arquivo
        try:
            dados = json.loads(arquivo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dados = {}
        self.vozes = {nome: np.array(v, dtype=np.float32) for nome, v in dados.items()}

    def guardar(self, nome: str, assinatura: np.ndarray):
        antiga = self.vozes.get(nome)
        self.vozes[nome] = _normalizar(assinatura if antiga is None else antiga + assinatura)
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        dados = {n: [round(float(x), 5) for x in v] for n, v in self.vozes.items()}
        self.arquivo.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")

    def reconhecer(self, assinaturas: dict[int, np.ndarray]) -> dict[int, str]:
        """Liga cada falante a no máximo uma voz salva, do par mais parecido para o menos parecido."""
        pares = sorted(
            ((float(a @ v), k, nome) for k, a in assinaturas.items() for nome, v in self.vozes.items()),
            reverse=True,
        )
        nomes: dict[int, str] = {}
        for sim, k, nome in pares:
            if sim >= LIMIAR_VOZ_SALVA and k not in nomes and nome not in nomes.values():
                nomes[k] = nome
        return nomes


def diarizar(audio: np.ndarray, falantes: int, modelo_segmentacao: Path, modelo_voz: Path,
             threads: int, progresso=None) -> list[tuple[float, float, int]]:
    """Turnos (início, fim, falante) com o número de falantes fixo."""
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(modelo_segmentacao)),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(modelo_voz), num_threads=threads),
        # Com o limiar automático, a reunião de 16/10 virou cerca de 300 falantes; por isso o número fixo.
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=falantes),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def retorno(feitos, total):
        if progresso:
            progresso(feitos / max(total, 1))
        return 0

    resultado = sd.process(audio, callback=retorno).sort_by_start_time()
    return [(t.start, t.end, t.speaker) for t in resultado]


def assinaturas_por_falante(audio: np.ndarray, turnos, extrator: ExtratorVoz) -> dict[int, np.ndarray]:
    por_falante: dict[int, list[tuple[float, float]]] = {}
    for inicio, fim, k in turnos:
        por_falante.setdefault(k, []).append((inicio, fim))
    resultado = {}
    for k, trechos in por_falante.items():
        trechos.sort(key=lambda t: t[1] - t[0], reverse=True)
        partes, soma = [], 0.0
        for inicio, fim in trechos:
            partes.append(audio[int(inicio * TAXA):int(fim * TAXA)])
            soma += fim - inicio
            if soma >= AMOSTRA_FALANTE:
                break
        resultado[k] = extrator.assinatura(np.concatenate(partes))
    return resultado


def rotulos(assinaturas: dict[int, np.ndarray], turnos, vozes: Vozes, prefixo: str) -> dict[int, str]:
    """Voz salva vira o nome dela; o resto vira "<prefixo> N", numerado pela ordem em que fala."""
    nomes = vozes.reconhecer(assinaturas)
    n = 0
    for _, _, k in turnos:
        if k not in nomes:
            n += 1
            nomes[k] = f"{prefixo} {n}"
    return nomes


def falante_em(turnos, inicio: float, fim: float) -> int | None:
    """Falante com mais sobreposição no intervalo; sem sobreposição, o turno mais próximo até 1 s."""
    melhor, maior = None, 0.0
    perto, distancia = None, 1.0
    for a, b, k in turnos:
        if a > fim + 1.0:
            break
        sobra = min(fim, b) - max(inicio, a)
        if sobra > maior:
            melhor, maior = k, sobra
        d = max(a - fim, inicio - b)
        if d < distancia:
            perto, distancia = k, d
    return melhor if melhor is not None else perto


def agrupar(assinaturas: np.ndarray, falantes: int) -> list[int]:
    """Average linkage pelo cosseno: junta grupos enquanto há mais que `falantes` ou a semelhança é alta."""
    n = len(assinaturas)
    if n == 0:
        return []
    soma = (assinaturas @ assinaturas.T).astype(np.float64)
    media = soma.copy()
    np.fill_diagonal(media, -np.inf)
    tamanho = np.ones(n)
    ativo = np.ones(n, dtype=bool)
    dono = np.arange(n)
    for restantes in range(n, 1, -1):
        i, j = np.unravel_index(int(np.argmax(media)), media.shape)
        if restantes <= falantes and media[i, j] < LIMIAR_TRECHO:
            break
        soma[i, :] += soma[j, :]
        soma[:, i] += soma[:, j]
        tamanho[i] += tamanho[j]
        ativo[j] = False
        dono[dono == j] = i
        linha = np.where(ativo, soma[i, :] / (tamanho[i] * tamanho), -np.inf)
        linha[i] = -np.inf
        media[i, :] = linha
        media[:, i] = linha
        media[j, :] = -np.inf
        media[:, j] = -np.inf
    return dono.tolist()


class AgrupadorAoVivo:
    """Rótulo provisório durante a gravação.

    A cada bloco, reagrupa todas as falas ouvidas até ali. Um grupo que nasce de um trecho com duas vozes
    não prende o resto da reunião, como acontecia comparando cada trecho só com os grupos anteriores.
    """

    def __init__(self, extrator: ExtratorVoz, vozes: Vozes, falantes: int, prefixo: str):
        self.extrator = extrator
        self.vozes = vozes
        self.falantes = falantes
        self.prefixo = prefixo
        self.assinaturas: list[np.ndarray | None] = []  # None: trecho curto, herda o falante anterior
        self.rotulos: list[str] = []

    def adicionar(self, audio: np.ndarray) -> int:
        """Guarda a assinatura do trecho e devolve o índice dele; o rótulo sai em `reagrupar`."""
        longo = len(audio) >= TRECHO_MINIMO * TAXA
        self.assinaturas.append(self.extrator.assinatura(audio) if longo else None)
        self.rotulos.append(self.rotulos[-1] if self.rotulos else f"{self.prefixo} 1")
        return len(self.assinaturas) - 1

    def reagrupar(self) -> list[str]:
        indices = [i for i, a in enumerate(self.assinaturas) if a is not None]
        if not indices:
            return self.rotulos
        matriz = np.stack([self.assinaturas[i] for i in indices])
        grupo_de = dict(zip(indices, agrupar(matriz, self.falantes)))
        grupos = sorted(set(grupo_de.values()), key=lambda g: min(i for i, x in grupo_de.items() if x == g))

        # Cada grupo herda o rótulo que a maioria dos seus trechos já tinha, para o texto na tela não trocar.
        nomes: dict[int, str] = self.vozes.reconhecer(
            {g: _normalizar(sum(matriz[k] for k, i in enumerate(indices) if grupo_de[i] == g)) for g in grupos})
        livres = [g for g in grupos if g not in nomes]
        for g in livres:
            anteriores = [self.rotulos[i] for i, x in grupo_de.items() if x == g]
            for rotulo in sorted(set(anteriores), key=anteriores.count, reverse=True):
                if rotulo not in nomes.values():
                    nomes[g] = rotulo
                    break
        n = 0
        for g in grupos:
            while g not in nomes:
                n += 1
                if f"{self.prefixo} {n}" not in nomes.values():
                    nomes[g] = f"{self.prefixo} {n}"
        ultimo = None
        for i in range(len(self.rotulos)):
            ultimo = nomes[grupo_de[i]] if i in grupo_de else (ultimo or self.rotulos[i])
            self.rotulos[i] = ultimo
        return self.rotulos
