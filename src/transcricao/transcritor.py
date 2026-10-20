"""Transcrição local com faster-whisper sobre as faixas gravadas, e gravação do resultado."""

import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from .falantes import (ExtratorVoz, Vozes, assinaturas_por_falante, diarizar, falante_em, modelos,
                       rotulos)
from .gravador import FAIXAS, TAXA

MODELOS = ["small", "medium", "large-v3-turbo"]

# Rótulo dos falantes quando a faixa é separada por voz: "Falante 1" no microfone, "Outros 1" no sistema.
PREFIXO = {"mic": "Falante", "sistema": "Outros"}

# Medição de 16/10/2026 num Ryzen 5 7535U, 1 h de áudio: large-v3-turbo 22 min, diarização 11 min.
# A barra de progresso reparte o tempo de cada faixa nessa proporção.
PESO_DIARIZACAO = 0.5

# Grupo de palavras mais curto que isto, no meio da fala de outra pessoa, volta para o falante anterior.
FALA_MINIMA = 0.6

# No teste de 16/10/2026 a troca de falante caiu de 1 a 3 palavras antes ou depois do ponto final certo.
# A troca anda até o ponto final mais próximo dentro desta distância, em palavras.
AJUSTE_PALAVRAS = 4
FIM_DE_FRASE = (".", "?", "!")

# Abaixo deste RMS o áudio é tratado como silêncio e não vai para o Whisper.
SILENCIO = 0.002

# Trecho do microfone com RMS abaixo desta fração do RMS do sistema, no mesmo horário, é eco do
# alto-falante. No teste de 16/10/2026 o eco ficou em 2 % (mic 0,0041 contra sistema 0,1831).
ECO_RAZAO = 0.1


@dataclass
class Segmento:
    inicio: float
    fim: float
    falante: str
    texto: str
    faixa: str = ""


def threads_cpu() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def faixa(pasta: Path, nome: str) -> Path | None:
    """Arquivo da faixa, em WAV ou FLAC."""
    for extensao in ("wav", "flac"):
        if (pasta / f"{nome}.{extensao}").exists():
            return pasta / f"{nome}.{extensao}"
    return None


def _ler(caminho: Path) -> np.ndarray:
    # O PyAV 19 recusa o argumento metadata_errors do faster-whisper 1.2.1; o array evita o PyAV.
    audio, taxa = sf.read(caminho, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if taxa != TAXA:
        raise ValueError(f"{caminho.name} está em {taxa} Hz; o app grava e lê só {TAXA} Hz.")
    return audio


def confiavel(parte) -> bool:
    """Critério de descarte do próprio Whisper para texto inventado ou repetido."""
    return parte.avg_logprob >= -1.0 and parte.compression_ratio <= 2.4


def _rms_trecho(pasta: Path, nome: str, inicio: float, fim: float, atraso: float) -> float:
    a = max(0, int((inicio - atraso) * TAXA))
    b = max(a + 1, int((fim - atraso) * TAXA))
    dados, _ = sf.read(faixa(pasta, nome), start=a, stop=b, dtype="float32")
    return float(np.sqrt(np.mean(dados**2))) if len(dados) else 0.0


def remover_eco(pasta: Path, segmentos: list[Segmento]) -> list[Segmento]:
    """Sem fone de ouvido, o microfone capta o alto-falante e repete a fala dos outros."""
    atraso = json.loads((pasta / "meta.json").read_text(encoding="utf-8")).get("atraso", {})
    tem_sistema = faixa(pasta, "sistema") is not None
    outros = [s for s in segmentos if s.faixa == "sistema"]
    resultado = []
    for s in segmentos:
        if s.faixa == "mic":
            texto_igual = any(
                o.inicio < s.fim + 1 and s.inicio < o.fim + 1
                and SequenceMatcher(None, s.texto.lower(), o.texto.lower()).ratio() > 0.6
                for o in outros
            )
            if texto_igual:
                continue
            if tem_sistema:
                mic = _rms_trecho(pasta, "mic", s.inicio, s.fim, atraso.get("mic", 0.0))
                sistema = _rms_trecho(pasta, "sistema", s.inicio, s.fim, atraso.get("sistema", 0.0))
                if mic < ECO_RAZAO * sistema:
                    continue
        resultado.append(s)
    return resultado


def _fim_de_frase(palavra) -> bool:
    return palavra[1].word.strip().endswith(FIM_DE_FRASE)


def _fragmento(palavras, anteriores) -> bool:
    """Grupo que não é fala de outra pessoa: até 2 palavras sem ponto final, ou mais curto que FALA_MINIMA.

    Resposta curta com ponto final depois de outra frase com ponto final ("inteiro? Pode.") não é fragmento.
    """
    if not palavras:
        return True
    if len(palavras) <= 2 and not _fim_de_frase(palavras[-1]):
        return True
    curto = palavras[-1][1].end - palavras[0][1].start < FALA_MINIMA
    frase_inteira = _fim_de_frase(palavras[-1]) and (not anteriores or _fim_de_frase(anteriores[-1]))
    return curto and not frase_inteira


def _ajustar_troca(anterior: list, seguinte: list):
    """Leva a troca de falante até o ponto final mais próximo, até AJUSTE_PALAVRAS palavras de distância."""
    if not anterior or not seguinte or _fim_de_frase(anterior[-1]):
        return
    # Ponto final no fim de `anterior`: as palavras depois dele passam para `seguinte`.
    for i in range(1, min(AJUSTE_PALAVRAS, len(anterior) - 1) + 1):
        if _fim_de_frase(anterior[-1 - i]):
            seguinte[:0] = anterior[-i:]
            del anterior[-i:]
            return
    # Ponto final no começo de `seguinte`: as palavras até ele passam para `anterior`.
    for i in range(min(AJUSTE_PALAVRAS, len(seguinte) - 1)):
        if _fim_de_frase(seguinte[i]):
            anterior.extend(seguinte[: i + 1])
            del seguinte[: i + 1]
            return


def _dividir(partes, turnos, nomes: dict[int, str], atraso: float, faixa: str) -> list[Segmento]:
    """Rotula a faixa palavra a palavra e quebra o texto onde o falante muda.

    Trabalha na faixa inteira, e não em cada segmento do Whisper, porque a troca de falante cai muitas
    vezes entre dois segmentos. Cada palavra vai como (índice do segmento, palavra).
    """
    grupos: list[list] = []  # [falante, palavras]
    for i, p in enumerate(partes):
        palavras = p.words or [SimpleNamespace(word=p.text, start=p.start, end=p.end)]
        for w in palavras:
            k = falante_em(turnos, w.start, w.end)
            if k is None and grupos:
                k = grupos[-1][0]
            if grupos and grupos[-1][0] == k:
                grupos[-1][1].append((i, w))
            else:
                grupos.append([k, [(i, w)]])
    juntos: list[list] = []
    for k, palavras in grupos:
        if juntos and (juntos[-1][0] == k or _fragmento(palavras, juntos[-1][1])):
            juntos[-1][1].extend(palavras)
        else:
            juntos.append([k, list(palavras)])
    if len(juntos) > 1 and _fragmento(juntos[0][1], []):
        juntos[1][1][:0] = juntos[0][1]
        del juntos[0]
    for anterior, seguinte in zip(juntos, juntos[1:]):
        _ajustar_troca(anterior[1], seguinte[1])

    # Um Segmento por falante e por segmento do Whisper, para o remover_eco comparar trechos curtos.
    resultado = []
    for k, palavras in juntos:
        rotulo = nomes.get(k, f"{PREFIXO[faixa]} ?")
        trecho: list = []
        for item in palavras + [(None, None)]:
            if trecho and item[0] != trecho[-1][0]:
                texto = "".join(w.word for _, w in trecho).strip()
                if texto:
                    resultado.append(Segmento(trecho[0][1].start + atraso, trecho[-1][1].end + atraso,
                                              rotulo, texto, faixa))
                trecho = []
            trecho.append(item)
    return resultado


def transcrever(pasta: Path, modelo: str, idioma: str = "pt", progresso=None, status=None,
                pessoas: dict[str, int] | None = None, arquivo_vozes: Path | None = None) -> list[Segmento]:
    """Transcreve as faixas; a faixa com mais de uma pessoa em `pessoas` é separada por voz.

    Grava falantes.json na pasta com a assinatura de voz de cada rótulo, para nomear depois.
    """
    from faster_whisper import WhisperModel

    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    pessoas = pessoas or {}
    if status:
        status(f"Carregando o modelo {modelo} (o primeiro uso baixa o modelo)...")
    whisper = WhisperModel(modelo, device="cpu", compute_type="int8", cpu_threads=threads_cpu())

    caminhos = [(nome, faixa(pasta, nome)) for nome in FAIXAS if faixa(pasta, nome)]
    peso = {nome: 1 + (PESO_DIARIZACAO if pessoas.get(nome, 1) > 1 else 0) for nome, _ in caminhos}
    total = sum(sf.info(c).duration * peso[nome] for nome, c in caminhos) or 1.0
    feito = 0.0
    segmentos: list[Segmento] = []
    assinaturas: dict[str, list[float]] = {}
    extrator = vozes = None
    for nome, caminho in caminhos:
        audio = _ler(caminho)
        duracao = len(audio) / TAXA
        separar = pessoas.get(nome, 1) > 1
        if len(audio) and float(np.sqrt(np.mean(audio**2))) > SILENCIO:
            if status:
                status(f"Transcrevendo {caminho.name} com {modelo}...")
            atraso = meta.get("atraso", {}).get(nome, 0.0)
            partes_aceitas = []
            partes, _ = whisper.transcribe(audio, language=idioma, vad_filter=True,
                                           condition_on_previous_text=False, word_timestamps=separar)
            for p in partes:
                if p.text.strip() and confiavel(p):
                    partes_aceitas.append(p)
                if progresso:
                    progresso((feito + p.end) / total)
            if not separar:
                segmentos += [Segmento(p.start + atraso, p.end + atraso, FAIXAS[nome], p.text.strip(), nome)
                              for p in partes_aceitas]
            else:
                if status:
                    status(f"Separando as vozes de {caminho.name} ({pessoas[nome]} pessoas)...")
                modelo_seg, modelo_voz = modelos(status)
                inicio_diar = feito + duracao
                turnos = diarizar(audio, pessoas[nome], modelo_seg, modelo_voz, threads_cpu(),
                                  progresso=lambda f: progresso and progresso(
                                      (inicio_diar + f * duracao * PESO_DIARIZACAO) / total))
                if extrator is None:
                    extrator = ExtratorVoz(modelo_voz, threads_cpu())
                    vozes = Vozes(arquivo_vozes) if arquivo_vozes else Vozes()
                por_falante = assinaturas_por_falante(audio, turnos, extrator)
                nomes = rotulos(por_falante, turnos, vozes, PREFIXO[nome])
                assinaturas.update({nomes[k]: [round(float(x), 5) for x in a] for k, a in por_falante.items()})
                segmentos += _dividir(partes_aceitas, turnos, nomes, atraso, nome)
        del audio
        feito += duracao * peso[nome]
        if progresso:
            progresso(feito / total)

    (pasta / "falantes.json").write_text(json.dumps(assinaturas, ensure_ascii=False), encoding="utf-8")
    segmentos.sort(key=lambda s: s.inicio)
    return remover_eco(pasta, segmentos)


def hora(segundos: float) -> str:
    s = int(segundos)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _blocos(segmentos: list[Segmento]):
    """Junta segmentos seguidos do mesmo falante num bloco só."""
    atual = None
    for s in segmentos:
        if atual and atual[1] == s.falante:
            atual[2].append(s.texto)
        else:
            if atual:
                yield atual
            atual = [s.inicio, s.falante, [s.texto]]
    if atual:
        yield atual


def salvar(pasta: Path, segmentos: list[Segmento], base: str = "transcricao", titulo: str = "Transcrição") -> Path:
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    linhas_txt, linhas_md = [], [f"# {titulo} — {meta['inicio'].replace('T', ' ')}", ""]
    for inicio, falante, textos in _blocos(segmentos):
        texto = " ".join(textos)
        linhas_txt.append(f"[{hora(inicio)}] {falante}: {texto}")
        linhas_md.append(f"**[{hora(inicio)}] {falante}:** {texto}")
        linhas_md.append("")
    if not segmentos:
        linhas_txt.append("(nenhuma fala detectada)")
        linhas_md.append("_(nenhuma fala detectada)_")
    (pasta / f"{base}.txt").write_text("\n".join(linhas_txt) + "\n", encoding="utf-8")
    (pasta / f"{base}.md").write_text("\n".join(linhas_md) + "\n", encoding="utf-8")
    return pasta / f"{base}.md"
