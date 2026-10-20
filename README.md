# Transcrição de reunião

App de Windows que grava uma reunião e transcreve o áudio no próprio PC com
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Nenhum áudio sai da máquina e não há custo de API.

## O que faz

- Grava duas faixas: o microfone (`mic.wav`) e o áudio do sistema por WASAPI loopback (`sistema.wav`).
  O áudio do sistema capta Teams, Meet, Zoom ou qualquer som que sai no alto-falante.
- Transcreve durante a reunião, em blocos de cerca de 30 s cortados num silêncio. O texto aparece na
  janela com 30 a 60 s de atraso.
- Marca quem falou pela faixa: microfone vira "Você" e sistema vira "Outros".
- Separa as pessoas por voz quando os campos "Pessoas na sala (com você)" ou "Pessoas do outro lado da
  chamada" passam de 1. A faixa do microfone vira "Falante 1", "Falante 2"..., e a do sistema vira
  "Outros 1", "Outros 2"... Ao vivo, o rótulo é provisório: a cada bloco de 30 s, o app reagrupa todas as
  falas ouvidas até ali, e um rótulo antigo pode mudar. A revisão refaz a separação com a gravação inteira
  e quebra o texto na palavra em que o falante muda.
- "Nomear falantes..." troca os rótulos da revisão por nomes e regrava a transcrição. O app guarda a voz
  de cada nome em `%LOCALAPPDATA%\Transcricao\vozes.json` e usa esse nome nas reuniões seguintes. Dê o nome
  "Você" à sua voz para o app reconhecer você na sala.
- Descarta o eco: sem fone de ouvido, o microfone repete a fala dos outros. O app remove o trecho do
  microfone com texto igual ao do sistema no mesmo horário, ou com RMS abaixo de 10 % do RMS do sistema.
- Ao parar, grava `transcricao.md` e `transcricao.txt`. Com a opção "Revisar ao final" ligada, roda uma
  segunda passada com um modelo maior e grava `transcricao_revisada.md` e `.txt`.

## Instalar e abrir

Requisitos: Windows 10 ou 11, [git](https://git-scm.com/) e [uv](https://docs.astral.sh/uv/). O uv baixa o
Python 3.12 se faltar. Para instalar o uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Para abrir o app sem gerar o .exe:

```powershell
git clone https://github.com/BrunoBallerini/transcricao-reuniao.git
cd transcricao-reuniao
uv sync
uv run transcricao
```

No primeiro uso, o faster-whisper baixa o modelo do Hugging Face para `%USERPROFILE%\.cache\huggingface`.

## Gerar e instalar o .exe

Na pasta do repositório, depois do `uv sync`:

```powershell
powershell -ExecutionPolicy Bypass -File empacotar\empacotar.ps1
```

O script roda o PyInstaller em modo pasta, copia `dist\Transcricao` para
`%LOCALAPPDATA%\Programs\Transcricao` e cria o atalho "Transcrição de reunião" no Menu Iniciar e na área
de trabalho. Rode de novo depois de cada mudança no código, com o app fechado.

O `.exe` não abre console. A saída e as falhas vão para `%LOCALAPPDATA%\Transcricao\transcricao.log`.

## Onde ficam os arquivos

Cada gravação cria `%USERPROFILE%\Transcricoes\AAAA-MM-DD_HH-MM-SS\` com:

| Arquivo | Conteúdo |
|---|---|
| `mic.wav`, `sistema.wav` | Áudio 16 kHz mono, cerca de 115 MB por hora por faixa |
| `meta.json` | Hora de início e atraso de abertura de cada faixa |
| `transcricao.md`, `.txt` | Transcrição ao vivo |
| `transcricao_revisada.md`, `.txt` | Segunda passada, se ligada |
| `falantes.json` | Assinatura de voz de cada rótulo da revisão, lida por "Nomear falantes..." |

A variável de ambiente `TRANSCRICAO_PASTA` troca a pasta base.

"Transcrever arquivo de áudio..." roda o modelo de revisão sobre um arquivo escolhido:

- Uma faixa (`mic.wav`, `sistema.wav`) ou o `meta.json` de uma pasta do app: revisa a gravação dessa
  pasta. Use quando o app fechou no meio da reunião: o áudio fica salvo.
- Qualquer outro áudio ou vídeo (mp3, m4a, mp4, ogg, webm, wav e outros): o app converte para `mic.wav`
  mono em 16 kHz numa pasta nova e transcreve. A separação de falantes usa a soma dos dois campos de
  pessoas.

## Modelos

Os modelos rodam na CPU em int8, com metade dos núcleos lógicos. Velocidade medida em 16/10/2026 num
Ryzen 5 7535U (6 núcleos, 16 GB), com 202 s de fala em português:

| Modelo | Download | Velocidade | Uso |
|---|---|---|---|
| `small` | 460 MB | 6,8x tempo real | Padrão ao vivo |
| `medium` | 1,5 GB | 1,83x tempo real | Revisão, opcional |
| `large-v3-turbo` | 1,6 GB | 2,70x tempo real | Padrão da revisão |
| `large-v3` | 3 GB | 1,12x tempo real | Fora da lista: 54 min por hora de reunião |

Os três últimos foram medidos em 5 min de uma reunião presencial (16/10/2026). A revisão com `medium` de
uma reunião de 1h02 levou 31 min. O texto dos modelos não foi comparado com um gabarito; a escolha do
`large-v3-turbo` como padrão vem da velocidade.

A separação por voz usa o [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), com o modelo de
segmentação `pyannote-segmentation-3-0` e o modelo de voz `3dspeaker eres2net base`. O app baixa os dois
(47 MB) para `%LOCALAPPDATA%\Transcricao\modelos` no primeiro uso. Na reunião de 1h02, a separação levou
658 s (6x tempo real). Com separação, a revisão de 5 min levou 177 s: o `large-v3-turbo` com horário por
palavra e a separação somados (estimativa para 1 h: 35 min).

## Limites conhecidos

- O app usa o microfone e o alto-falante padrão do Windows no momento em que a gravação começa. Trocar de
  dispositivo no meio da reunião não muda a gravação.
- Com os campos de pessoas em 1, "Outros" junta todas as pessoas da chamada e "Você" é tudo o que o
  microfone capta, inclusive pessoas na mesma sala.
- A separação precisa do número de pessoas. Sem esse número, o limiar automático do sherpa-onnx criou
  cerca de 300 falantes na reunião de 16/10. Se uma pessoa da sala não fala, o rótulo extra fica com
  poucos segundos de fala, em geral interjeições.
- Duas pessoas falando ao mesmo tempo ficam num falante só. A troca de falante pode cair 1 palavra fora
  do ponto certo quando o Whisper não põe ponto final perto da troca.
- Fala baixa no microfone enquanto o sistema toca alto cai na regra de eco e sai da transcrição.
- O pacote `sherpa-onnx` do PyPI não traz as DLLs nativas no Windows; elas vêm do `sherpa-onnx-core`.
  Sem ele, o Windows carrega o `onnxruntime.dll` 1.17 do `System32`, e o sherpa-onnx falha com
  "The requested API version [28] is not available".
- O faster-whisper 1.2.1 falha ao abrir arquivo com o PyAV 19 (`metadata_errors`). O app lê o WAV com
  soundfile e passa o array ao modelo, sem o PyAV.
