# Testbed de Detecção Facial — projeto HyperFace

App web local para testar padrões de camuflagem anti-detecção facial. Você envia
uma imagem, ela passa por nove algoritmos de detecção em paralelo, e o app mostra
onde cada um encontrou rostos, com que confiança, e — o que realmente importa —
**quantas faces falsas o seu padrão conseguiu disparar**.

Roda inteiramente na sua máquina, sem custo, sem conta e sem chave de API. Depois
do setup inicial funciona offline.

---

## Por que a métrica é "faces falsas" e não "achou o rosto"

O [HyperFace](https://adam.harvey.studio/hyperface/), de Adam Harvey, não tenta
esconder o rosto. Ele faz o oposto: enche o entorno de faces falsas para que o
detector prefira uma delas. Nas palavras do próprio Harvey, o padrão

> "offers a higher confidence score for a nearby false face by exploiting a
> common algorithmic preference for the highest confidence facial region".

Ou seja: sistemas que consomem um detector quase sempre pegam **a maior** caixa
ou **a mais confiante** e ignoram o resto. Se uma face falsa ganhar nesse
critério, o rosto verdadeiro nunca chega à etapa de reconhecimento. É isso que
este testbed mede.

Vale registrar o que a própria página do Harvey admite: os padrões publicados
valem **somente** para o Viola-Jones / Haar Cascade e nunca foram testados
contra outros algoritmos. Ele encerra dizendo que o valor do padrão como
camuflagem "was temporary and has now passed". Este app existe justamente para
você verificar isso por conta própria, com números.

---

## Instalação

Requer Windows com Python 3.11 ou mais recente. Testado em Python 3.14.

```powershell
.\setup.ps1     # cria o ambiente, instala dependências e baixa os modelos
.\run.ps1       # inicia o servidor e abre o navegador em http://127.0.0.1:8010
```

Variações úteis:

```powershell
.\setup.ps1 -Core -SkipModels   # instalação mínima, só Haar/LBP/YuNet
.\run.ps1 -Port 9000 -NoBrowser
.\run.ps1 -Reload               # recarrega ao editar o código
```

### Onde ficam o ambiente e os modelos

Em `%LOCALAPPDATA%\HyperFaceTestbed`, **fora** da pasta do projeto. O projeto
vive numa pasta do Google Drive, e criar um venv ali levou 4,5 minutos contra
poucos segundos em disco local — são milhares de arquivos pequenos, o pior caso
possível para uma pasta sincronizada. Os modelos são sempre re-baixáveis, então
não faz sentido ocupar cota de sincronização com eles.

Para escolher outro diretório, defina `HFT_HOME` antes de rodar o setup.

---

## Os nove detectores

Organizados em camadas pelo peso das dependências. Se algum extra não instalar,
o app continua funcionando: o detector aparece como indisponível na interface,
com o motivo, e os demais rodam normalmente.

| Detector | Camada | O que representa |
|---|---|---|
| **Haar Cascade (Viola-Jones)** | 1 | O alvo original do HyperFace. Cinco perfis XML selecionáveis, incluindo o `frontalface_default` que Harvey atacou. |
| **LBP Cascade** | 1 | Mesma arquitetura, features mais baratas. Era o que rodava em hardware embarcado e câmeras de baixo custo. |
| **YuNet** | 1 | Detector DNN oficial do OpenCV desde a 4.5.4. Menos de 100 mil parâmetros; a escolha natural para vídeo em tempo real sem GPU. |
| **SCRFD-10GF** | 2 | O detector do pacote `buffalo_l` do InsightFace. É a referência livre mais próxima do estado da arte usado em sistemas de vigilância reais. |
| **SCRFD-500MF** | 2 | Variante enxuta do SCRFD (500 MFLOPs contra 10 GFLOPs), para checar se um modelo menor é mais fácil de enganar. |
| **UltraFace RFB** | 2 | Rede de ~1 MB para dispositivos de borda. Representa o detector que roda dentro da própria câmera. |
| **dlib HOG + SVM** | 3 | Harvey cita explicitamente os detectores "HoG/SVM-based edge" como o segundo alvo possível de um padrão HyperFace. |
| **dlib CNN (MMOD)** | 3 | CNN treinada com Max-Margin Object Detection. Robusta, mas lenta na CPU. |
| **BlazeFace (MediaPipe)** | 3 | Detector do Google usado em produtos de consumo com câmera em tempo real. |

### Sobre "os algoritmos usados em sistemas de segurança reais"

Os líderes do [NIST FRTE](https://pages.nist.gov/frvt/html/frvt11.html) — Idemia,
NEC, SenseTime, Megvii, Clearview — são todos **proprietários e caixa-preta**.
Os fornecedores submetem bibliotecas compiladas; não há como rodá-los aqui, a
qualquer preço.

O que existe de livre e mais próximo deles é o SCRFD e o RetinaFace para
detecção. Por isso o SCRFD-10GF é o detector mais importante da lista depois do
Haar: se o seu padrão passar por ele, passou pelo que há de melhor fora dos
sistemas fechados.

---

## Como usar

1. **Envie a imagem.** Arraste na área de upload, clique para escolher, ou
   simplesmente cole com Ctrl+V. Há também um link para carregar o padrão
   sintético de exemplo que vem junto.

2. **Escolha o preset.** O preset **HyperFace (Harvey, 2017)** reproduz a
   configuração que ele descreve: `scaleFactor` 1.05, `minSize` 24 e
   `minNeighbors` baixo. Os outros presets existem para comparação: *OpenCV
   padrão* é o que um sistema desatento rodaria, *CCTV* aproxima uma câmera de
   vigilância, *Permissivo* mostra toda janela aceita pelo cascade.

3. **Selecione os detectores** e ajuste parâmetros individuais abrindo cada um
   na lista. O atalho *rápidos* deixa os dois dlib de fora, que são ordens de
   magnitude mais lentos que o resto.

4. **Clique em Detectar.** As caixas aparecem sobre a imagem, com cor por
   detector e espessura proporcional à confiança — a convenção que o próprio
   Harvey usa nas imagens dele, borda grossa para detecção forte e fina para
   candidata fraca. Passe o mouse sobre qualquer caixa para ver os números.

5. **Marque o rosto verdadeiro.** Ative *Marcar rosto verdadeiro* e arraste um
   retângulo sobre o rosto real. É esse passo que liga as métricas: cada caixa
   passa a ser classificada como rosto real ou face falsa.

6. **Exporte.** Gera um JSON reprodutível (com todos os parâmetros usados) e o
   PNG anotado em resolução original, dentro de `data/results/`, com timestamp.
   É assim que você compara versões do padrão entre si.

---

## Como ler as métricas

| Métrica | O que significa |
|---|---|
| **faces falsas** | Caixas que não têm sobreposição relevante com o rosto que você marcou. É a contagem crua de iscas que o padrão gerou. |
| **score do rosto real** | A confiança da melhor caixa que acertou o rosto verdadeiro. |
| **maior score falso** | A confiança da isca mais forte. |
| **decoy_ratio** | Score falso máximo ÷ score do rosto real. Acima de 1 significa que a isca é mais convincente que o rosto de verdade. |
| **a maior caixa é falsa** | Um sistema que usa "a maior face" seria enganado. |
| **a caixa mais confiante é falsa** | Um sistema que usa "a face mais confiante" seria enganado. |

O veredito no canto de cada cartão resume os dois últimos:

- **padrão venceu** — a maior caixa ou a mais confiante é falsa
- **ruído gerado** — houve faces falsas, mas o rosto verdadeiro ainda lidera
- **detector resistiu** — nenhuma face falsa, e o rosto verdadeiro foi achado
- **nada detectado** — o detector não achou nem o rosto real; não diz nada
  sobre o padrão, só que ele não está enxergando essa imagem

O **histograma** mostra a distribuição dos scores brutos, separando reais de
falsas. Para o Haar são os `levelWeights`, e é ali que dá para ver se as iscas
estão concentradas na faixa fraca ou se já invadiram a faixa do rosto real.

O controle de **IoU** ajusta o quão tolerante é o casamento entre uma caixa
detectada e o rosto que você marcou. O padrão de 0.30 é permissivo de propósito:
detectores enquadram o rosto de formas bem diferentes entre si.

---

## Validação

O comando `python -m tools.validate` monta o cenário completo — um rosto real no
centro, cercado de faces falsas — e roda tudo pela API HTTP, do upload às
métricas, exatamente como a interface faz. Resultado com o preset HyperFace,
numa imagem de 1400x1000:

| Detector | Caixas | Falsas | Score real | Score falso | decoy | Veredito |
|---|---:|---:|---:|---:|---:|---|
| Haar Cascade | 51 | **50** | 1.000 | 1.000 | 1.00 | **padrão venceu** |
| dlib HOG + SVM | 26 | **25** | 0.747 | 0.800 | **1.07** | **padrão venceu** |
| YuNet | 23 | **22** | 0.845 | 0.882 | **1.04** | **padrão venceu** |
| LBP Cascade | 19 | **18** | 0.871 | 0.871 | 1.00 | **padrão venceu** |
| UltraFace RFB | 11 | 10 | 0.999 | 0.919 | 0.92 | ruído gerado |
| SCRFD-10GF | 1 | **0** | 0.848 | — | — | detector resistiu |
| SCRFD-500MF | 1 | **0** | 0.767 | — | — | detector resistiu |
| dlib CNN (MMOD) | 1 | **0** | 0.741 | — | — | detector resistiu |
| BlazeFace | 0 | 0 | — | — | — | nada detectado |

Os três detectores marcados com `decoy` acima de 1.00 são os casos em que o
ataque de fato funcionou no sentido estrito do HyperFace: a isca ficou **mais
convincente que o rosto de verdade**. Um sistema que escolhe a face mais
confiante e segue em frente entregaria a isca para a etapa de reconhecimento.

O BlazeFace não achou nem o rosto verdadeiro, o que é esperado: o modelo
`blaze_face_short_range` é feito para um rosto grande e centralizado, como numa
selfie, e aqui o rosto ocupa uma fração pequena do quadro. Isso não é resistência
ao padrão, e o veredito diz isso explicitamente.

Esse resultado confirma a tese do projeto e revela uma nuance que vale para o
seu trabalho:
a divisão não é simplesmente "clássico contra CNN". O **YuNet é uma rede
convolucional e mesmo assim foi vencido**, com a isca mais confiante que o rosto
real — provavelmente porque é minúsculo (menos de 100 mil parâmetros) e
otimizado para velocidade. Já o SCRFD-10GF, com muito mais capacidade, ignorou
o padrão por completo. A fronteira real parece ser a capacidade do modelo, não a
família do algoritmo, e isso importa muito na prática: YuNet é exatamente o tipo
de detector que roda em câmera de vigilância barata, justamente onde um padrão
de camuflagem teria mais utilidade.

O padrão sintético explora o mesmo princípio do HyperFace: o Haar decide
olhando diferenças de intensidade entre regiões retangulares vizinhas (olhos
mais escuros que as bochechas, ponte do nariz mais clara que os olhos). Uma
mancha abstrata que reproduza essas relações de luminância ativa o cascade sem
se parecer com um rosto nem para um humano nem para uma rede profunda, que
aprendeu textura facial de verdade.

---

## Ferramentas de linha de comando

```powershell
# Validação ponta a ponta pela API (precisa do servidor rodando)
python -m tools.validate

# Roda todos os detectores numa imagem e imprime uma tabela
python -m tools.smoke_test caminho\da\imagem.png

# Gera um padrão sintético de faces falsas
python -m tools.make_pattern saida.png --faces 60 --size 1600x1200 --seed 42

# Baixa ou re-baixa os modelos
python -m app.downloader
```

---

## Notas técnicas

**O `levelWeight` é o que torna a avaliação possível.** O Haar Cascade tem sim um
score de confiança, só não pelo método que todo mundo usa. Em vez de
`detectMultiScale`, o app chama:

```python
rects, reject_levels, level_weights = cascade.detectMultiScale3(
    gray, scaleFactor=1.05, minNeighbors=1, minSize=(24, 24),
    outputRejectLevels=True,
)
```

`level_weights[i]` é a certeza acumulada no estágio final do cascade. Sem ele
todas as detecções pareceriam igualmente fortes e não daria para dizer se uma
face falsa venceu a verdadeira. O app também faz uma segunda passagem com
`detectMultiScale2` para obter o número de janelas vizinhas que concordaram com
cada caixa — as "overlaps" que Harvey menciona.

**O OpenCV 5 mudou duas coisas que afetaram o projeto.** Os XMLs dos cascades
deixaram de vir empacotados (`cv2/data` está vazio), então o app baixa os
arquivos originais do repositório do OpenCV. E o suporte a Caffe foi removido
por completo (`readNetFromCaffe` não existe mais), o que inviabilizou o clássico
SSD ResNet-10; o papel dele na comparação ficou com o YuNet e o UltraFace, que
são ONNX.

**O SCRFD é implementado à mão sobre onnxruntime**, em vez de usar o pacote
`insightface`. Além de evitar dependências pesadas e problemas de suporte a
Python 3.14, dá acesso direto ao score bruto de cada âncora, que é exatamente o
número de que as métricas precisam. A topologia do modelo (níveis de pirâmide,
número de âncoras, presença de landmarks) é descoberta pelo número de tensores
de saída.

**Os scores são normalizados para 0..1** com uma função logística quando o
algoritmo devolve um valor sem escala fixa, como o `levelWeight` do Haar ou o
score de SVM do dlib. A logística é monotônica, então não altera o ranking das
detecções — serve só para permitir comparar detectores no mesmo gráfico. O valor
nativo fica preservado no campo `raw_score` e aparece no tooltip e no JSON.

---

## Licenças dos modelos

O código deste app é livre para uso, mas os pesos têm licenças próprias:

- **YuNet** e **SCRFD/buffalo_l**: o código do InsightFace é MIT, mas os **pesos
  são licenciados apenas para pesquisa não comercial**.
- **UltraFace**: MIT, via ONNX Model Zoo.
- **Cascades Haar/LBP**: Apache 2.0, do repositório do OpenCV.
- **dlib MMOD**: uso livre, de `dlib.net`.
- **BlazeFace**: Apache 2.0, do MediaPipe.

Para um testbed de pesquisa pessoal nada disso impede o uso. Se o projeto virar
algo publicado ou comercial, o ponto de atenção é o InsightFace.

---

## Fora de escopo

Reconhecimento de **identidade** (dizer *quem* é o rosto, com ArcFace 1:1 ou
1:N). O app faz apenas detecção. A arquitetura do registry deixa espaço para
adicionar isso depois, caso você queira testar se a máscara também quebra a
identificação e não só a detecção.
