import cv2
import numpy as np
import os
import time as t


NUM_QUESTOES = 50
ALTERNATIVAS = 5
COLUNAS = 3
ALTURA_IMAGEM = 1200
LARGURA_IMAGEM = 800
TAMANHO_GAUSS = (5,5)
SIGMA_GAUSS = 1


def titulo() -> None:
    print("--------------------")
    print("         OMR")
    print("--------------------\n")
    

def processar_imagem(imagem):
    # tranforma a imagem em escala de cinza
    imagem_preto_branco = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # aplica gaussian blur para reduzir ruido
    imagem_sem_ruido = cv2.GaussianBlur(imagem_preto_branco, (5, 5), 1)
    
    # detecao de contorno com algoritmo de Canny
    imagem_canny = cv2.Canny(imagem_sem_ruido, 10, 50)
    
    return imagem_canny

#------------ processamento ----------

def ordenaPontosVertice(pontos):
    pontos = pontos.reshape((4, 2))

    # matriz com pontos reordenados
    pontos_aux = np.zeros((4, 1, 2), np.int32)
    soma = pontos.sum(1)

    pontos_aux[0] = pontos[np.argmin(soma)]  # [0,0]
    pontos_aux[3] = pontos[np.argmax(soma)]  # [w,h]
    subtracao = np.diff(pontos, axis=1)
    pontos_aux[1] = pontos[np.argmin(subtracao)]  # [w,0]
    pontos_aux[2] = pontos[np.argmax(subtracao)]  # [h,0]

    return pontos_aux


# Localiza o retangulo que contem 25 questoes conforme o modelo
# da folha de respostas.
# Esta funcao recebe como entrada os contornos da funcao do OpenCV
def localizaRetangulos(contornos):
    retangulos = []

    for i in contornos:
        if cv2.contourArea(i) > 40: # verifica se a area é relevante (se nao se trata de retangulos pequenos) 
            perimetro = cv2.arcLength(i, True) # calcula o perimetro do contorno
            contorno_aproximado = cv2.approxPolyDP(i, 0.02 * perimetro, True) #aproxima o contorno usando a função aprroxPolyDP
            # que reduz o número de pontos do contorno para simplificar sua forma. O parâmetro 0.02*perimetro é a precisão da aproximação

            # Verifica se o contorno aproximado tem 4 pontos. É uma indicação que é um quadrilátero e pode ser um retângulo
            if len(contorno_aproximado) == 4:
                retangulos.append(i) # adiciona à lista de retângulos

    retangulos = sorted(retangulos, key=cv2.contourArea, reverse=True)     # Ordena retângulos pela área em ordem decrescente, contornos com maior área aparecem primeiro

    return retangulos # Retorna a lista de contornos que foram identificados como retângulos.


# localiza os vertices dos retangulos para realizar o recorte de área com as questões
def localizaVertices(contorno):
    perimetro = cv2.arcLength(contorno, True)     # calcula o perimetro da figura     # true indica que é o um contorno fechado
    vertices_aproximados = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)     # realiza uma aproximacao da linha de contorno
    return vertices_aproximados # retorna um array de pontos que representam os vértices do contorno aproximado


# Quebra o bloco de questoes em 25 linha (1 linha por questao)
# e também cada questao em 5 colunas (5 alternativas)
def divideBlocoPorQuestao(imagem):
    questoes = []

    # quebra as questoes em 25 linhas
    linhas = np.vsplit(imagem, 25)

    # quebra cada uma das 25 linhas em 5 colunas(5 alternativas)
    for l in linhas:
        colunas = np.hsplit(l, 5)
        for alternativa in colunas:
            questoes.append(alternativa)
    return questoes


# Processa as questões e salva num vetor a resposta equivalente
# A=0, B=1 ....
def processaQuestoes(retangulo):
    pixels = np.zeros((25, 5))
    coluna = 0  # alternativa
    linha = 0  # questao

    for questao in retangulo:
        # conta o numero de pixels não nulos no corte da alternativa
        total_pixels = cv2.countNonZero(questao)
        pixels[linha][coluna] = total_pixels

        coluna += 1
        if coluna == 5:
            linha += 1
            coluna = 0

    # vetor de respostas
    respostas = []

    for x in range(0, 25):
        aux = pixels[x]

        # verifica se questao possui alternativa assinalada
        # caso a qtde de pixels nao nulos seja menor que 1200 nao possui alternativa assinalada
        limite_minimo = np.amin(aux)*1.5
        if(np.amax(aux) > limite_minimo):
            alternativa_assinalada = np.where(aux == np.amax(aux))

            # ordena para localizar a segunda maior
            aux.sort()

            # compara as duas maiores alternativas, se a segunda maior tiver 80% do valor da proxima
            # considera como duas questoes assinaladas e anula a questao
            if((aux[3]/np.amax(aux)) > 0.8):
                alternativa_assinalada[0][0] = -2
        else:
            alternativa_assinalada[0][0] = -1

        respostas.append(alternativa_assinalada[0][0])

    return respostas


# processa o retangulo fazendo a leitura das questoes
def processaRetantagulo(retangulo, imagem_original, largura, altura):
    if retangulo.size != 0:
        # reordena os pontos do retangulo
        retangulo = ordenaPontosVertice(retangulo)

        # pontos da matriz de transformação para  o retangulo
        pt1 = np.float32(retangulo)
        pt2 = np.float32(
            [[0, 0], [largura, 0], [0, altura], [largura, altura]])

        # cria matriz de transformação do retangulo esquerdo
        matriz = cv2.getPerspectiveTransform(pt1, pt2)

        # aplica transformacao nos retangulos
        retangulo_warp = cv2.warpPerspective(
            imagem_original, matriz, (largura, altura))

        # APLICA O THRESHOLD
        retangulo_warp_gray = cv2.cvtColor(
            retangulo_warp, cv2.COLOR_BGR2GRAY)

        retangulo_thresh = cv2.threshold(
            retangulo_warp_gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

        questoes = divideBlocoPorQuestao(retangulo_thresh)

        return questoes


# processa a imagem e faz a leitura das alternativas salvando em um vetor
def geraVetorResposta(imagem):
    largura = 800
    altura = 1200

    imagem = cv2.resize(imagem, (largura, altura))
    imagem_processada = processar_imagem(imagem)

    # procura os contornos na imagem
    contornos, hierarquia = cv2.findContours(imagem_processada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # procura retangulo
    retangulos = localizaRetangulos(contornos)

    retangulo_esquerdo = localizaVertices(retangulos[0])  # questoes 1 a 25
    retangulo_direito = localizaVertices(retangulos[1])  # questoes 26 a 50

    # questoes do lado esquerdo
    questoes_esquerdo = processaRetantagulo(retangulo_esquerdo, imagem, largura, altura)

    # questos do lado direito
    questoes_direito = processaRetantagulo(retangulo_direito, imagem, largura, altura)

    # concatena os dois retangulos de respostas
    respostas = processaQuestoes(questoes_esquerdo) + processaQuestoes(questoes_direito)

    print("Respostas:", respostas)
    return respostas


def calculaNota(gabarito, respostas):
    qtd_corretas = 0
    index = 1

    for i in range(0, 50):
        if respostas[i] == gabarito[i]:  # se a resposta for igual a do gabarito
            qtd_corretas += 1
        index += 1

    nota = qtd_corretas/5

    return nota, qtd_corretas


def escolher_imagem():
    imagem_aluno = ""
    imagem_gabarito = ""
    
    arquivos_imagens = os.listdir("./Images")
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        titulo()
        print('Arquivos disponíveis:')
        for i, arquivo in enumerate(arquivos_imagens):
            print(f"{i+1}) {arquivo}")

        num_imagem_aluno = int(input("\nDigite o número da imagem para corrigir: "))
                
        if isinstance(num_imagem_aluno, int) and (num_imagem_aluno <= len(arquivos_imagens) and num_imagem_aluno > 0):
            caminho_aluno = os.path.join("./Images", arquivos_imagens[num_imagem_aluno-1])
            break
        else:
            print('A escolha precisa estar nas opções acima!')
            t.sleep(2)
                        
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        titulo()
        print('Arquivos disponíveis:')
        for i, arquivo in enumerate(arquivos_imagens):
            print(f"{i+1}) {arquivo}")

        num_imagem_gabarito = int(input("\nDigite o número da imagem para o gabarito: "))
                
        if isinstance(num_imagem_gabarito, int) and (num_imagem_gabarito <= len(arquivos_imagens) and num_imagem_gabarito > 0):
            caminho_gabarito = os.path.join("./Images", arquivos_imagens[num_imagem_gabarito-1])
            break
        else:
            print('A escolha precisa estar nas opções acima!')
            t.sleep(2)
    
    imagem_aluno = cv2.imread(caminho_aluno)
    imagem_gabarito = cv2.imread(caminho_gabarito)
    #cv2.imshow("Prova ALuno", imagem_aluno)
    #cv2.waitKey(0)
    return imagem_aluno, imagem_gabarito


def main():
    # carrega imagem
    imagem_gabarito, imagem_aluno = escolher_imagem()

    # processa a imagem e transforma em um vetor
    gabarito = geraVetorResposta(imagem_gabarito)
    aluno = geraVetorResposta(imagem_aluno)

    # calcula a nota, numero de acertos
    nota, acertos = calculaNota(gabarito, aluno)
    print("Você acertou", acertos, "/", NUM_QUESTOES, "e sua nota foi :", nota)

if __name__ == '__main__':
    main()