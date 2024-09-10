import cv2 as cv2
import numpy as np
import os as os
import time as t


NUM_QUESTOES = 50
NUM_ALTERNATIVAS = 5
ALTERNATIVAS = {"N/A":-2, "NULL":-1, "A":0, "B":1, "C":2, "D":3, "E":4}
DISTRIBUICAO_QUESTOES = [25, 25]
ALTURA_IMAGEM = 1200
LARGURA_IMAGEM = 800
TAMANHO_GAUSS = (5,5)
SIGMA_GAUSS = 1


def titulo() -> None:
    print("--------------------")
    print("         OMR")
    print("--------------------\n")


def escolher_imagem() -> tuple[cv2.Mat, cv2.Mat]:
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
    return imagem_aluno, imagem_gabarito


def processar_imagem(imagem_original:cv2.Mat) -> cv2.Mat:
    imagem_preto_branco = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    imagem_sem_ruido = cv2.GaussianBlur(imagem_preto_branco, TAMANHO_GAUSS, SIGMA_GAUSS)
    imagem_canny = cv2.Canny(imagem_sem_ruido, 10, 50)
    return imagem_canny


def localizar_retangulos(contornos:cv2.Mat) -> list:
    retangulos = []

    for i in contornos:
        if cv2.contourArea(i) > 40:
            perimetro = cv2.arcLength(i, True)
            contorno_aproximado = cv2.approxPolyDP(i, 0.02 * perimetro, True)

            if len(contorno_aproximado) == 4:
                retangulos.append(i)

    retangulos = sorted(retangulos, key=cv2.contourArea, reverse=True)
    return retangulos


def localizar_vertices(contorno:cv2.Mat) -> cv2.Mat:
    perimetro = cv2.arcLength(contorno, True)
    vertices_aproximados = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
    return vertices_aproximados


def ordenar_pontos_vertices(pontos:cv2.Mat) -> cv2.Mat:
    pontos = pontos.reshape((4, 2))

    pontos_aux = np.zeros((4, 1, 2), np.int32)
    soma = pontos.sum(1)

    pontos_aux[0] = pontos[np.argmin(soma)]
    pontos_aux[3] = pontos[np.argmax(soma)]
    subtracao = np.diff(pontos, axis=1)
    pontos_aux[1] = pontos[np.argmin(subtracao)]
    pontos_aux[2] = pontos[np.argmax(subtracao)]

    return pontos_aux


def dividir_coluna_por_questao(imagem:cv2.Mat, num_questoes_coluna:int) -> list:
    questoes = []

    linhas = np.vsplit(imagem, num_questoes_coluna)

    for l in linhas:
        colunas = np.hsplit(l, NUM_ALTERNATIVAS)
        for alternativa in colunas:
            questoes.append(alternativa)
    return questoes


def processar_retangulo(retangulo:cv2.Mat, imagem_original:cv2.Mat, num_questoes_coluna:int) -> list:
    if retangulo.size != 0:
        retangulo = ordenar_pontos_vertices(retangulo)

        pt1 = np.float32(retangulo)
        pt2 = np.float32([[0, 0], [LARGURA_IMAGEM, 0], [0, ALTURA_IMAGEM], [LARGURA_IMAGEM, ALTURA_IMAGEM]])

        matriz = cv2.getPerspectiveTransform(pt1, pt2)
        retangulo_warp = cv2.warpPerspective(imagem_original, matriz, (LARGURA_IMAGEM, ALTURA_IMAGEM))
        retangulo_warp_gray = cv2.cvtColor(retangulo_warp, cv2.COLOR_BGR2GRAY)
        retangulo_thresh = cv2.threshold(retangulo_warp_gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
        questoes = dividir_coluna_por_questao(retangulo_thresh, num_questoes_coluna)
        return questoes
    

def processar_questoes(retangulo:list) -> list:
    pixels = np.zeros((25, 5))
    alternativa = 0
    num_questao = 0

    for questao in retangulo:
        total_pixels = cv2.countNonZero(questao)
        pixels[num_questao][alternativa] = total_pixels

        alternativa += 1
        if alternativa == 5:
            num_questao += 1
            alternativa = 0

    respostas = []
    for x in range(0, 25):
        aux = pixels[x]
        limite_minimo = np.amin(aux) * 1.5
        if(np.amax(aux) > limite_minimo):
            alternativa_assinalada = np.where(aux == np.amax(aux))
            aux.sort()
            if((aux[3]/np.amax(aux)) > 0.8):
                alternativa_assinalada[0][0] = -2
        else:
            alternativa_assinalada[0][0] = -1

        respostas.append(alternativa_assinalada[0][0])

    return respostas


def gerar_resposta(imagem:cv2.Mat) -> list:
    imagem = cv2.resize(imagem, (LARGURA_IMAGEM, ALTURA_IMAGEM))
    imagem_processada = processar_imagem(imagem)

    contornos, hierarquia = cv2.findContours(imagem_processada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retangulos = localizar_retangulos(contornos)

    respostas = []
    for i in range(0, len(DISTRIBUICAO_QUESTOES)):
        retangulo_coluna = localizar_vertices(retangulos[i])
        questoes_coluna = processar_retangulo(retangulo_coluna, imagem, DISTRIBUICAO_QUESTOES[i])
        respostas += processar_questoes(questoes_coluna)

    return respostas


def calcular_nota(gabarito:list, respostas:list) -> tuple[float, int]:
    qtd_corretas = 0

    for i in range(0, NUM_QUESTOES):
        if respostas[i] == gabarito[i]:
            qtd_corretas += 1

    nota = round((float(qtd_corretas)/NUM_QUESTOES)*10, 1)

    return nota, qtd_corretas


def main() -> None:
    imagem_gabarito, imagem_aluno = escolher_imagem()
    gabarito = gerar_resposta(imagem_gabarito)
    aluno = gerar_resposta(imagem_aluno)

    os.system('cls' if os.name == 'nt' else 'clear')
    titulo()
    print("Gabarito:", gabarito)
    print("Respostas:", aluno)
    
    nota, acertos = calcular_nota(gabarito, aluno)
    print("Você acertou", acertos, "/", NUM_QUESTOES, "e sua nota foi :", nota)


if __name__ == '__main__':
    main()