import cv2 as cv2
import numpy as np
import os as os
import time as t


class Imagem_OMR():
    def __init__(self, eh_gabarito = False):
        self.__NUM_QUESTOES = 50
        self.__NUM_ALTERNATIVAS = 5
        self.__ALTERNATIVAS = {"N/A":-2, "NULL":-1, "A":0, "B":1, "C":2, "D":3, "E":4}
        self.__DISTRIBUICAO_QUESTOES = [25,25]
        self.__ALTURA_IMAGEM = 1200
        self.__LARGURA_IMAGEM = 800
        self.__TAMANHO_GAUSS = (5,5)
        self.__SIGMA_GAUSS = 1
        
        self.__caminho = None
        self.__imagem = None
        self.__imagem_formatada = None
        
        self.__respostas = []
        self.__acertos = 0
        self.__nota = 0
        self.__gabarito = eh_gabarito
        
        # carrega imagem
        self.escolher_imagem()

        # processa a imagem e transforma em um vetor
        self.gerar_resposta()
    
    
    def __del__(self):
        self.__respostas.clear()
        self.__acertos = 0
        self.__nota = 0
        self.__gabarito = False
        #os.system('cls' if os.name == 'nt' else 'clear')
    
    
    def titulo(self) -> None:
        print("--------------------")
        print("         OMR")
        print("--------------------\n")


    def escolher_imagem(self) -> None:
        arquivos_imagens = os.listdir("./Images")
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.titulo()
            print('Arquivos disponíveis:')
            for i, arquivo in enumerate(arquivos_imagens):
                print(f"{i+1}) {arquivo}")

            if self.__gabarito:
                num_imagem = int(input("\nDigite o número da imagem para ser o gabarito: "))
            else:
                num_imagem = int(input("\nDigite o número da imagem para corrigir: "))
                    
            if isinstance(num_imagem, int) and (num_imagem <= len(arquivos_imagens) and num_imagem > 0):
                self.__caminho = os.path.join("./Images", arquivos_imagens[num_imagem-1])
                break
            else:
                print('A escolha precisa estar nas opções acima!')
                t.sleep(2)
        
        self.__imagem = cv2.imread(self.__caminho)


    def processar_imagem(self) -> None:
        # tranforma a imagem em escala de cinza
        imagem_preto_branco = cv2.cvtColor(self.__imagem, cv2.COLOR_BGR2GRAY)
        
        # aplica gaussian blur para reduzir ruido
        imagem_sem_ruido = cv2.GaussianBlur(imagem_preto_branco, self.__TAMANHO_GAUSS, self.__SIGMA_GAUSS)
        
        # detecao de contorno com algoritmo de Canny
        self.__imagem_formatada = cv2.Canny(imagem_sem_ruido, 10, 50)


    # Localiza o retangulo que contem 25 questoes conforme o modelo
    # da folha de respostas.
    # Esta função recebe como entrada os contornos da funcao do OpenCV
    def localizar_retangulos(self, contornos:cv2.Mat) -> list:
        retangulos = []

        for i in contornos:
            if cv2.contourArea(i) > 40:                                                 # verifica se a area é relevante (se nao se trata de retangulos pequenos) 
                perimetro = cv2.arcLength(i, True)                                      # calcula o perimetro do contorno
                contorno_aproximado = cv2.approxPolyDP(i, 0.02 * perimetro, True)       # aproxima o contorno usando a função aprroxPolyDP
                # que reduz o número de pontos do contorno para simplificar sua forma. O parâmetro 0.02*perimetro é a precisão da aproximação

                # Verifica se o contorno aproximado tem 4 pontos. É uma indicação que é um quadrilátero e pode ser um retângulo
                if len(contorno_aproximado) == 4:
                    retangulos.append(i)                                                # adiciona à lista de retângulos

        retangulos = sorted(retangulos, key=cv2.contourArea, reverse=True)              # Ordena retângulos pela área em ordem decrescente, contornos com maior área aparecem primeiro

        return retangulos


    # localiza os vertices dos retangulos para realizar o recorte de área com as questões
    def localizar_vertices(self, contorno:cv2.Mat) -> cv2.Mat:
        perimetro = cv2.arcLength(contorno, True)                                       # calcula o perimetro da figura     # true indica que é o um contorno fechado
        vertices_aproximados = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)       # realiza uma aproximacao da linha de contorno
        return vertices_aproximados                                                     # retorna um array de pontos que representam os vértices do contorno aproximado


    def ordenar_pontos_vertices(self, pontos:cv2.Mat) -> cv2.Mat:
        pontos = pontos.reshape((4, 2))

        # matriz com pontos reordenados
        pontos_aux = np.zeros((4, 1, 2), np.int32)
        soma = pontos.sum(1)

        pontos_aux[0] = pontos[np.argmin(soma)]         # [0,0]
        pontos_aux[3] = pontos[np.argmax(soma)]         # [largura,altura]
        subtracao = np.diff(pontos, axis=1)
        pontos_aux[1] = pontos[np.argmin(subtracao)]    # [largura,0]
        pontos_aux[2] = pontos[np.argmax(subtracao)]    # [altura,0]

        return pontos_aux


    def dividir_coluna_por_questao(self, imagem:cv2.Mat, num_questoes_coluna:int) -> list:
        questoes = []

        # quebra as questoes em 25 linhas
        linhas = np.vsplit(imagem, num_questoes_coluna)

        # quebra cada uma das 25 linhas em 5 colunas(5 alternativas)
        for l in linhas:
            colunas = np.hsplit(l, self.__NUM_ALTERNATIVAS)
            for alternativa in colunas:
                questoes.append(alternativa)
        return questoes


    # processa o retangulo fazendo a leitura das questoes
    def processar_retantagulo(self, retangulo:cv2.Mat, imagem_original:cv2.Mat, num_questoes_coluna:int) -> list:
        if retangulo.size != 0:
            # reordena os pontos do retangulo
            retangulo = self.ordenar_pontos_vertices(retangulo)

            # pontos da matriz de transformação para o retangulo
            pt1 = np.float32(retangulo)
            pt2 = np.float32([[0, 0], [self.__LARGURA_IMAGEM, 0], [0, self.__ALTURA_IMAGEM], [self.__LARGURA_IMAGEM, self.__ALTURA_IMAGEM]])

            # cria matriz de transformação do retangulo esquerdo
            matriz = cv2.getPerspectiveTransform(pt1, pt2)

            # aplica transformacao nos retangulos
            retangulo_warp = cv2.warpPerspective(imagem_original, matriz, (self.__LARGURA_IMAGEM, self.__ALTURA_IMAGEM))

            # APLICA O THRESHOLD
            retangulo_warp_gray = cv2.cvtColor(retangulo_warp, cv2.COLOR_BGR2GRAY)

            retangulo_thresh = cv2.threshold(retangulo_warp_gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

            questoes = self.dividir_coluna_por_questao(retangulo_thresh, num_questoes_coluna)

            return questoes
        

    # Processa as questões e salva num vetor a resposta equivalente
    def processar_questoes(self, retangulo:list) -> list:
        pixels = np.zeros((25, 5))
        alternativa = 0
        num_questao = 0

        for questao in retangulo:
            # conta o número de pixels não nulos no corte da alternativa
            total_pixels = cv2.countNonZero(questao)
            pixels[num_questao][alternativa] = total_pixels

            alternativa += 1
            if alternativa == 5:
                num_questao += 1
                alternativa = 0

        respostas = []
        for x in range(0, 25):
            aux = pixels[x]

            # verifica se questao possui alternativa assinalada
            # caso a qtde de pixels nao nulos seja menor que 1200 nao possui alternativa assinalada
            limite_minimo = np.amin(aux) * 1.5
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


    # processa a imagem e faz a leitura das alternativas salvando em um vetor
    def gerar_resposta(self) -> None:
        self.__imagem = cv2.resize(self.__imagem, (self.__LARGURA_IMAGEM, self.__ALTURA_IMAGEM))
        self.processar_imagem()

        # procura os contornos na imagem
        contornos, hierarquia = cv2.findContours(self.__imagem_formatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # procura retangulo
        retangulos = self.localizar_retangulos(contornos)

        for i in range(0, len(self.__DISTRIBUICAO_QUESTOES)):
            retangulo_coluna = self.localizar_vertices(retangulos[i])
            questoes_coluna = self.processar_retantagulo(retangulo_coluna, self.__imagem, self.__DISTRIBUICAO_QUESTOES[i])
            self.__respostas += self.processar_questoes(questoes_coluna)



class Respostas():
    def __init__(self, gabarito:Imagem_OMR, aluno:Imagem_OMR):
        self.__gabarito = gabarito
        self.__aluno = aluno
    
    
    def __del__(self):
        self.__gabarito.__del__()
        self.__aluno.__del__()
        #os.system('cls' if os.name == 'nt' else 'clear')
    
    
    def calcular_nota(self, gabarito:Imagem_OMR, aluno:Imagem_OMR) -> None:
        for i in range(0, gabarito.__NUM_QUESTOES):
            if aluno.__respostas[i] == gabarito.__respostas[i]:             # se a resposta for igual a do gabarito
                aluno.__acertos += 1

        aluno.__nota = round((float(aluno.__acertos)/gabarito.__NUM_QUESTOES)*10, 1)


    def main(self) -> None:
        # calcula a nota, numero de acertos
        self.calcular_nota(self.__gabarito, self.__aluno)
        print("Você acertou", self.__aluno.__acertos, "/", self.__gabarito.__NUM_QUESTOES, "e sua nota foi :", self.__aluno.__nota)


if __name__ == '__main__':
    gabarito = Imagem_OMR(True)
    aluno = Imagem_OMR(False)
    omr = Respostas(gabarito, aluno)
    omr.main()